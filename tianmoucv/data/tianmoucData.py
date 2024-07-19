import numpy as np
import cv2,sys
import torch
import math,time,subprocess
import torch.nn.functional as F
import os
flag = True
try:
    from tianmoucv.rdp_usb import rod_decoder_py as rdc
except:
    print("WARNING: no decoder found, try to compile it under ./rod_decoder_py")
    current_file_path = os.path.abspath(__file__)
    parent_folder_path = os.path.dirname(os.path.dirname(current_file_path))
    aim_path = os.path.join(parent_folder_path,'rdp_usb')
    os.chdir(aim_path)
    current_path = os.getcwd()
    print("Current Path:", current_path)
    subprocess.run(['sh', './compile_pybind.sh'])
    from tianmoucv.rdp_usb import rod_decoder_py as rdc
    print('compile decoder successfully')
    print('If you still get this message,please try:\n 1. run it in a python script (only once) \n 2. use source code install to see what happened')
  
    
#用于重建
from tianmoucv.proc.reconstruct import poisson_blending
from tianmoucv.isp import default_rgb_isp
from .tianmoucData_basic import TianmoucDataReader_basic
from .tianmoucDataSampleParser import TianmoucDataSampleParser

class TianmoucDataReader(TianmoucDataReader_basic):
    '''
    - TianmoucDataReader(version 0.3.5.6 in dev)
        - Data structure
            ├── dataset
            │   ├── matchkey(sample name)
            │   │   ├── cone
            │   │       ├── info.txt
            │   │       ├── xxx.tmdat
            │   │   ├── rod
            │   │       ├── info.txt
            │   │       ├── xxx.tmdat
            │   │   ├── cone1
            │   │       ├── info.txt
            │   │       ├── xxx.tmdat
            │   │   ├── rod1
            │   │       ├── info.txt
            │   │       ├── xxx.tmdat
        -  @dataPath：
            - a path to a sample/dataset: string
            - a list of dataset path/sample: [string1, string2]   
            - if the path is belong to a dataset/datasets or a list of samples, please use matchkey to index a certain sample, or set matchkey = None
            - if you use multiple camera, set camera_idx
        - @matchkey：
            - matchkey should be unified, is the only key to select data sample
            - matchkey is the folder name containing the cop/aop data folder
        - @N: number of continous samples, N=1 as default
            - the returned sample will contain (N+1) COP，and N*M+1 AOP
            - for example N=1, in 750@8bit, M=750/30=25, M = 25, a sample contain 2 COP and 26 AOP
        - @camera_idx：
            - camera_idx = 0 as default
            - in multi-camera system, use camera_idx to create a dataset fot certain camera
        - @MAXLEN:
            - the maximum read data length, set -1 to read all data, used only in training task
        [Output]
        create a dataset, the containings please refer to TianmoucDataReader.__getitem__()

    '''
    def __init__(self,path,
                 N=1,
                 camera_idx= 0,
                 showList=True,
                 MAXLEN=-1,
                 matchkey=None,
                 cachePath=None,
                 ifcache = False, 
                 speedUpRate=1,
                 ifUniformSampling = False,
                 print_info=True,
                 training=True,
                 strict = True,
                 use_data_parser = False,
                 dark_level = 0):
        
        self.N = N
        self.use_data_parser = use_data_parser
        super().__init__(path,
                 showList=showList,
                 MAXLEN=MAXLEN,
                 camera_idx= camera_idx,
                 matchkey=matchkey,
                 cachePath=cachePath,
                 ifcache = ifcache, 
                 speedUpRate=speedUpRate,
                 ifUniformSampling = ifUniformSampling,
                 print_info=print_info,
                 training=training,
                 strict = strict,
                 dark_level = dark_level) # 调用父类的属性赋值方法
        

    #你可以重写这个extration逻辑以获得更复杂的数据读取方法，例如抽帧等        
    def extraction(self,rate,MAXLEN,ifUniformSampling):
        for key in self.fileDict:
            pass
            #wait for debug
            if self.print_info:
                new_legalFileList = []
                legalFileList = self.fileDict[key]['legalData'] 
                #把相邻N个同步后的包合并为一个sample来读取
                newsample_merge = dict([])
                accum_count = 1
                for sampleid in range(len(legalFileList)-1):
                    sample_0 = legalFileList[sampleid]
                    sample_1 = legalFileList[sampleid+1]
                    cone1 = sample_0['coneid']
                    cone2 = sample_1['coneid']

                    if accum_count == 1:
                        newsample_merge['sysTimeStamp'] = sample_0['sysTimeStamp']
                        newsample_merge['coneid'] = cone1
                        newsample_merge['rodid'] = sample_0['rodid']
                        newsample_merge[self.pathways[1]] = sample_0[self.pathways[1]]
                        newsample_merge[self.pathways[0]] = sample_0[self.pathways[0]]
                        newsample_merge['labels'] = sample_0['labels'] 
                    if accum_count == self.N:
                        new_legalFileList.append(newsample_merge)
                        newsample_merge = dict([])
                        accum_count = 1
                        continue
                    #拼接更多的sample，以待一起读取
                    if cone1[1]==cone2[0]:
                        newsample_merge['coneid'] += cone2[1:]
                        newsample_merge['rodid'] += sample_1['rodid'][1:]
                        newsample_merge[self.pathways[1]] += sample_1[self.pathways[1]][1:]
                        newsample_merge[self.pathways[0]] += sample_1[self.pathways[0]][1:]
                        accum_count += 1
                    else:
                        newsample_merge = dict([])
                        accum_count = 1
                        continue
                
                if len(new_legalFileList)>MAXLEN:
                    new_legalFileList = new_legalFileList[:MAXLEN]
                    
                self.fileDict[key]['legalData']  = new_legalFileList
                print(key,'origin length:',len(new_legalFileList))

    #同理，你也可以通过修改这个函数获得更复杂的数据预处理手段
    def packRead(self,idx,key,ifSync =True, needPreProcess = True):
        '''
        use the decoder and isp preprocess to generate a paired (RGB,n*TSD) sample dict:
        
            - COP
                - COP is stored seprately as F0，F1，F2 ... FN+1, use string key to get them
                - COP's framerate is 30.3fps
                - i = 0~N
                - sample['Fi_without_isp'] = only demosaced frame data, without addtional isp 3*320*640, t=t_0+i*T
                - sample['Fi_HDR']: RGB+SD Blended HDR frame data, 3*320*640, t=t_0+i*T ms
                - sample['Fi']: preprocessed frame data, 3*320*640, t=t_0+i*T ms
            - AOP
                - AOP data is stored in a large tensor, [T,C,H,W]
                - channel 0 is TD, channel 1 is SDL, channel 2 is SDR
                - sample['tsdiff'] = TSD data upsample to (N+1)*3*320*640, from t=t_0 to t=t+ N*T ms (eg. T=33 in 757 @ 8 bit  mode)
                - sample['rawDiff']: raw TSD data, (N+1)*3*160*160, from t=t_0 to t=t+ N*T ms (eg. T=33 in 757 @ 8 bit  mode)
                - all tianmoucv API use sample['rawDiff'] for input, sample['tsdiff'] is used by some NN-based method
            - sample['meta']: 
                - a dict
                - file path infomation 
                - camera timestamps for each data
                - more other details
            - sample['labels']: 
                - list of labels, if you have one
            - sample['sysTimeStamp']: unix system time stamp in us, use for multi-sensor sync, -1 if not supported
                - you can calculate Δt = sysTimeStamp1-sysTimeStamp2, unit is 'us'
        '''       
        sample = dict([])
        metaInfo = dict([])
        legalSample = self.fileDict[key]['legalData'][idx]
        conefilename = self.fileDict[key][self.pathways[1]] # only read first one, if you want to use dual camera you can read it again
        rodfilename  = self.fileDict[key][self.pathways[0]]  # only read first one, if you want to use dual camera you can read it again
        coneAddrs = legalSample[self.pathways[1]]
        rodAddrs = legalSample[self.pathways[0]]
        
        rgb_list = []
        coneTimeStamp_list = []
        raw_list = []

        for i in range(self.N+1):
            #print('caddr:',coneAddrs[i])
            frame,timestamp = self.readConeFast(conefilename,coneAddrs[i])
            frame_raw = np.reshape(frame.copy(), (self.cone_height,self.cone_width))
            frame = np.reshape(frame.astype(np.float32),(self.cone_height,self.cone_width))
            raw_list.append(frame_raw)
            rgb_list.append(frame)
            coneTimeStamp_list.append(timestamp.astype(np.int64))

        metaInfo['C_name'] = conefilename
        metaInfo['C_timestamp'] = coneTimeStamp_list
        metaInfo['R_name'] = rodfilename
        metaInfo['R_timestamp'] = []
        metaInfo['key'] = key
        metaInfo['sample_length'] = len(self.fileDict[key]['legalData'])
        
        itter = len(rodAddrs)
        if itter<0:
            print('>>>>>>>>>>>>>WARNING:',key,coneStartId, cone_id, coneRange)
            print('>>>>>>>>>>>>>WARNING:',itter , rodAddrs[itter-1] , rodAddrs[0])
            return None
        
        tsd = torch.zeros([3,itter,self.rod_height,self.rod_width])

        for i in range(itter):
            startAddr = rodAddrs[i]
            sdl,sdr,td,rodTimeStamp = self.readRodFast(rodfilename,startAddr)
            tsd[0,i,:,:] = torch.Tensor(td.astype(np.float32)).view(self.rod_height,self.rod_width)
            tsd[1,i,:,:] = torch.Tensor(sdl.astype(np.float32)).view(self.rod_height,self.rod_width)
            tsd[2,i,:,:] = torch.Tensor(sdr.astype(np.float32)).view(self.rod_height,self.rod_width)
            metaInfo['R_timestamp'].append(rodTimeStamp.astype(np.int64))

        sample['rawDiff'] = tsd
        mingap = (itter-1)//self.N
        if needPreProcess:
            tsdiff_inter  = self.tsd_preprocess(tsd)
            tsdiff_resized = F.interpolate(tsdiff_inter,(320,640),mode='bilinear')
            sample['tsdiff'] = tsdiff_resized
            
            for i in range(self.N+1): 
                frame,frame_without_isp = self.rgb_preprocess(rgb_list[i])
                frame_raw = raw_list[i]
                sample['F'+str(i)+'_without_isp'] = frame_without_isp
                sample['F'+str(i)] = frame
                sample['F' + str(i)+"_raw"] = frame_raw
                SD_t = tsd[1:,mingap*i,...]
                sample['F'+str(i)+'_HDR'] = self.HDRRecon(SD_t / 128,frame)
        sample['meta'] = metaInfo
        sample['labels'] = legalSample['labels']
        sample['sysTimeStamp'] = legalSample['sysTimeStamp']
        
        dataRatio = self.fileDict[key]['dataRatio']
        sample['dataRatio']= dataRatio
        return sample
    
    def tsd_preprocess(self,tsdiff):
        return self.upsampleTSD_conv(tsdiff)/128.0   
    
    def rgb_preprocess(self,F_raw):
        F,F_without_isp = default_rgb_isp(F_raw,blc=self.blc)
        return F,F_without_isp
 
    def __getitem__(self, index):
        #定位在哪个sample里
        key,relativeIndex = self.locateSample(index)
        sample = self.packRead(relativeIndex, key)
        if self.use_data_parser:
            sample = TianmoucDataSampleParser(sample)
        '''
        for key in sample:
            print(key)
            if isinstance(sample[key],torch.Tensor) or isinstance(sample[key],np.ndarray):
                print(sample[key].shape)
            else:
                print(sample[key])        
        '''
        return sample