import numpy as np
import cv2,sys
import torch
import math,time
import torch.nn.functional as F
import os
flag = True
try:
    from tianmoucv.rdp_usb import rod_decoder_py as rdc
except:
    print("FATAL ERROR: no decoder found, please complie the decoder under ./rod_decoder_py")
#用于重建
from tianmoucv.isp import laplacian_blending
from tianmoucv.isp import default_rgb_isp,fourdirection2xy
from .tianmoucData_basic import TianmoucDataReader_basic

class TianmoucDataReader(TianmoucDataReader_basic):
    '''
     继承datareader，一次性读多帧
        - N:一次性读取N帧COP成为一个片段
        - path: string或者string的列表，会自动扫描其下所有的tmdat sample
        - showList:是否打印信息
        - MAXLEN:每个sample的最大长度，防止超长sample干扰训练
        - matchkey:是否匹配某个sample name
        - cachePath:缓存目录，None则每次重新构建数据集
        - ifcache:是否存下所有数据的地址，方便大规模数据集下次读取
        - speedUpRate:数据稀疏采样
                *这部分处理不会被存储
        输出数据是F0,F1,...,FN，F0_HDR,F1_HDR,...,FN_HDR，以及25*N*speedUpRate 帧ROD 连续
        存储在一个字典里，上述名称为key
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
                 rodfilepersample = 25):
        
        self.N = N
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
                 rodfilepersample = rodfilepersample) # 调用父类的属性赋值方法
        
        print('tianmoucData_multiple.py TODO：add overlap parameter')
            
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
                
                if len(new_legalFileList)>MAXLEN:
                    new_legalFileList = new_legalFileList[:MAXLEN]
                    
                self.fileDict[key]['legalData']  = new_legalFileList
                print(key,'origin length:',len(new_legalFileList))

    #同理，你也可以通过修改这个函数获得更复杂的数据预处理手段
    def packRead(self,idx,key,ifSync =True, needPreProcess = True):
        '''
        use the decoder and isp preprocess to generate a paired (RGB,n*TSD) sample dict:
        
            - sample['tsdiff_160x320'] = RAW TSD data ajusted to coorect space(with hollow)
            - sample['tsdiff'] = TSD data upsample to 320*640
            - sample['F0_without_isp'] = only demosaced frame data, 3*320*640, t=t_0
            - sample['F1_without_isp'] = only demosaced frame data, 3*320*640, t=t_0
            - sample['F0_HDR']: RGB+SD Blended HDR frame data, 3*320*640, t=t_0
            - sample['F1_HDR']: RGB+SD Blended HDR frame data, 3*320*640, t=t_0+33ms
            - sample['F0']: preprocessed frame data, 3*320*640, t=t_0
            - sample['F1']: preprocessed frame data, 3*320*640, t=t_0+33ms
            - sample['rawDiff']: raw TSD data, N*3*160*160, from t=t_0 to t=t+33ms
            - sample['meta']: path infomation and and timestamps for each data
            - sample['labels']: list of labels, if you have one
            - sample['sysTimeStamp']: system time stamp in us, use for multi-sensor sync
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
        
        for i in range(self.N+1):
            #print('caddr:',coneAddrs[i])
            frame,timestamp = self.readConeFast(conefilename,coneAddrs[i])
            frame = np.reshape(frame.astype(np.float32),(self.cone_height,self.cone_width))
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
            sample['tsdiff_160x320'] = tsdiff_inter
            tsdiff_resized = F.interpolate(tsdiff_inter,(320,640),mode='bilinear')
            sample['tsdiff'] = tsdiff_resized
            
            for i in range(self.N+1): 
                frame,frame_without_isp = self.rgb_preprocess(rgb_list[i])
                sample['F'+str(i)+'_without_isp'] = frame_without_isp
                sample['F'+str(i)] = frame
                SD_t = tsd[1:,mingap*i,...]
                sample['F'+str(i)+'_HDR'] = self.HDRRecon(SD_t/128.0,frame)
        sample['meta'] = metaInfo
        sample['labels'] = legalSample['labels']
        sample['sysTimeStamp'] = legalSample['sysTimeStamp']
        return sample
    
    def tsd_preprocess(self,tsdiff):
        return self.upsampleTSD_conv(tsdiff)/128.0   
    
    def rgb_preprocess(self,F_raw):
        F,F_without_isp = default_rgb_isp(F_raw)
        return F,F_without_isp
 
    def __getitem__(self, index):
        #定位在哪个sample里
        key,relativeIndex = self.locateSample(index)
        sample = self.packRead(relativeIndex, key)
        '''
        for key in sample:
            print(key)
            if isinstance(sample[key],torch.Tensor) or isinstance(sample[key],np.ndarray):
                print(sample[key].shape)
            else:
                print(sample[key])        
        '''
        return sample