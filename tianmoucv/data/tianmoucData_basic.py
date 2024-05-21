import numpy as np
import os
import cv2,sys
import torch
import math,time,subprocess
import torch.nn.functional as F

try:
    from tianmoucv.rdp_usb import rod_decoder_py as rdc
except:
    print("FATAL ERROR: no decoder found, please complie the decoder under ./rod_decoder_py")

#用于重建
#用于rgb的ISP
from tianmoucv.proc.reconstruct import laplacian_blending
from tianmoucv.isp import default_rgb_isp,SD2XY,ACESToneMapping

from ctypes import *

flag = True

class TianmoucDataReader_basic():
    '''
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
    def __init__(self,path,showList=True,
                 MAXLEN=-1,
                 matchkey=None,
                 cachePath=None,
                 ifcache = False, 
                 speedUpRate=1,
                 ifUniformSampling = False,
                 print_info=True,
                 training=True,
                 strict = True,
                 camera_idx= 0,
                 dark_level = None):
        self.print_info = print_info

        modedict = {0:'para',1:'lvds'}
        self.rod_height = 160
        self.rod_width = 160
        self.showList = showList
        self.cone_height = 320
        self.cone_width = 320
        self.rodFileSize = 0
        self.sampleNumDict = dict([])
        self.sampleNum = 0
        self.EXT="tmdat"

        self.blc = 0
        if os.path.exists(dark_level):
            if dark_level.split('.')[-1] == 'npz':
                self.blc =  np.load(dark_level)['blc']
            if dark_level.split('.')[-1] == 'npy':
                self.blc =  np.load(dark_level)

        self.training = training

        # These buffers are used for fast data decoding
        # Rod buffer
        self.temp_diff_np = np.zeros((1, self.rod_width * self.rod_height), dtype=np.int8)
        self.spat_diff_left_np = np.zeros((1, self.rod_width * self.rod_height), dtype=np.int8)
        self.spat_diff_right_np = np.zeros((1, self.rod_width * self.rod_height), dtype=np.int8)
        self.pkt_size_np = np.zeros([1], dtype=np.int32)
        self.pkt_size_td = np.zeros([1], dtype=np.int32)
        self.pkt_size_sd = np.zeros([1], dtype=np.int32) 
        self.rod_img_timestamp_np = np.zeros([1], dtype=np.uint64)
        self.rod_fcnt_np = np.zeros([1], dtype=np.int32)
        self.rod_adcprec_np = np.zeros([1], dtype=np.int32)
        #cone buffer
        self.cone_raw = np.zeros((1, self.cone_width * self.cone_height), dtype=np.int16)
        self.c_img_timestamp_np = np.zeros([1], dtype=np.uint64)
        self.c_fcnt_np = np.zeros([1], dtype=np.int32)
        self.c_adcprec_np = np.zeros([1], dtype=np.int32)
        self.error = False

        if camera_idx == 0:
            self.pathways = ['rod','cone']
        else:
            self.pathways = ['rod_'+str(camera_idx),'cone_'+str(camera_idx)]
                    
        # Use a .npy file to store the single sample address in a .tmdat file
        # It will save large amount of time, but be sure to clear it if you use a same path to store different data
        if (not cachePath is None and os.path.exists(cachePath)) and matchkey is None:
            datalist = np.load(cachePath, allow_pickle=True)
            self.fileDict = datalist.item()
            if self.print_info:
                print("--->data loader: succesfully read cached file:",cachePath)
        else:
            self.fileDict = dict([])
            self.addMoreSample(self.fileDict,path,matchkey=matchkey,strict = strict,camera_idx= 0)
            if ifcache:
                print('making cache file...')
                np.save(cachePath, self.fileDict)
        
        #Scan all samples to generate a file list, use speedUpRate to control the sparsity
        self.extraction(speedUpRate,MAXLEN,ifUniformSampling)
            
        # print the dataset information
        for key in self.fileDict:
            self.sampleNumDict[key] =  self.dataNum(key)
            self.sampleNum += self.dataNum(key)
            if self.print_info:
                print(key[0:20],'..., --- containing [RGB,NxTSD] sample:',self.dataNum(key),' packs')
                
        if self.sampleNum == 0:
            print("ERROR: no data found! please check your data path and sample key")
            print(path,matchkey)
            
    #用递归的方法获取所有tmdat结尾的数据
    def find_all_tmdat_file(self, top_path, fileDict, camera_idx= 0, matchkey=None):

        if not os.path.exists(top_path):
            if self.print_info:
                print(top_path, 'does not exsist')
            return None

        if os.path.isdir(top_path):
            file_list = os.listdir(top_path)
            #null folder
            if len(file_list) == 0:
                return None
            #如果不是空文件夹
            else:
                check_pw_completeness = True
                #如果缺少pathway或者文件数量不正确，标记为错误
                for pw in self.pathways:
                    if not pw in file_list:
                        check_pw_completeness = False
                    else:
                        pw_path = os.path.join(top_path, pw)
                        raw_file_list = os.listdir(pw_path)
                        if len(raw_file_list)==1:#如果数据原始文件存在且仅有一个
                            raw_data_file_path = os.path.join(pw_path, raw_file_list[0])
                            if not (os.path.isfile(raw_data_file_path) and raw_data_file_path.split('.')[-1]=='tmdat'):
                                check_pw_completeness = False
                                print(raw_data_file_path,':not exsists')

                #如果数据通路文件夹存在且符合规律,说明探到底层，终止并返回
                if check_pw_completeness:
                    sp_top_path = top_path.split(os.sep)
                    dataset_top =  os.path.join(*sp_top_path[:-1])
                    sample_name = sp_top_path[-1]
                    sample_key = sample_name+'@'+dataset_top
                    us_value = 0
                    if (matchkey is None) or sample_name == matchkey:
                        for pw in self.pathways:
                            pw_path = os.path.join(top_path, pw)
                            raw_file_list = os.listdir(pw_path)
                            tmdat_file_list = []
                            for file in raw_file_list:
                                if file.split('.')[-1]=='tmdat':
                                    tmdat_file_list.append(file)
                                    
                                #临时的处理方案，用系统时间戳做对齐
                                try:
                                    if pw == self.pathways[1] and file.split('.')[-1]=='txt':
                                        extra_info_path = os.path.join(pw_path, file)
                                        with open(extra_info_path, 'r') as txtfile:
                                            first_line = txtfile.readline()
                                            first_line = first_line.replace('us', '')
                                            first_line = first_line.replace('s', '')
                                            first_line = first_line.replace(',', '')
                                            first_line = first_line.replace('\n', '')
                                            us_value = int(first_line)
                                except:
                                    us_value = -1
                                        
                            if len(tmdat_file_list)==1:#如果数据原始文件存在且仅有一个
                                raw_data_file_path = os.path.join(pw_path, tmdat_file_list[0])
                                if not sample_key in fileDict:
                                    fileDict[sample_key] = {}    
                                fileDict[sample_key][pw] = raw_data_file_path
                                fileDict[sample_key]['sysTimeStamp'] = us_value
                    return None     
                else:
                    for file in file_list:
                        self.find_all_tmdat_file(os.path.join(top_path, file), fileDict, camera_idx= camera_idx, matchkey=matchkey)
                    return None     
        
    def addMoreSample(self,fileDict,dataset_top, matchkey=None, strict = True, camera_idx= 0):
        '''
        scan one sample to generate a sync-ed (RGB_n,RGB_{n+1},TSD_n,TSD_{n+m}) pair list
        @dataset_top: 数据集顶层路径
        dataset_top
            - sample 1
            - sample 2
             - rod
             - cone
             - rod_1
             - cone_1
             ...
            - sample n
        '''
        # find all legal sample list
        if isinstance(dataset_top,list):
            for path in dataset_top:
                self.find_all_tmdat_file(path,fileDict, camera_idx= camera_idx, matchkey=matchkey)
        elif isinstance(dataset_top,str):
            self.find_all_tmdat_file(dataset_top,fileDict, camera_idx= camera_idx, matchkey=matchkey)
        else:
            print('dataset_top:',dataset_top,' is not list or string')

            
        keylist = [key for key in fileDict]
        # for each sample, extract all paired data id

        for key in keylist:
            labels = []
            labelFileName = os.path.join(key.split('@')[1],key.split('@')[0],'label.csv')
            if os.path.exists(labelFileName):
                with open(labelFileName, mode='r', newline='') as file:
                    csv_reader = csv.reader(file)
                    for row in csv_reader:
                        labels.append(row)
            
            #for each sample
            coneFileRawFile = fileDict[key][self.pathways[1]]
            rodFileRawFile = fileDict[key][self.pathways[0]]
            systimeStamp = fileDict[key]['sysTimeStamp']
            rodfilepersample = 25
            
            try:
                rod_tmdat_path = '/'+os.path.join(*rodFileRawFile.split('/')[:-1])
                rod_tmdat_path_list = os.listdir(rod_tmdat_path)
                for meta_file_name in rod_tmdat_path_list:
                    if meta_file_name.split('.')[-1]=='txt':
                        with open(os.path.join(rod_tmdat_path,meta_file_name), mode='r', newline='') as file:
                            lines = file.readlines()
                            line = lines[1]
                            # 提取第二行中的四个变量的值
                            values = line.split(',')
                            exp_time = int(values[0].split(':')[1].split(' ')[0].strip())
                            gain = int(values[1].split(':')[1].strip())
                            rod_mode = int(values[2].split(':')[1].strip())
                            rod_adc_precision = int(values[3].split(':')[1].strip())
    
                            if rod_adc_precision == 8 and rod_mode == 0:
                                rodfilepersample = 50
                            if rod_adc_precision == 8 and rod_mode == 1:
                                rodfilepersample = 25
                            if rod_adc_precision == 4 and rod_mode == 0:
                                rodfilepersample = 110
                            if rod_adc_precision == 4 and rod_mode == 1:
                                rodfilepersample = 50    
                            if rod_adc_precision == 2 and rod_mode == 0:
                                rodfilepersample = 330
                            if rod_adc_precision == 2 and rod_mode == 1:
                                rodfilepersample = 110
            except:
                #旧文件，默认都是25或者自适应读取
                pass

                                        
            rodTimeList = []
            coneTimeList = []
            rodcntList = []
            conecntList = []
                
            #use decoder with C++ backend, generate frame-based data from tmdat compact stream
            rodAddrlist =rdc.construct_frm_list(rodFileRawFile,rodTimeList,rodcntList)
            coneAddrlist =rdc.construct_frm_list(coneFileRawFile, coneTimeList, conecntList)
                
            # find the corresponding addr bias for each single sample in tmdat
            rodAddrlist = [c_uint64(e).value for e in rodAddrlist]
            coneAddrlist = [c_uint64(e).value for e in coneAddrlist]  
            rodTimeList = [c_uint64(e).value for e in rodTimeList]
            coneTimeList = [c_uint64(e).value for e in coneTimeList]

            legalFileList = []    
            cStartID = 0
            search_index = 0
                
            # find first RGB with possible matched TSD
            while(rodTimeList[0] > coneTimeList[cStartID]):
                cStartID += 1

            # Sync each RGB sample with a list of TSD
            for cidx in range(cStartID,len(coneTimeList)-1):
                ct1 = coneTimeList[cidx]
                ct2 = coneTimeList[cidx+1]
                ridx1 = 0
                ridx2 = 0
                #find first rod/tsd data
                for ridx in range(search_index,len(rodTimeList)):
                    rt1 = rodTimeList[ridx]
                    if rt1 < ct1 - 5:
                        continue
                    else:
                        search_index = ridx
                        ridx1 = ridx
                        break
                #find last rod/tsd data
                for ridx in range(search_index,len(rodTimeList)):
                    rt2 = rodTimeList[ridx]
                    if rt2 < ct2-5:
                        continue    
                    else:
                        search_index = ridx
                        ridx2 = ridx
                        break
                            
                # check if satisfy the sample rate 
                flag = False
                if strict:
                    flag = (ridx2 - ridx1 == rodfilepersample)
                else:
                    flag = ridx2 > ridx1
                        
                # if it is legal, build a data dict for one pack
                if flag:
                    legalSample=dict([])
                    legalSample['sysTimeStamp'] = systimeStamp
                    legalSample['coneid'] = [i for i in range(cidx,cidx+2)]
                    legalSample['rodid']  = [i for i in range(ridx1,ridx2+1)]
                    legalSample[self.pathways[1]] = [coneAddrlist[i] for i in range(cidx,cidx+2)]
                    legalSample[self.pathways[0]] = [rodAddrlist[i] for i in range(ridx1,ridx2+1)]
                    if len(labels) == 4:
                        legalSample['labels'] = labels
                    else:
                        legalSample['labels'] = [['HDR', '0'], ['HS', '0'], 
                                             ['Blur', '0'], ['Noisy', '0']]
                        
                    legalFileList.append(legalSample)
                else:
                    if self.print_info:
                        print('recoreded rod:',ridx2 - ridx1,' expected:',rodfilepersample)
                        print('if you wish to read all data neglecting data loss, set strict=False for datareader')
                    continue
                    
            fileDict[key]['legalData'] = legalFileList
            fileDict[key]['dataRatio'] = rodfilepersample

        return fileDict  
              
    
    def extraction(self,rate,MAXLEN,ifUniformSampling):
        '''
        cut the MAXLEN
        '''
        print('>>> @rate paramter is not implemented in thie version')
        for key in self.fileDict:
            if self.print_info:
                legalFileList = self.fileDict[key]['legalData'] 
                if len(self.fileDict[key]['legalData'])>MAXLEN:
                    self.fileDict[key]['legalData']  = self.fileDict[key]['legalData'][:MAXLEN]


    # read a rod file directly
    def readRodFast(self,rodFileRawFile,startAddr):
        '''
        AOP/ROD/TD+SD decoding
        '''
        ret_code = rdc.get_one_rod_fullinfo(rodFileRawFile, startAddr,
                                                            self.temp_diff_np, 
                                                            self.spat_diff_left_np, 
                                                            self.spat_diff_right_np,
                                                            self.pkt_size_np, 
                                                            self.pkt_size_td, 
                                                            self.pkt_size_sd,
                                                            self.rod_img_timestamp_np, 
                                                            self.rod_fcnt_np, 
                                                            self.rod_adcprec_np,
                                                            self.rod_height, 
                                                            self.rod_width)
            
        rodtimeStamp = self.rod_img_timestamp_np[0]
        return self.spat_diff_left_np.copy(), self.spat_diff_right_np.copy(),self.temp_diff_np.copy(),rodtimeStamp
          

    def readConeFast(self,coneFileRawFile,startAddr):
        '''
        COP/Cone/RAW-BGGR decoding
        '''
        ret_code = rdc.get_one_cone_fullinfo(coneFileRawFile, startAddr,
                                                            self.cone_raw,
                                                            self.c_img_timestamp_np, 
                                                            self.c_fcnt_np, 
                                                            self.c_adcprec_np,
                                                            self.cone_height, 
                                                            self.cone_width)
        return self.cone_raw.copy(), self.c_img_timestamp_np[0]

    def packRead(self,idx,key,ifSync =True, needPreProcess = True):
        '''
        use the decoder and isp preprocess to generate a paired (RGB,n*TSD) sample dict:
        
            - sample['tsdiff'] = TSD data upsample to 320*640
            - sample['rawDiff']: raw TSD data, N*3*160*160, from t=t_0 to t=t+ T ms (T=33 in 757 @ 8 bit  mode)
            - sample['F0_without_isp'] = only demosaced frame data, 3*320*640, t=t_0
            - sample['F1_without_isp'] = only demosaced frame data, 3*320*640, t=t_0
            - sample['F0_HDR']: RGB+SD Blended HDR frame data, 3*320*640, t=t_0
            - sample['F1_HDR']: RGB+SD Blended HDR frame data, 3*320*640, t=t_0+T ms
            - sample['F0']: preprocessed frame data, 3*320*640, t=t_0
            - sample['F1']: preprocessed frame data, 3*320*640, t=t_0+T ms
            - sample['meta']: path infomation and and timestamps for each data
            - sample['labels']: list of labels, if you have one
            - sample['sysTimeStamp']: system time stamp in us, use for multi-sensor sync
        '''       
        sample = dict([])
        metaInfo = dict([])
        legalSample = self.fileDict[key]['legalData'][idx]
        conefilename = self.fileDict[key][self.pathways[1]]
        rodfilename  = self.fileDict[key][self.pathways[0]]
        dataRatio = self.fileDict[key]['dataRatio']
        
        coneAddrs = legalSample[self.pathways[1]]
        rodAddrs = legalSample[self.pathways[0]]
        
        start_frame_raw,coneTimeStamp1 = self.readConeFast(conefilename,coneAddrs[0])
        end_frame_raw,coneTimeStamp2 = self.readConeFast(conefilename,coneAddrs[1])
        
        start_frame_raw = np.reshape(start_frame_raw.astype(np.float32),(self.cone_height,self.cone_width))
        end_frame_raw = np.reshape(end_frame_raw.astype(np.float32),(self.cone_height,self.cone_width))
        
        metaInfo['C_name'] = conefilename
        metaInfo['C_timestamp'] = (coneTimeStamp1.astype(np.int64),coneTimeStamp2.astype(np.int64))
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
        SD_0 = 0
        SD_1 = 0
        
        for i in range(itter):
            startAddr = rodAddrs[i]
            sdl,sdr,td,rodTimeStamp = self.readRodFast(rodfilename,startAddr)
            metaInfo['R_timestamp'].append(rodTimeStamp.astype(np.int64))
            tsd[0,i,:,:] = torch.Tensor(td.astype(np.float32)).view(self.rod_height,self.rod_width)
            tsd[1,i,:,:] = torch.Tensor(sdl.astype(np.float32)).view(self.rod_height,self.rod_width)
            tsd[2,i,:,:] = torch.Tensor(sdr.astype(np.float32)).view(self.rod_height,self.rod_width)
            if i == 0:
                SD_0 = tsd[1:,i,...]
            if i == itter - 1:
                SD_1 = tsd[1:,i,...]
            
        if needPreProcess:
            start_frame,end_frame,tsdiff_inter,F0_without_isp,F1_without_isp  = self.preprocess(start_frame_raw,end_frame_raw,tsd)
            #sample['tsdiff_160x320'] = tsdiff_inter
            tsdiff_resized = F.interpolate(tsdiff_inter,(320,640),mode='bilinear')
            sample['tsdiff'] = tsdiff_resized
            sample['F0_without_isp'] = F0_without_isp
            sample['F1_without_isp'] = F1_without_isp
            sample['F0_HDR'] = self.HDRRecon(SD_0/128.0,start_frame)
            sample['F1_HDR'] = self.HDRRecon(SD_1/128.0,end_frame)
        sample['F0'] = start_frame
        sample['F1'] = end_frame
        sample['rawDiff'] = tsd
        sample['meta'] = metaInfo
        sample['labels'] = legalSample['labels']
        sample['sysTimeStamp'] = legalSample['sysTimeStamp']
        sample['dataRatio']= dataRatio
        return sample
    
    def HDRRecon(self,SD,F0):
        '''
        HDR fusion
        '''
        F0 = torch.Tensor(F0)
        Ix,Iy= SD2XY(SD)#0-1
        Ix = F.interpolate(torch.Tensor(Ix).unsqueeze(0).unsqueeze(0), size=(320,640), mode='bilinear').squeeze(0).squeeze(0)
        Iy = F.interpolate(torch.Tensor(Iy).unsqueeze(0).unsqueeze(0), size=(320,640), mode='bilinear').squeeze(0).squeeze(0)
        blend_hdr = laplacian_blending(-Ix,-Iy, srcimg=F0, iteration=20, mask_rgb=True, mask_th = 32)
        return blend_hdr
    
    def preprocess(self,F0_raw,F1_raw,tsdiff):
        '''
        use isp in TIANMOUCV
        '''
        ts = time.time()
        F0,F0_without_isp = default_rgb_isp(F0_raw,blc=self.blc)
        F1,F1_without_isp = default_rgb_isp(F1_raw,blc=self.blc)
        te1 = time.time()
        tsdiff_inter = self.upsampleTSD_conv(tsdiff)/128.0
        te2 = time.time()
        return F0,F1,tsdiff_inter,F0_without_isp,F1_without_isp

    def upsampleTSD_conv(self,tsdiff):
        '''
        adjust the data space and upsampling, please refer to tianmoucv doc for detail
        '''
        # 获取输入Tensor的维度信息
        h,w = tsdiff.shape[-2:]
        w *= 2
        tsdiff_expand = torch.zeros([*tsdiff.shape[:-2],h,w])
        tsdiff_expand[...,::2,::2] = tsdiff[...,::2,:]
        tsdiff_expand[...,1::2,1::2] = tsdiff[...,1::2,:]
        channels, T, height, width = tsdiff_expand.size()
        input_tensor = tsdiff_expand.view(channels*T, height, width).unsqueeze(0)
        # 定义卷积核
        kernel = torch.zeros(1, 1, 3, 3)
        kernel[:, :, 1, 0] = 1/4
        kernel[:, :, 1, 2] = 1/4
        kernel[:, :, 0, 1] = 1/4
        kernel[:, :, 2, 1] = 1/4
        # 对输入Tensor进行反射padding
        padded_tensor = F.pad(input_tensor, (1, 1, 1, 1), mode='reflect')
        # 将原tensor复制一份用于填充结果
        output_tensor = input_tensor.clone()

        # 将卷积结果填充回原tensor
        for c in range(channels*T):
            #print(padded_tensor[0,:8,:8])
            output = F.conv2d(padded_tensor[:,c:c+1,...], kernel, padding=0)
            #print(output[0,:8,:8])
            output_tensor[:, c:c+1, 0:-1:2, 1:-1:2] = output[:, :, 0:-1:2, 1:-1:2]
            output_tensor[:, c:c+1, 1:-1:2, 0:-1:2] = output[:, :, 1:-1:2, 0:-1:2]
            #print(output_tensor[0,:8,:8])
        return output_tensor[0,...].view(channels, T, height, width)

    
    def dataNum(self,key):
        return len(self.fileDict[key]['legalData'])
        
    def __len__(self):
        return self.sampleNum 
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        for key in self.sampleNumDict:
            fmt_str = key + ' sample cell num:' + self.sampleNumDict[key] + '\n'
        return fmt_str
    
    def locateSample(self,index):
        '''
        local the vedio sample and index bias
        '''
        relativeIndex = index
        
        key = None
        for key in self.sampleNumDict:
            numKey = self.sampleNumDict[key]
            #print(relativeIndex,numKey)
            if relativeIndex >= numKey:
                relativeIndex -= numKey
            else:
                return key,relativeIndex
        return key,relativeIndex
    
    def __getitem__(self, index):
        #定位在哪个sample里
        key,relativeIndex = self.locateSample(index)
        if key is None:
            print(">>>>>>>>>>>>>>>[Data reader ERROR] no data found! please check your data path and sample key")
            return None
        sample = self.packRead(relativeIndex, key)
        return sample