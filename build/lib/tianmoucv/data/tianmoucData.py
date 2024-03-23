import numpy as np
import os
import cv2,sys
import math,time

import torch
import torch.nn.functional as F

try:
    from tianmoucv.rod_decode_pybind_usb import rod_decoder_py as rdc
except:
    print("FATAL ERROR: no decoder found, please complie the decoder under ./rod_decoder_py")

#用于重建
from tianmoucv.isp import fourdirection2xy,laplacian_blending
#用于rgb的ISP
from tianmoucv.isp import default_rgb_isp

flag = True

from ctypes import *

class TianmoucDataReader():
    '''
        - pathList:一个装有多个数据集的list，用以合并数据集，如果只有一个可以输入
        ;格式:[数据集路径1,数据集路径2,...]
        - showList:打印出来看看有哪些数据
        - MAXLEN:接受多少个输入作为训练数据
        - matchkey:是否匹配某个具体样本名，无输入则使用所有数据
        - savedFileDict:提前存储的数据目录
        - ifSaveFileDict:是否存下所有数据目录
        - speedUpRate:数据加速压缩比例

        输出数据是F0,F1，以及25*speedUpRate 帧ROD 连续
    
    逻辑:
    
        1. 对每个rod，找到向下最接近的cone
        2. 对这个cone，存向上的所囊括的rod的索引
        3. 读取数据的时候，可以选择度哪个rod以及其对应的cone，也可以用另一个逻辑
        4. 索引文件可以保存下来，只要key存在就不用再算一遍
        5. 抽帧逻辑，合并N个sample，保留所有的TDSD，去掉中间的COP帧
    '''
    def __init__(self,pathList,showList=True,
                 MAXLEN=-1,
                 matchkey=None,
                 savedFileDict=None,
                 ifSaveFileDict = False, 
                 speedUpRate=1,
                 ifUniformSampling = False,
                 print_info=True,
                 strict = True):
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
        self.rodfilepersample = 0

        #rod buffer
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
                    
        #'/filedict.npy'
        if savedFileDict is not None and os.path.exists(savedFileDict) and matchkey is None:
            datalist = np.load(savedFileDict, allow_pickle=True)
            self.fileDict = datalist.item()
            if self.print_info:
                print("--->data loader: succesfully read saved file list:",savedFileDict)
        else:
            self.fileDict = dict([])
            self.fileList = []
            for DataTop in pathList:
                self.addMoreSample(self.fileDict,DataTop,matchkey,strict=strict)
            if ifSaveFileDict:
                np.save(savedFileDict, self.fileDict)

        self.extraction(speedUpRate,MAXLEN,ifUniformSampling)
            
        for key in self.fileDict:
            self.sampleNumDict[key] =  self.dataNum(key)
            self.sampleNum += self.dataNum(key)
            if self.print_info:
                print(key,'---legal sample num:',self.dataNum(key))
            

    def addMoreSample(self,fileDict,dataset_top,matchkey,strict = True):

        signals= ['rod','cone']
        ext=".tmdat"
        # find all legal sample list
        if not os.path.isdir(dataset_top):
            print(dataset_top, 'is not a directory')
            return fileDict
        
        # add all legal sample key
        for folderlist in os.listdir(dataset_top):
            if matchkey is not None and folderlist!=matchkey:
                continue
            firstSave = True
            folder_root = os.path.join(dataset_top, folderlist)
            for sg in signals:
                try:
                    fileList = []
                    dataListRoot = os.path.join(folder_root, sg)
                    for fl in os.listdir(dataListRoot):
                        flpath = os.path.join(dataListRoot, fl)
                        if (os.path.isfile(flpath) and flpath[-6:]==ext):
                            fileList.append(flpath)
                    if len(fileList)>0:
                        if firstSave == True:
                            fileDict[folderlist+'@'+dataset_top] = {}
                            fileDict[folderlist+'@'+dataset_top][sg] = fileList
                            firstSave = False
                        else:
                            fileDict[folderlist+'@'+dataset_top][sg] = fileList
                    elif folderlist in fileDict:
                        fileDict.pop(folderlist)
                        break
                except:
                    continue
                    print('bad entry for ',sg,' in ',folder_root)
                    
        #sort files if there are multiple samples in one key(little possibility)
        keylist = [key for key in fileDict]
        for key in keylist:
            for sg in signals:
                fileDict[key][sg] = sorted(fileDict[key][sg])
                if len(fileDict[key][sg])>1:
                    print('warning! multiple files in single sample dir ',key,' and it may cause wrong results')
                print(fileDict[key][sg])
                
        # for each sample, extract all paired data id
        for key in keylist:

            labels = []
            labelFileName = os.path.join(key.split('@')[1],key.split('@')[0],'label.csv')
            if os.path.exists(labelFileName):
                with open(labelFileName, mode='r', newline='') as file:
                    # 读取CSV文件内容
                    csv_reader = csv.reader(file)
                    for row in csv_reader:
                        labels.append(row)
                        print(labels,row)
            else:
                print('labelFileName:',labelFileName,'doesnot have status label')
                
            coneFileList = fileDict[key]['cone']
            rodFileList = fileDict[key]['rod']
            for i in range(len(rodFileList)):
                coneFileRawFile = coneFileList[i]
                rodFileRawFile = rodFileList[i]
                rodTimeList = []
                coneTimeList = []
                rodcntList = []
                conecntList = []

                rodAddrlist =rdc.construct_frm_list(rodFileRawFile,rodTimeList,rodcntList)
                coneAddrlist =rdc.construct_frm_list(coneFileRawFile, coneTimeList, conecntList)
                
                rodAddrlist = [c_uint64(e).value for e in rodAddrlist]
                coneAddrlist = [c_uint64(e).value for e in coneAddrlist]  
                rodTimeList = [c_uint64(e).value for e in rodTimeList]
                coneTimeList = [c_uint64(e).value for e in coneTimeList]

                legalFileList = []    
                cStartID = 0
                search_index = 0
                
                while(rodTimeList[0] > coneTimeList[cStartID]):
                    cStartID += 1

                for cidx in range(cStartID,len(coneTimeList)-1):
                    ct1 = coneTimeList[cidx]
                    ct2 = coneTimeList[cidx+1]
                    #print('-------------ct1,ct2:',ct1,ct2)
                    ridx1 = 0
                    ridx2 = 0
                    #find first rod
                    for ridx in range(search_index,len(rodTimeList)):
                        rt1 = rodTimeList[ridx]
                        #print('rt1:',rt1,ridx)
                        if rt1 < ct1 - 5:
                            continue
                        else:
                            search_index = ridx
                            ridx1 = ridx
                            #print('ridx1:',ridx1)
                            break
                    #print('search_index:',search_index)
                    for ridx in range(search_index,len(rodTimeList)):
                        rt2 = rodTimeList[ridx]
                        #print('rt2:',rt2,ridx)
                        if rt2 < ct2-5:
                            continue    
                        else:
                            search_index = ridx
                            ridx2 = ridx
                            #print('ridx2:',ridx2)
                            break
                    
                    flag = False
                    if strict:
                        flag = (ridx2 - ridx1 == 25)
                    else:
                        flag = ridx2 > ridx1
                        
                    if flag:
                        legalSample=dict([])
                        legalSample['coneid'] = [i for i in range(cidx,cidx+2)]
                        legalSample['rodid']  = [i for i in range(ridx1,ridx2+1)]
                        legalSample['cone'] = [coneAddrlist[i] for i in range(cidx,cidx+2)]
                        legalSample['rod']  = [rodAddrlist[i] for i in range(ridx1,ridx2+1)]
                        if len(labels) == 4:
                            legalSample['labels'] = labels
                        else:
                            legalSample['labels'] = [['HDR', '0'], ['HS', '0'], 
                                             ['Blur', '0'], ['Noisy', '0']]
                        
                        legalFileList.append(legalSample)
                    else:
                        continue
                        '''
                        legalSample['meta'] = {'key':key,
                                               'C0':coneListSorted[coneID],
                                               'C1':coneListSorted[coneID+1],
                                               'R0':rodListSorted[rodRange[0]//self.rodfilepersample],
                                               'R0_bias:':rodRange[0]%self.rodfilepersample,
                                               'R1':rodListSorted[rodRange[0]//self.rodfilepersample],
                                               'R1_bias:':rodRange[0]%self.rodfilepersample,
                                               'C_time': (conetimestamp1,conetimestamp2),
                                               'R_time': (rodtimestamp1,rodtimestamp2),
                                               'C/R':self.aopcoprate,
                                               'RpF':self.rodfilepersample}
                        '''

                fileDict[key]['legalData'] = legalFileList
                fileDict[key]['RpF'] = self.rodfilepersample

        return fileDict  
              
    def extraction(self,rate,MAXLEN,ifUniformSampling):
        for key in self.fileDict:
            pass
            #wait for debug
            if self.print_info:
                legalFileList = self.fileDict[key]['legalData'] 
                if len(self.fileDict[key]['legalData'])>MAXLEN:
                    self.fileDict[key]['legalData']  = self.fileDict[key]['legalData'][:MAXLEN]
                print(key,'origin length:',len(legalFileList))

    # read a rod file directly
    def readRodFast(self,rodFileRawFile,startAddr):
        addr = np.zeros([1], dtype=np.uint64)
        addr[0] = startAddr
        #print(startAddr)
        ret_code = rdc.get_one_rod_fullinfo(rodFileRawFile, addr,
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
        ret_code = rdc.get_one_cone_fullinfo(coneFileRawFile, startAddr,
                                                            self.cone_raw,
                                                            self.c_img_timestamp_np, 
                                                            self.c_fcnt_np, 
                                                            self.c_adcprec_np,
                                                            self.cone_height, 
                                                            self.cone_width)
        return self.cone_raw.copy(), self.c_img_timestamp_np[0]

    def packRead(self,idx,key,ifSync =True, needPreProcess = True):
        
        sample = dict([])
        metaInfo = dict([])
        legalSample = self.fileDict[key]['legalData'][idx]
        conefilename = self.fileDict[key]['cone'][0]
        rodfilename  = self.fileDict[key]['rod'][0]
        coneAddrs = legalSample['cone']
        rodAddrs = legalSample['rod']

        start_frame,coneTimeStamp1 = self.readConeFast(conefilename,coneAddrs[0])
        end_frame,coneTimeStamp2 = self.readConeFast(conefilename,coneAddrs[1])
        
        start_frame = np.reshape(start_frame.astype(np.float32),(self.cone_height,self.cone_width))
        end_frame = np.reshape(end_frame.astype(np.float32),(self.cone_height,self.cone_width))
        
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
            start_frame,end_frame,tsdiff_inter,F0_without_isp,F1_without_isp  = self.preprocess(start_frame,end_frame,tsd)
            sample['tsdiff_160x320'] = tsdiff_inter
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
        return sample
    
    def HDRRecon(self,SD,F0):
        F0 = torch.Tensor(F0)
        Ix,Iy= fourdirection2xy(SD)
        Ix = F.interpolate(torch.Tensor(Ix).unsqueeze(0).unsqueeze(0), size=(320,640), mode='bilinear').squeeze(0).squeeze(0)
        Iy = F.interpolate(torch.Tensor(Iy).unsqueeze(0).unsqueeze(0), size=(320,640), mode='bilinear').squeeze(0).squeeze(0)
        blend_hdr = laplacian_blending(-Ix,-Iy, srcimg=F0,iteration=20, mask_rgb=True)
        return blend_hdr
    
    def preprocess(self,F0_raw,F1_raw,tsdiff):
        
        ts = time.time()
        F0,F0_without_isp = default_rgb_isp(F0_raw)
        F1,F1_without_isp = default_rgb_isp(F1_raw)
        te1 = time.time()
        tsdiff_inter = self.upsampleTSD_conv(tsdiff)/128.0
        te2 = time.time()

        return F0,F1,tsdiff_inter,F0_without_isp,F1_without_isp

    def upsampleTSD_conv(self,tsdiff):
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
        relativeIndex = index
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