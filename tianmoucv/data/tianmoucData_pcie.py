import numpy as np
import os
import struct
import cv2,sys
import matplotlib.pyplot as plt
import torch
import math,time
import torch.nn.functional as F

import csv

flag = True
sys.path.append("../")
try:
    from tianmoucv.rdp_pcie import rod_decoder_py as rdc
except:
    import subprocess
    print("WARNING: no decoder found, try to compile under ./rod_decoder_py")
    current_file_path = os.path.abspath(__file__)
    parent_folder_path = os.path.dirname(os.path.dirname(current_file_path))
    aim_path = os.path.join(parent_folder_path,'rdp_pcie')
    os.chdir(aim_path)
    current_path = os.getcwd()
    print("Current Path:", current_path)
    subprocess.run(['sh', './compile_pybind.sh'])
    from tianmoucv.rdp_pcie import rod_decoder_py as rdc
    print('compile decoder successfully')

    
#用于色彩调整
from tianmoucv.isp import fourdirection2xy,default_rgb_isp
from tianmoucv.proc.reconstruct import laplacian_blending

adc_bit_prec = 8
width = 160
height = 160
#interface =
ROD_8B_ONE_FRM = 0x9e00       #158KB * 1024 / 4;//0x9e00
ROD_4B_ONE_FRM = 0x4D00 
ROD_2B_ONE_FRM = 0x1D00

_DEBUG = False

def findDataNum(filename):
    data_info = filename.split('/')[-1]
    data_info = data_info.split('.')[0]
    num_stamp = data_info.split('_')
    return int(num_stamp[0])

def findDataTimestamp(filename):
    data_info = filename.split('/')[-1]
    data_info = data_info.split('.')[0]
    num_stamp = data_info.split('_')
    return int(num_stamp[1])
    
def getCorrectRodSize(nbit,nfile):
    if nbit == 2:
        return ROD_2B_ONE_FRM*nfile * 4 
    if nbit == 4:
        return ROD_4B_ONE_FRM*nfile * 4  
    if nbit == 8:
        return ROD_8B_ONE_FRM*nfile * 4 
    

class TianmoucDataReader_pcie():
    '''
        - pathList:一个装有多个数据集的list，用以合并数据集，如果只有一个可以输入;
            格式:[数据集路径1,数据集路径2,...]
        - showList:打印出来看看有哪些数据
        - MAXLEN:接受多少个输入作为训练数据
        - matchkey:是否匹配某个具体样本名，无输入则使用所有数据
        - savedFileDict:提前存储的数据目录
        - ifSaveFileDict:是否存下所有数据目录
        - speedUpRate:数据加速压缩比例

        输出数据是F0,F1，以及25*speedUpRate 帧ROD 连续
        
    **注意**
    
        PCIE版本数据需要预先输入相机模式
        @parameter rod_adc_bit= 2,4,8 为AOP的工作模式
        @cameraMode cameraMode = 0,1 分为parallel(0) or lvds(1)
    
    逻辑:
    
        1. 对每个rod，找到向下最接近的cone
        2. 对这个cone，存向上的所囊括的rod的索引
        3. 读取数据的时候，可以选择度哪个rod以及其对应的cone，也可以用另一个逻辑
        4. 索引文件可以保存下来，只要key存在就不用再算一遍
        5. 抽帧逻辑，合并N个sample，保留所有的TDSD，去掉中间的COP帧
    '''
    def __init__(self,pathList,showList=True,
                 rod_adc_bit=8,
                 cameraMode=0, #parallel(0) or lvds(1)
                 MAXLEN=-1,
                 matchkey=None,
                 savedFileDict=None,
                 ifSaveFileDict = False, 
                 speedUpRate=1,
                 ifUniformSampling = False,
                 print_info=True):
        self.print_info = print_info

        modedict = {0:'para',1:'lvds'}
        if self.print_info:
            print('CAMERA MODE SET: \n (1)rod_adc_bit =',rod_adc_bit,
                  '\n (2)cameraMode  =',modedict[cameraMode],
                  '\n (3)matchkey    =',matchkey)
        
        
        self.rod_height = 160
        self.rod_width = 160
        self.showList = showList
        self.cone_height = 320
        self.cone_width = 320
        self.rod_adc_bit = rod_adc_bit
        
        if rod_adc_bit == 8 and cameraMode==0:
            self.rodfilepersample = 2
            self.aopcoprate = 25
            self.rodInterval = 131
        if rod_adc_bit == 8 and cameraMode==1:
            self.rodfilepersample = 3
            self.aopcoprate = 50
            self.rodInterval = 65           
        if rod_adc_bit == 4 and cameraMode==0:
            self.rodfilepersample = 2
            self.aopcoprate = 50
            self.rodInterval = 65    
        if rod_adc_bit == 4 and cameraMode==1:
            self.rodfilepersample = 4
            self.aopcoprate = 100
            self.rodInterval = 33    
        if rod_adc_bit == 2 and cameraMode==0:
            self.rodfilepersample = 3
            self.aopcoprate = 110
            self.rodInterval = 33    
        if rod_adc_bit == 2 and cameraMode==1:
            self.rodfilepersample = 8
            self.aopcoprate = 330
            self.rodInterval = 10     
            
        self.coneInterval = 3333
        self.rodFileSize = 0
        self.sampleNumDict = dict([])
        self.sampleNum = 0
        self.rod_adc_bit = rod_adc_bit
        
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
                self.addMoreSample(self.fileDict,DataTop,matchkey)
            if ifSaveFileDict:
                np.save(savedFileDict, self.fileDict)

        self.extraction(speedUpRate,MAXLEN,ifUniformSampling)
            
        for key in self.fileDict:
            self.sampleNumDict[key] =  self.dataNum(key)
            self.sampleNum += self.dataNum(key)
            if self.print_info:
                print(key,'---legal sample num:',self.dataNum(key))
            
    #input:
    #   @fileDict：
    #    --- key [sample name]
    #         sample name : sampleName+'@'+sampleGroup
    #         --- 'rod' : a list of sorted rod file name
    #         --- 'cone': a list of sorted cone file name
    #         --- 'legalData': a list of legal data meta information
    #output:
    #  @fileDict
    #     update this dict
    def addMoreSample(self,fileDict,dataset_top,matchkey):
        if self.print_info:
            print(self.rodfilepersample,'rod sample per file')
        signals= ['rod','cone']
        ext=".bin"
        # find all legal sample list
        if not os.path.isdir(dataset_top):
            print(dataset_top, 'is not a directory')
            return fileDict
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
                        if (os.path.isfile(flpath) and flpath[-4:]==ext):
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
        
        # save all legal samples
        keylist = [key for key in fileDict]
        # sort the aop and cop sample in terms of the time stamp
        for key in keylist:
            for sg in signals:
                list2 = sorted(fileDict[key][sg],key=findDataNum)
                fileDict[key][sg] = list2
        for key in keylist:
            
            labels = []
            labelFileName = os.path.join(key.split('@')[1],key.split('@')[0],'label.csv')
            if os.path.exists(labelFileName):
                with open(labelFileName, mode='r', newline='') as file:
                    # 读取CSV文件内容
                    csv_reader = csv.reader(file)
                    for row in csv_reader:
                        labels.append(row)
                        #print(labels,row)
            else:
                print('labelFileName:',labelFileName,'doesnot have status label')
                                
            legalFileList = []
            # delete all decrepted sample
            try:
                rodListSorted = fileDict[key]['rod']
                coneListSorted = fileDict[key]['cone']
            except:
                fileDict.pop(key)
                print('key:',key,' has some problems, popped')
                continue
                
            # (0) Check self.rodfilepersample, 检查文件是否出错
            MINTIMEINTERVAL = self.rodInterval
            if len(rodListSorted) > 20:
                rt1 = findDataTimestamp(rodListSorted[20])
                rt0 = findDataTimestamp(rodListSorted[10])
                MINTIMEINTERVAL = (rt1-rt0)/(10*self.rodfilepersample)
                if abs(self.rodInterval-MINTIMEINTERVAL)>5 and self.print_info:
                    print('**WARNING** Your file may be damaged:',MINTIMEINTERVAL,self.rodInterval)
                self.rodfilepersample = round((rt1-rt0)/(10*self.rodInterval)) 
                self.rodFileSize = getCorrectRodSize(self.rod_adc_bit,self.rodfilepersample)
                if self.rodFileSize != os.path.getsize(rodListSorted[20]) and self.print_info:
                    print('**WARNING** Your file may be damaged: correct size:',self.rodFileSize,\
                          ' rod size:',os.path.getsize(rodListSorted[20]))
                
            if _DEBUG:
                print('MINTIMEINTERVAL:',MINTIMEINTERVAL,' self.rodfilepersample:',self.rodfilepersample)
            rodSearchStart = 0
            for coneID in range(len(coneListSorted)-1):
                # create a new legal sample
                legalSample = dict([])
                rodtimestamp1 = -500
                rodtimestamp2 = -500
                rodRange = [-1,-1]
                conetimestamp1 = findDataTimestamp(coneListSorted[coneID])
                conetimestamp2 = findDataTimestamp(coneListSorted[coneID+1])
                coneInterval = conetimestamp2-conetimestamp1
                if _DEBUG:
                    if abs(coneInterval)> self.coneInterval:
                        print(key)
                        print('CONE INTERVL:',conetimestamp2-conetimestamp1)
                        print(coneListSorted[coneID-2:coneID+2])
                # (1)find the first rod sample and gurantee the  time(r)+n*dt > time(c)> time(r)
                for rodID in range(rodSearchStart,len(rodListSorted)):
                    rodtimestamp1 = findDataTimestamp(rodListSorted[rodID])
                    if rodtimestamp1 + MINTIMEINTERVAL*self.rodfilepersample <= conetimestamp1 :
                        continue
                    else:
                        deltaT = conetimestamp1 - rodtimestamp1
                        if deltaT < -MINTIMEINTERVAL*self.rodfilepersample:
                            rodRange[0] = -1
                            if _DEBUG and rodID>0:
                                print('ERROR:RODID[0] is too big')
                                print('cid:',coneID,'ct:',conetimestamp1)
                                print('rid:',rodID,'rt:',rodtimestamp1)
                        else:
                            bias = round(deltaT/MINTIMEINTERVAL)
                            rodRange[0] = rodID*self.rodfilepersample + bias
                        break
                # (2)find the last rod sample
                rodSearchStart = max(rodRange[0]//self.rodfilepersample-1,0)
                if rodRange[0] < 0 or coneInterval>self.coneInterval:
                    continue
                for rodID in range(rodSearchStart,len(rodListSorted)):
                    rodtimestamp2 = findDataTimestamp(rodListSorted[rodID])
                    if rodtimestamp2+ MINTIMEINTERVAL*self.rodfilepersample<conetimestamp2:
                        continue
                    else:
                        deltaT = conetimestamp2 - rodtimestamp2
                        if deltaT < -MINTIMEINTERVAL*self.rodfilepersample:
                            rodRange[1] = -1
                            if _DEBUG and rodID>0:
                                print('step2,ERROR:RODID[1] is too big')
                                print('step2,cid:',coneID+1,'ct:',conetimestamp2)
                                print('step2,rid:',rodID,'rt:',rodtimestamp2)
                                print('step2,deltaT:',deltaT,'deltaT/MINTIMEINTERVAL:',deltaT/MINTIMEINTERVAL)
                        else:
                            #check which file is it in
                            bias = round(deltaT/MINTIMEINTERVAL)
                            rodRange[1] = rodID*self.rodfilepersample + bias
                        break
                rodRange[1] += 1
                rodSearchStart = max(rodRange[1]//self.rodfilepersample-1,0)
                #(3)final check and add them to the training list
                if rodRange[1] < rodRange[0]+self.aopcoprate:
                    if self.print_info:
                        print('TOO FEW FILE')
                        print('cid:',coneID+1,'ct:',conetimestamp2)
                        print('rid:',rodID,'rt:',rodtimestamp2)
                        print('rodRange[1] - rodRange[0]:',rodRange[1] - rodRange[0])
                        print('deltaT:',deltaT,'deltaT/MINTIMEINTERVAL:',deltaT/MINTIMEINTERVAL)
                    continue
                legalSample['cone'] = [coneID,coneID+1]
                legalSample['rod']  = rodRange

                if len(labels) == 4:
                    legalSample['labels'] = labels
                else:
                    legalSample['labels'] = [['HDR', '0'], ['HS', '0'], 
                                             ['Blur', '0'], ['Noisy', '0']]
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
                legalFileList.append(legalSample)
                #print(legalSample)
            fileDict[key]['legalData'] = legalFileList
            fileDict[key]['RpF'] = self.rodfilepersample

        return fileDict  
              
    def extraction(self,rate,MAXLEN,ifUniformSampling):
        for key in self.fileDict:
            newLegalFileList = []
            sample_i = self.fileDict[key]
            rdpersample = sample_i['RpF']
            legalFileList = sample_i['legalData']
            #叠加放缩
            scaledLength = len(legalFileList)//rate
            #分散取样
            gap = 1
            if MAXLEN > 0 and scaledLength > MAXLEN and ifUniformSampling:
                gap = scaledLength//MAXLEN
            for i in range(0,scaledLength-1,gap):
                legalSample = dict([])
                coneidrange1 = legalFileList[i*rate]['cone'] 
                coneidrange2 = legalFileList[(i+1)*rate-1]['cone'] 
                rodidrange1 = legalFileList[i*rate]['rod']
                rodidrange2 = legalFileList[(i+1)*rate-1]['rod']
                c0 = coneidrange1[0]
                c1 = coneidrange2[1]
                r0 = rodidrange1[0]
                r1 = rodidrange2[1]
                if c1-c0 == rate:
                    legalSample['cone'] = (c0,c1)
                    legalSample['rod'] = (r0,r1)
                    legalSample['labels'] = legalFileList[i*rate]['labels']
                    newLegalFileList.append(legalSample)
                    if MAXLEN > 0 and len(newLegalFileList) > MAXLEN:
                        break
            sample_i['legalData'] = newLegalFileList
            self.fileDict[key] = sample_i
            if self.print_info:
                print(key,'origin length:',len(legalFileList),
                      ' legal length:',len(newLegalFileList))

        
    def dataNum(self,key):
        return len(self.fileDict[key]['legalData'])
        
    def __len__(self):
        return self.sampleNum 
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        for key in self.sampleNumDict:
            fmt_str = key + ' sample cell num:' + self.sampleNumDict[key] + '\n'
        return fmt_str

    
    # read a rod file directly
    def readRodFast(self,key,rod_id):
        if rod_id >= len(self.fileDict[key]['rod'])*self.rodfilepersample:
            print('invalid coneid for ',len(self.fileDict[key]['rod']),' rod data')
            return None,None,-1
        
        rodFileID = rod_id // self.rodfilepersample
        filename = self.fileDict[key]['rod'][rodFileID]
        rodtimeStamp = findDataTimestamp(filename)
        bytesize = self.rodFileSize
        size = bytesize // 4
        temp_diff_np = np.zeros((self.rodfilepersample, self.rod_width * self.rod_height), dtype=np.int8)
        spat_diff_left_np = np.zeros((self.rodfilepersample, self.rod_width * self.rod_height), dtype=np.int8)
        spat_diff_right_np = np.zeros((self.rodfilepersample, self.rod_width * self.rod_height), dtype=np.int8)
        pkt_size_np = np.zeros([self.rodfilepersample], dtype=np.int32)
        pkt_size_td = np.zeros([self.rodfilepersample], dtype=np.int32)
        pkt_size_sd = np.zeros([self.rodfilepersample], dtype=np.int32)
        one_frm_size = size // self.rodfilepersample
        ret_code = rdc.rod_decoder_py_byfile_td_sd_bw(filename, self.rodfilepersample, size, one_frm_size,
                                                temp_diff_np, spat_diff_left_np, spat_diff_right_np,
                                                pkt_size_np,pkt_size_td,pkt_size_sd,
                                                self.rod_height , self.rod_width)
        sd_list = []
        td_list = []
        for b in range(self.rodfilepersample):
            width = self.rod_width
            height = self.rod_height
            temp_diff_np1 = np.reshape(temp_diff_np[b, ...], (width, height))
            spat_diff_left_np1 = np.reshape(spat_diff_left_np[b, ...], (width, height))
            spat_diff_right_np1 = np.reshape(spat_diff_right_np[b,...], (width, height))
            sdl = spat_diff_left_np1
            sdr = spat_diff_right_np1
            td = temp_diff_np1
            sd_list.append((sdl, sdr))
            td_list.append(td)
            
        rodtimeStamp += (rod_id % self.rodfilepersample)* self.rodInterval
        return sd_list[rod_id % self.rodfilepersample],td_list[rod_id % self.rodfilepersample],rodtimeStamp,filename
          
    # read a cone file directly
    def readConeFast(self,key,cone_id,viz=True,useISP=False,ifSync =True):
        if cone_id >= len(self.fileDict[key]['cone']):
            print('invalid coneid for ',len(self.fileDict[key]['cone']),' cone data')
            return None,-1
        conefilename = self.fileDict[key]['cone'][cone_id]
        conetimeStamp = findDataTimestamp(conefilename)
        size = os.path.getsize(conefilename)
        #wihei = int(np.sqrt((size - 64) // 4))
        raw_vec = np.zeros(self.cone_height * self.cone_width, dtype=np.int16)
        rdc.cone_reader_py_byfile(conefilename, size // 4, raw_vec, self.cone_height, self.cone_width)
        cone_raw = np.reshape(raw_vec, (self.cone_height, self.cone_width))
        start_frame = cone_raw
        #rgb_processed = start_frame
        return start_frame, conetimeStamp, conefilename

    def packRead(self,idx,key,ifSync =True, needPreProcess = True):
        sample = dict([])
        legalSample = self.fileDict[key]['legalData'][idx]
        coneRange = legalSample['cone']
        rodRange  = legalSample['rod']
        metaInfo = dict([])
        
        start_frame,coneTimeStamp1,conefilename1 = self.readConeFast(key,coneRange[0])
        end_frame,coneTimeStamp2,conefilename2 = self.readConeFast(key,coneRange[1])
        
        metaInfo['C_name'] = (conefilename1,conefilename2)
        metaInfo['C_timestamp'] = (coneTimeStamp1,coneTimeStamp2)
        metaInfo['R_name'] = []
        metaInfo['R_timestamp'] = []
        metaInfo['R_bias'] = []
        metaInfo['key'] = key
        metaInfo['sample_length'] = len(self.fileDict[key]['legalData'])

        itter = rodRange[1] - rodRange[0]
        if itter<0:
            print(key,coneStartId, cone_id, coneRange)
            print(itter , rodRange[1] , rodRange[0])
            return None
        tsd = torch.zeros([3,itter,160,160])
        SD_0 = 0
        SD_1 = 0
        
        for i in range(itter):
            sd,td,rodTimeStamp,filename = self.readRodFast(key,rodRange[0] + i)
            metaInfo['R_name'].append(filename)
            metaInfo['R_timestamp'].append(rodTimeStamp)
            metaInfo['R_bias'].append((rodRange[0] + i) % self.rodfilepersample)
            sdl,sdr = sd
            tsd[0,i,:,:] = torch.Tensor(td)
            tsd[1,i,:,:] = torch.Tensor(sdl)
            tsd[2,i,:,:] = torch.Tensor(sdr)
            SD = tsd[1:,i,...]
            if i == 0:
                SD_0 = tsd[1:,i,...]
            if i == itter - 1:
                SD_1 = tsd[1:,i,...]
            
        if needPreProcess:
            start_frame,end_frame,F0_without_isp,F1_without_isp,tsdiff_inter  = self.preprocess(start_frame,end_frame,tsd)
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
        
    def locateSample(self,index):
        relativeIndex = index
        for key in self.sampleNumDict:
            numKey = self.sampleNumDict[key]
            if relativeIndex >= numKey:
                relativeIndex -= numKey
            else:
                return key,relativeIndex
        return key,relativeIndex

    
    def preprocess(self,F0_raw,F1_raw,tsdiff):
        
        F0,F0_without_isp = default_rgb_isp(F0_raw)
        F1,F1_without_isp = default_rgb_isp(F1_raw)
        tsdiff_inter = self.upsampleTSD_conv(tsdiff)/128.0
        
        return F0,F1,F0_without_isp,F1_without_isp,tsdiff_inter
    
    def __getitem__(self, index):
        #定位在哪个sample里
        key,relativeIndex = self.locateSample(index)
        sample = self.packRead(relativeIndex, key)
        return sample
    
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