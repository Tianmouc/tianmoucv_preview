import numpy as np
import cv2,sys
import torch
import math,time,subprocess
import torch.nn.functional as F
import torch.nn as nn

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
from .denoise_utils import custom_round,conv_and_threshold,conv_and_threshold_1, denoise_defualt_args


class TianmoucDataReader(TianmoucDataReader_basic):
    '''
    - TianmoucDataReader(version 0.3.7.4 in dev)
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
                 aop_denoise = False,
                 aop_denoise_args = denoise_defualt_args(),
                 showList=True,
                 MAXLEN=-1,
                 uniformSampler=False,
                 matchkey=None,
                 cachePath=None,
                 ifcache = False, 
                 print_info=True,
                 training=True,
                 strict = True,
                 use_data_parser = False,
                 dark_level = 0):
        
        self.N = N
        self.print_info = print_info
        self.use_data_parser = use_data_parser
        super().__init__(path,
                 showList=showList,
                 MAXLEN=MAXLEN,
                 uniformSampler = uniformSampler,
                 camera_idx= camera_idx,
                 matchkey=matchkey,
                 cachePath=cachePath,
                 ifcache = ifcache, 
                 print_info=print_info,
                 training=training,
                 strict = strict,
                 dark_level = dark_level) # 调用父类的属性赋值方法

        self.aop_denoise = aop_denoise
        self.aop_denoise_args = aop_denoise_args
        if self.aop_denoise:
            print('[tianmoucv Datareader]Customising your aop_denoise_args using tianmoucv.data.denoise_utils.denoise_defualt_args()')
            print('[tianmoucv Datareader]Better to provide:{\'TD\':[np.array]*2,\'SDL\':[np.array]*2*aop_cop_rate,\'SDR\':[np.array]*2*aop_cop_rate}')
            if self.aop_denoise_args.aop_dark_dict['TD'] is None:
                if self.aop_denoise_args.self_calibration:
                    print('[tianmoucv Datareader] cannot find TD dark, use self calibration')
                    self.aop_denoise_args.aop_dark_dict['TD'] = self.td_fpn_calibration_(Num=min(500,self.__len__()-1))
                else:
                    print('[tianmoucv Datareader] cannot find SDL dark, use ZERO')
                    self.aop_denoise_args.aop_dark_dict['TD'] = [torch.zeros(160, 160) for _ in range(2)]
                    
            if self.aop_denoise_args.aop_dark_dict['SDL'] is None:
                
                if self.aop_denoise_args.self_calibration:
                    print('[tianmoucv Datareader] cannot find SDL dark, use self calibration')
                    SDL_dark,SDR_dark = self.sd_fpn_calibration_(Num=min(500,self.__len__()-1))
                    self.aop_denoise_args.aop_dark_dict['SDL'] = SDL_dark
                    self.aop_denoise_args.aop_dark_dict['SDR'] = SDR_dark 
                else:
                    print('[tianmoucv Datareader] cannot find SDL dark, use ZERO')
                    self.aop_denoise_args.aop_dark_dict['SDL'] = [torch.zeros(160, 160) for _ in range(2)]
                    self.aop_denoise_args.aop_dark_dict['SDR'] = [torch.zeros(160, 160) for _ in range(2)] 
                
            self.choose_correct_fpn(thr_1=self.aop_denoise_args.gain) #判断奇偶帧
            self.aop_denoise_args.print_info()
            print('[tianmoucv Datareader Warning] Doesn\'t support multiple keys for Denoise in this version')
            

    #你可以重写这个extration逻辑以获得更复杂的数据读取方法，例如抽帧等        
    def extraction(self,MAXLEN,uniformSampler):
        for key in self.fileDict:
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
                
            if MAXLEN>0:
                if len(new_legalFileList)>MAXLEN:
                    if uniformSampler:
                        adptiverate = len(new_legalFileList)//MAXLEN
                        new_legalFileList  = [new_legalFileList[i] for i in range(0, adptiverate*MAXLEN, adptiverate)]
                    else:
                        new_legalFileList  = new_legalFileList[:MAXLEN]

            self.fileDict[key]['legalData']  = new_legalFileList
            if self.print_info:
                print('[tianmoucv Datareader]',key,'extracted length:',len(new_legalFileList))

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

        #read all rgb frames
        for i in range(self.N+1):
            #print('caddr:',coneAddrs[i])
            frame,timestamp = self.readConeFast(conefilename,coneAddrs[i])
            frame_raw = np.reshape(frame.copy(), (self.cone_height,self.cone_width))
            frame = np.reshape(frame.astype(np.float32),(self.cone_height,self.cone_width))
            raw_list.append(frame_raw)
            rgb_list.append(frame)
            coneTimeStamp_list.append(timestamp.astype(np.int64))

        cone_id = legalSample['coneid']
        rod_id = legalSample['rodid']

        #prepare all meta infomation
        metaInfo['C_name'] = conefilename
        metaInfo['C_timestamp'] = coneTimeStamp_list
        metaInfo['C_idx'] = cone_id
        metaInfo['R_name'] = rodfilename
        metaInfo['R_timestamp'] = []
        metaInfo['R_idx'] = rod_id
        metaInfo['key'] = key
        metaInfo['sample_length'] = len(self.fileDict[key]['legalData'])
        itter = len(rodAddrs)

        #read all aop frames
        tsd = torch.zeros([3,itter,self.rod_height,self.rod_width])
        for i in range(itter):
            startAddr = rodAddrs[i]
            sdl,sdr,td,rodTimeStamp = self.readRodFast(rodfilename,startAddr)
            tsd[0,i,:,:] = torch.Tensor(td.astype(np.float32)).view(self.rod_height,self.rod_width)
            tsd[1,i,:,:] = torch.Tensor(sdl.astype(np.float32)).view(self.rod_height,self.rod_width)
            tsd[2,i,:,:] = torch.Tensor(sdr.astype(np.float32)).view(self.rod_height,self.rod_width)
            metaInfo['R_timestamp'].append(rodTimeStamp.astype(np.int64))

        #denoising
        if self.aop_denoise:
            tsd = self.denoise_based_on_SD_(tsd, rod_id, 
                                            TD_dark=self.aop_denoise_args.aop_dark_dict['TD'],
                                            SD_dark_left=self.aop_denoise_args.aop_dark_dict['SDL'], 
                                            SD_dark_right=self.aop_denoise_args.aop_dark_dict['SDR'], 
                                            thr_1=self.aop_denoise_args.thr_1,
                                            thr_2=self.aop_denoise_args.thr_2,
                                            thr_3=self.aop_denoise_args.thr_3)

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
        '''
        定位在哪个sample里
        '''
        if index >= self.__len__():
            print('[tianmoucv Datareader ERROR] INDEX OUT OF RANGE!')
            return None     
        key,relativeIndex = self.locateSample(index)
        if key is None:
            print('[tianmoucv Datareader ERROR] No data Found for this key!')
            return None   
        sample = self.packRead(relativeIndex, key)
        if self.use_data_parser:
            sample = TianmoucDataSampleParser(sample)
        return sample

    ##################################################################################################################################
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>[去噪功能实验区]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ###################################################################################################################################

    def get_dark_noise(self):
        '''
        inspect dark noise or calibrate dark noise data
        '''
        return self.aop_denoise_args.aop_dark_dict
        
    def choose_correct_fpn(self,thr_1=1,idx=0):
        '''
        空间噪声
        '''
        TD_dark = self.aop_denoise_args.aop_dark_dict['TD']
        SDL_dark= self.aop_denoise_args.aop_dark_dict['SDL']
        SDR_dark= self.aop_denoise_args.aop_dark_dict['SDR']
        tsdiff,rodid = self.get_raw_tsdiff_(idx)

        timelen = tsdiff.shape[1]-1
        
        #判断奇偶帧匹配
        cal = tsdiff[0, 0, ...]
        cal_0 = cal - TD_dark[0]
        cal_1 = cal - TD_dark[1]
        var_org = torch.var(cal)
        var_cal0 = torch.var(cal_0)
        var_cal1 = torch.var(cal_1)
    
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 9.0
    
        match_tensor_0 = cal_0.unsqueeze(0).unsqueeze(0)
        smoothed_matrix = F.conv2d(match_tensor_0, kernel, padding=1)
        smoothed_matrix = smoothed_matrix.squeeze(0).squeeze(0)
        mask_0 = torch.abs(smoothed_matrix) >= thr_1
    
        match_tensor_1 = cal_1.unsqueeze(0).unsqueeze(0)
        smoothed_matrix = F.conv2d(match_tensor_1, kernel, padding=1)
        smoothed_matrix = smoothed_matrix.squeeze(0).squeeze(0)
        mask_1 = torch.abs(smoothed_matrix) >= thr_1
        mask_0 = mask_0.float()
        mask_1 = mask_1.float()

        var_0 = torch.var(mask_0)  #dark0和tsdiff0的匹配
        var_1 = torch.var(mask_1)  #dark1和tsdiff1的匹配
  
        TD_corrected_dark =  [torch.zeros(160, 160) for _ in range(2)]
        SDL_corrected_dark =  [torch.zeros(160, 160) for _ in range(2*timelen)]
        SDR_corrected_dark =  [torch.zeros(160, 160) for _ in range(2*timelen)]

        if len(SDL_dark) <  2*timelen:
            print('[tianmoucv Datareader Warning] Dark Noise mode does not match the data,use average-copy version(无法去横条纹)')
            SDL_dark = [torch.mean(torch.stack(SDL_dark[0:len(SDL_dark)//2],dim=0),dim=0),torch.mean(torch.stack(SDL_dark[len(SDL_dark)//2:],dim=0),dim=0)]
            SDL_dark = [SDL_dark[0]]*timelen + [SDL_dark[1]]*timelen
            SDR_dark = [torch.mean(torch.stack(SDR_dark[0:len(SDR_dark)//2],dim=0),dim=0),torch.mean(torch.stack(SDR_dark[len(SDR_dark)//2:],dim=0),dim=0)]
            SDR_dark = [SDR_dark[0]]*timelen + [SDR_dark[1]]*timelen
            
        #如果第一帧是偶数帧，并且第一帧更适配[0]
        #或者第一帧是偶数帧，并且第一帧更适配[1]
        if (var_cal0 < var_cal1 and rodid[0] % 2 ==0) or (var_cal0 > var_cal1 and rodid[0] % 2 ==1):
            TD_corrected_dark[1] = TD_dark[1]
            TD_corrected_dark[0] = TD_dark[0]
            for j in range(timelen):
                SDL_corrected_dark[j] = SDL_dark[j] 
                SDL_corrected_dark[j+timelen] = SDL_dark[j+timelen] 
                SDR_corrected_dark[j] = SDR_dark[j] 
                SDR_corrected_dark[j+timelen] = SDR_dark[j+timelen] 
        else:
            TD_corrected_dark[1] = TD_dark[0]
            TD_corrected_dark[0] = TD_dark[1]
            for j in range(timelen):
                SDL_corrected_dark[j] = SDL_dark[j+timelen] 
                SDL_corrected_dark[j+timelen] = SDL_dark[j] 
                SDR_corrected_dark[j] = SDR_dark[j+timelen] 
                SDR_corrected_dark[j+timelen] = SDR_dark[j] 

        #已经换好顺序的fpn-dark
        self.aop_denoise_args.aop_dark_dict['TD'] = TD_corrected_dark
        self.aop_denoise_args.aop_dark_dict['SDL'] = SDL_corrected_dark
        self.aop_denoise_args.aop_dark_dict['SDR'] = SDR_corrected_dark
        return TD_corrected_dark, SDL_corrected_dark, SDR_corrected_dark
        

    def get_raw_tsdiff_(self, index):
        '''
        只拿原始rod数据和其绝对id，提升效率
        '''
        if index >= self.__len__():
            print('[tianmoucv Datareader ERROR] INDEX OUT OF RANGE!')
            return None     
        key,relativeIndex = self.locateSample(index)
        if key is None:
            print('[tianmoucv Datareader ERROR] No data Found for this key!')
            return None   
        legalSample = self.fileDict[key]['legalData'][index]
        rodfilename  = self.fileDict[key][self.pathways[0]]  # only read first one, if you want to use dual camera you can read it again
        rodAddrs = legalSample[self.pathways[0]]
        rod_id = legalSample['rodid']
        itter = len(rodAddrs)
        assert itter>=0
        tsd = torch.zeros([3,itter,self.rod_height,self.rod_width])
        for i in range(itter):
            startAddr = rodAddrs[i]
            sdl,sdr,td,rodTimeStamp = self.readRodFast(rodfilename,startAddr)
            tsd[0,i,:,:] = torch.Tensor(td.astype(np.float32)).view(self.rod_height,self.rod_width)
            tsd[1,i,:,:] = torch.Tensor(sdl.astype(np.float32)).view(self.rod_height,self.rod_width)
            tsd[2,i,:,:] = torch.Tensor(sdr.astype(np.float32)).view(self.rod_height,self.rod_width)
        return tsd,rod_id
        
    def td_fpn_calibration_(self,Num=20):
        '''
        计算奇数帧和偶数帧TD空间噪声，Num为标定所用的index数量
        '''
        tsdiff_,_ = self.get_raw_tsdiff_(0)
        tsdiff = tsdiff_.clone()
        timelen = tsdiff.shape[1]-1
        TD_mean = [torch.zeros(160, 160) for _ in range(2)]
        TD_mean_odd = torch.zeros(160, 160)
        TD_mean_even = torch.zeros(160, 160)

        odd_count = 0
        even_count = 0
        for i in range(Num):
            tsdiff_,rod_id = self.get_raw_tsdiff_(i)
            #print('[>>>>>>>>>>>debug>>>>>>>>>>>>>>>]',rod_id)
            if tsdiff_ is None or tsdiff.shape[1] < timelen:
                print('[tianmoucv Datareader warninig] td_fpn_calibration_ lost data')
                continue
            tsdiff = tsdiff_.clone()
            
            for j in range(timelen):
                TD_0 = tsdiff[0, j, ...]
                if rod_id[j]%2==0:#偶数rod帧
                    TD_mean_even += TD_0
                    even_count += 1
                if rod_id[j]%2==1:#偶数rod帧
                    TD_mean_odd += TD_0
                    odd_count += 1
                
        TD_mean_even/= even_count
        TD_mean_odd/= odd_count
        TD_mean_odd = custom_round(TD_mean_odd)
        TD_mean_even = custom_round(TD_mean_even)
        
        TD_mean[0]=TD_mean_even
        TD_mean[1]=TD_mean_odd

        print('[tianmoucv Datareader] TD CALIB NUM:',odd_count,even_count)
        
        return TD_mean

    def sd_fpn_calibration_(self,Num=20):
        '''
        计算奇数index帧和偶数index帧TD空间噪声，Num为标定所用的index数量
        返回标定的SD_mean_left, SD_mean_right
        '''
        tsdiff_,_ = self.get_raw_tsdiff_(0)
        tsdiff = tsdiff_.clone()
        timelen = tsdiff.shape[1]-1
        odd_count = 0
        even_count = 0
        SD_mean_left = [torch.zeros(160, 160) for _ in range(2*timelen)]
        SD_mean_right = [torch.zeros(160, 160) for _ in range(2*timelen)]

        for i in range(Num):
            tsdiff_,rod_id = self.get_raw_tsdiff_(i)
            if tsdiff_ is None or tsdiff.shape[1] < timelen:
                print('[tianmoucv Datareader warninig] sd_fpn_calibration_ lost data')
                continue
            tsdiff = tsdiff_.clone()

            if rod_id[0]%2==0:#偶数rod帧,考虑条纹噪声
                even_count += 1
                for j in range(timelen):
                    SDL = tsdiff[1, j, ...]
                    SDR = tsdiff[2, j, ...]
                    SD_mean_left[j] += SDL
                    SD_mean_right[j] += SDR
            else:#偶数rod帧,考虑条纹噪声
                odd_count += 1
                for j in range(timelen):
                    SDL = tsdiff[1, j, ...]
                    SDR = tsdiff[2, j, ...]
                    SD_mean_left[j+timelen] += SDL
                    SD_mean_right[j+timelen] += SDR

        for j in range(timelen):
            SD_mean_left[j] /= even_count
            SD_mean_left[j] = custom_round(SD_mean_left[j])
            SD_mean_right[j] /= even_count
            SD_mean_right[j] = custom_round(SD_mean_right[j])

            SD_mean_left[j+timelen] /= odd_count
            SD_mean_left[j+timelen] = custom_round(SD_mean_left[j+timelen])
            SD_mean_right[j+timelen] /= odd_count
            SD_mean_right[j+timelen] = custom_round(SD_mean_right[j+timelen])   

        print('[tianmoucv Datareader] SD CALIB NUM:',odd_count,even_count)

        return SD_mean_left, SD_mean_right


    def denoise_based_on_SD_(self, raw_tsd, rod_id, TD_dark=None, SD_dark_left=None, SD_dark_right=None, thr_1=1, thr_2=3, thr_3=3):
        '''
        TD_dark为TD空间噪声，SD_dark_left与SD_dark_right为SD空间噪声，
        thr_1为对角线值平均以后的滤波阈值，thr_2和thr_3为
        时空相关性滤波中组滤波和细滤波的阈值。返回SD_denoise shape为[3,itter，160，160]
        '''
        denoise_raw_tsd=torch.zeros(3, raw_tsd.shape[1], 160, 160)

        timelen = raw_tsd.shape[1] - 1

        for j in range(raw_tsd.shape[1]):
            if rod_id[j]%2 == 0:
                raw_tsd[0, j, ...] -= TD_dark[0]
            else:
                raw_tsd[0, j, ...] -= TD_dark[1]

            if rod_id[0]%2 == 0:
                raw_tsd[1, j, ...] -= SD_dark_left[j%timelen]
                raw_tsd[2, j, ...] -= SD_dark_right[j%timelen]    
            else:
                raw_tsd[1, j, ...] -= SD_dark_left[j%timelen+timelen]
                raw_tsd[2, j, ...] -= SD_dark_right[j%timelen+timelen]
                    
            #TD只去除空间噪声
            denoise_raw_tsd[0, j, ...]=raw_tsd[0,j,...]
            #取出SDleft和SDright
            LSD = (raw_tsd[1, j, ...]).clone()
            RSD = (raw_tsd[2, j, ...]).clone()
            # 取出SDul...
            sdul = LSD[0::2, ...]
            sdll = LSD[1::2, ...]
            sdur = RSD[0::2, ...]
            sdlr = RSD[1::2, ...]
            # sdul与sdlr作差之后平均，使用一个阈值进行过滤后进行空间相关性滤波
            TD_1 = conv_and_threshold((sdul - sdlr) / 2, 3, thr_2)
            TD_1 = conv_and_threshold_1(TD_1, 3, thr_3)
            TD_10 = torch.abs(TD_1) >=thr_1
            TD_1 = (TD_1) * TD_10.float()
            # sdll与sdur作差之后平均，使用一个阈值进行过滤后进行空间相关性滤波
            TD_2 = conv_and_threshold((sdll - sdur) / 2, thr_3, thr_2)
            TD_2 = conv_and_threshold_1(TD_2, 3, 5)
            TD_20 = torch.abs(TD_2) >= thr_1
            TD_2 = (TD_2) * TD_20.float()
            #重建SDleft与SDright对应的模板
            LSD_reconstructed = torch.stack([TD_10.float(), TD_20.float()], dim=1).view(
                    LSD.shape)
            RSD_reconstructed = torch.stack([TD_20.float(), TD_10.float()], dim=1).view(
                    LSD.shape)

            #在原SDleft与SDright帧中根据模板提取信号
            denoise_raw_tsd[1, j, ...] = raw_tsd[1, j, ...]*LSD_reconstructed
            denoise_raw_tsd[2, j, ...] = raw_tsd[2, j, ...]*RSD_reconstructed
                
            '''
            old td
            '''
            td = raw_tsd[0:1,j:j+1,...]
            # Apply smoothing
            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 9.0  # Normalize
            smoothed_matrix = F.conv2d(td, kernel, padding=1)
            smoothed_matrix = smoothed_matrix.squeeze(0).squeeze(0)
            # Create a mask based on the smoothed output
            mask_1 = torch.abs(smoothed_matrix) >= thr_1
            td = td.squeeze(0).squeeze(0) * mask_1
            # Further processing using conv_and_threshold functions
            td = conv_and_threshold(td, 3, thr_2)
            denoise_raw_tsd[0, j, ...]  = conv_and_threshold_1(td, 3, thr_3)

        return denoise_raw_tsd


    ##################################################################################################################################
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>[去噪功能实验区]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ###################################################################################################################################

