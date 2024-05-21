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
    print("WARNING: no decoder found, try to compile under ./rod_decoder_py")
    current_file_path = os.path.abspath(__file__)
    parent_folder_path = os.path.dirname(os.path.dirname(current_file_path))
    aim_path = os.path.join(parent_folder_path,'rdp_usb')
    os.chdir(aim_path)
    current_path = os.getcwd()
    print("Current Path:", current_path)
    subprocess.run(['sh', './compile_pybind.sh'])
    from tianmoucv.rdp_usb import rod_decoder_py as rdc
    print('compile decoder successfully')
    
#用于重建
from tianmoucv.proc.reconstruct import laplacian_blending
from tianmoucv.isp import default_rgb_isp
from .tianmoucData_basic import TianmoucDataReader_basic

class TianmoucDataReader(TianmoucDataReader_basic):
    '''
    - TianmoucDataReader注释(0.3.5版本)
        
        **输入**
        
        - 输入dataPath：该路径下应当包含1个或多个子目录，每个子目录对应1段Tianmouc视频。
            - 支持string格式(仅输入1个地址)或list格式(输入1个或多个地址)。
               - 这个地址可以是一个tmdat sample的绝对路径，也可以是数据集的路径
               - 如果是数据集路径(path或者path的任意多层子文件夹内有多个tmdat),可以用matchkey读取特定sample，也可以合并读取
            - 对于单目数据，每个sample下应包含rod和cone两个目录，多目数据额外还有目录rod_N和cone_N，N为相机编号N>=1
            - 双目数据补充：20240201测试结果，在实验过程中重启GUI不会导致两个相机标签交换
                - 只要不插拔并交换接线，整个数据集中相机的idx将保持不变
        - 输入N：返回的dataset中包含多个sample，每个sample包含(N+1)帧COP，以及中间的所有AOP帧。
            - 默认N=1，在757fps模式下sample中有F0，F1两帧COP，以及中间的(25+1)帧AOP，最后一帧AOP与下一个sample第1帧AOP相同，可以跳过。
        - 输入matchkey：在dataPath所有路径下的子目录名称中匹配对应的，否则会输出所有数据。
            - 若输入超过1个路径，建议不同路径下不要出现同名子目录，否则可能出现bug。
        - 输入camera_idx：默认为0，表示识别单目输入，若为双目数据，则camera_idx=0,1分别录取双目数据。
        - 原先版本中的输入参数MAXLEN强制默认设为-1，即始终为读取全部数据。
        
        **输出**
        
        - 输出dataset调用方式类似于列表，通过sample = dataset[index]逐一获取数据。
        - sample为字典类型，包含如下key
            - COP帧依次记录为F0，F1，F2...F(N)
                - COP的精确帧率为30.3fps
                - 'F0'默认使用ISP算法调色, 可以关闭
                - 'F0_without_isp'不加额外处理，若加红外滤光片应使用这个数据
                - 'F0_HDR'为简易融合算法处理结果，由同步的SD和RGB合成高动态图
            - AOP帧
                - 'rawDiff'为AOP像素原始输出(160×160), 为tianmuocv非神经网络预处理接口的输入
                - 'tsdiff'为rawDiff直接插值得到的与COP同分辨率的图像(320×640), 用于神经网络的输入
                - 上述三个对应的key_value均为张量格式，torch.Size([3, X, height, width])
                    - 第0个维度为3，分别依次对应TD，SD1，SD2
                    - 第1个维度对应AOP帧数目，在757fps模式下X=N×25+1, 每25帧为一个单位
                    - 第2，3个维度对应AOP帧的分辨率
            - 'sysTimeStamp'为系统初始时间，用于在多目相机情况下进行时间对齐。
                - 两相机之间初始时间差为sysTimeStamp1-sysTimeStamp2，单位为秒
                - COP对齐时若Δt>33ms/2，建议让相机1的第K+Δt/33ms帧COP与相机2的第K帧COP对齐，这样时间差更小。
            - 'labels'用于标注HDR，HS，Blur，Noisy等4种极端情况分类，暂未实装。
            - 'meta'包含了该段目录的一些元数据，如文件存储目录，时间戳等等，需要详细数据分析时使用
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
                 dark_level = os.path.dirname(os.path.abspath(__file__))+'/blc/camera605.npz'):
        
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
                 dark_level = dark_level) # 调用父类的属性赋值方法
        
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
        '''
        for key in sample:
            print(key)
            if isinstance(sample[key],torch.Tensor) or isinstance(sample[key],np.ndarray):
                print(sample[key].shape)
            else:
                print(sample[key])        
        '''
        return sample