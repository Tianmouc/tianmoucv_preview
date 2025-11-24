#coding=utf-8
print('>>this data reader is a dev version in 2024-01-06')
from ctypes import *
import ctypes

from threading import local

import numpy as np
import time
import os
from queue import Queue

global_aop_queue = Queue()
global_cop_queue = Queue()
result_queue = Queue()
inference_queue = Queue()

ROD_W =160
ROD_H =160
CONE_W =320
CONE_H =320

############################################################
#
# Basic data type
#
############################################################
class tianmoucData(Structure):
    _fields_ = [("isRod", c_int),#1:rod,0:cone
                ("timestamp", c_ulonglong),
                ("sdl_p", POINTER(c_ubyte)),
                ("sdr_p", POINTER(c_ubyte)),
                ("td_p",  POINTER(c_ubyte)),
                ("rgb_p", POINTER(c_float))]
                
############################################################
#
# A simple streaming reader
# Author: Y. Lin
# 2024 01 06
#
############################################################
class TMstreamingDataReader():
    
    '''
        这段代码实现了一个名为TMstreamingDataReader的类，作为数据读取器。
        类的初始化方法（init）中，加载了SDK库文件，并初始化了相机。

        @parameter cfg_path: csv格式的配置文件
        @parameter lib_path: 预编译的sdk动态链接库，ubuntu为.so，windows为.dll，macOS为,dylib
        @parameter device_id: 相机的id，支持多目相机同时使用

        主要函数说明：
        
        - __init__方法：加载SDK库文件，初始化相机，准备数据包。 
        - setExposure方法：设置相机的曝光时间。 
        - __enter__方法：开始读取数据。 
        - __exit__方法：处理异常并停止相机。 
        - __call__方法：读取数据帧并返回。
        
        这段代码的功能是通过SDK库读取相机的数据帧并进行处理，提供了读取数据和设置曝光时间的接口。
        在使用该类时，可以通过with语句自动调用__enter__和__exit__方法，确保资源的正确释放。
    '''

    def __init__(self,cfg_path = None,lib_path = None,device_id=0):
        # load sdk lib
        file_path = os.path.realpath(__file__)
        file_path = os.path.dirname(file_path)

        if cfg_path is None:
            cfg_path = os.path.join(file_path,"lib","Golden_HCG_seq.csv")
        if lib_path is None:
            lib_path = os.path.join(file_path,"lib","libtianmouc.so")

        self.cfg_path  = str(cfg_path).encode("utf-8")
        tm_sdk = cdll.LoadLibrary(lib_path)
        tm_sdk.tmOpenCamera.restype = ctypes.c_uint64

        # init camera
        cameraHandle = tm_sdk.tmOpenCamera(device_id)
        print('[TMstreamingDataReader]type(cameraHandle)',type(cameraHandle),tm_sdk.tmOpenCamera.restype)

        # prepare data pack
        coneData = pointer(c_float(CONE_W*CONE_H*2*3))
        tdData1 = pointer(c_ubyte(ROD_W*ROD_H))
        sdData1 = pointer(c_ubyte(ROD_W*ROD_H))
        sdData2 = pointer(c_ubyte(ROD_W*ROD_H))
        idRod = c_int(1)
        timestamp = c_ulonglong(1)
        tianmoucDataValue = [idRod,timestamp,tdData1,sdData1,sdData2,coneData]
        self.dataPack = tianmoucData(*tianmoucDataValue)
        self.dataPack_p	 = pointer(self.dataPack)

        #store data buffer and ssdk
        self.tm_sdk = tm_sdk
        self.cameraHandle = c_uint64(cameraHandle)
        if cameraHandle > 0:
            print('[TMstreamingDataReader] Tianmouc camera:',hex(cameraHandle),'is raedy')
            self.tm_sdk.IICconfig_download(self.cameraHandle,self.cfg_path)
            print('[TMstreamingDataReader] Tianmouc camera:',hex(cameraHandle),' configure downloaded')
        else:
            print('[TMstreamingDataReader] fail to get correct camera handle:',hex(cameraHandle))
            

    def setExposure(self,aoptime,coptime,aopG=1,copG=1):
        aoptime = min(max(aoptime,4),1240)
        coptime = min(max(coptime,4),15000)
        print(aoptime,coptime,aopG,copG)
        self.tm_sdk.tmExposureSet(self.cameraHandle,aoptime,coptime,aopG,copG,8,1,1)
        #1:par,8:bit

    #use python-with to start reader
    def __enter__(self):
        # start camera
        print('[TMstreamingDataReader] download',self.cfg_path)
        self.tm_sdk.tmStartTransfer(self.cameraHandle)
        print('[TMstreamingDataReader] Tianmouc data transfer start:',self.cameraHandle)
        return self
    

    # deal with exception
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[TMstreamingDataReader] catch exception:',exc_type, exc_val, exc_tb)
        #stop camera
        self.tm_sdk.tmCameraUninit(self.cameraHandle)
        return True


    # functional object
    def __call__(self):
        #ts = time.time()
        ret = self.tm_sdk.tmGetFrame(self.cameraHandle,self.dataPack_p)
        
        if not ret:
            print('[TMstreamingDataReader] loss data or loss connection, this frame is not correct')
        
        if self.dataPack.isRod == 1:
            td_buffer = cast(self.dataPack.td_p, POINTER(c_ubyte*ROD_W*ROD_H))[0]
            sdl_buffer = cast(self.dataPack.sdl_p, POINTER(c_ubyte*ROD_W*ROD_H))[0]
            sdr_buffer = cast(self.dataPack.sdr_p, POINTER(c_ubyte*ROD_W*ROD_H))[0]
            TD = np.frombuffer(td_buffer,dtype=np.byte, count=ROD_W*ROD_H).reshape(ROD_H,ROD_W)
            SDL = np.frombuffer(sdl_buffer,dtype=np.byte, count=ROD_W*ROD_H).reshape(ROD_H,ROD_W)
            SDR = np.frombuffer(sdr_buffer,dtype=np.byte, count=ROD_W*ROD_H).reshape(ROD_H,ROD_W)
            isRod = self.dataPack.isRod
            timestamp = self.dataPack.timestamp
            return isRod, timestamp,(TD,SDL,SDR)
        else:
            RGB_buffer = cast(self.dataPack.rgb_p, POINTER(c_float*CONE_H*CONE_W*2*3))[0]
            RGB = np.frombuffer(RGB_buffer,dtype=np.float32).reshape(CONE_H,CONE_W*2,3)
            isRod = self.dataPack.isRod
            timestamp = self.dataPack.timestamp
            return isRod, timestamp, RGB

