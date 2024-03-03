#coding=utf-8

import numpy as np
import threading
import time

import cv2
import torch
import torch.nn.functional as F

#用于重建
from tianmoucv.isp import fourdirection2xy,poisson_blend
#用于去马赛克
from tianmoucv.isp import lyncam_raw_comp,demosaicing_npy
#用于色彩调整
from tianmoucv.isp import default_rgb_adjust
#start camera and read
from tianmoucv.camera.sdk_utils import TMstreamingDataReader
from tianmoucv.camera.sdk_utils import global_aop_queue,global_cop_queue,result_queue

# deal with aop d\ata and put the result back to result queue
def AOP_thread():
    while True:
        time.sleep(0.01)
        if global_aop_queue.qsize() > 0:
            rawdata = global_aop_queue.get()
            td,sdl,sdr = rawdata
            showimg = np.stack([td,sdl,sdr],axis=-1)
            showimg[abs(showimg)<3] = 0
            result_queue.put(('aop',showimg/32.0))

            SD = torch.Tensor(np.stack([sdl,sdr],axis=-1))
            Ix,Iy= fourdirection2xy(SD)
            gray = poisson_blend(-Ix,-Iy,iteration=20).numpy()
            gray = (gray-np.min(gray))/(np.max(gray)-np.min(gray))
            result_queue.put(('aop_gray',gray))
    
def COP_thread():
    while True:
        time.sleep(0.01)
        if global_cop_queue.qsize() > 0:
            rawdata = global_cop_queue.get()
            rgb = rawdata.astype(np.float32)
            rgb = default_rgb_adjust(rgb)
            result_queue.put(('cop',rgb[...,[2,1,0]]))
        
######################################################
# 
#   main loop
#     
######################################################
cfg_path = None #"/home/lyh/projects/lyncam_proj/lyncam_driver/cfg_csv/para_8bit/Golden_HCG_seq.csv"
lib_path = None #"/home/lyh/projects/tianmoucv/cpplib/libtianmouc.so"
tianmoucReader = TMstreamingDataReader(cfg_path=cfg_path,lib_path=lib_path)

with tianmoucReader as reader:
    count = 0
    threadaop = threading.Thread(target=AOP_thread)
    threadcop = threading.Thread(target=COP_thread)
    threadaop.daemon = True  # Set the thread as a daemon thread
    threadcop.daemon = True  # Set the thread as a daemon thread
    threadcop.start()
    threadaop.start()

    while True:
        isrod, rawdata = reader()
        if isrod:
            count += 1
            if count % 25 == 0:
                global_aop_queue.put(rawdata)
        else:
            global_cop_queue.put(rawdata)
            
        if result_queue.qsize() > 0:
            result_name,data = result_queue.get()
            cv2.imshow(result_name, data)
            cv2.waitKey(1)
        

