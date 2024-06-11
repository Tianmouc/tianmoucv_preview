import os
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
import torch
import time

from .basic import *
from tianmoucv.isp import *
from tianmoucv.proc.nn.spy_modules import *
from tianmoucv.tools import check_url_or_local_path,download_file

class TianmoucOF_SpyNet(nn.Module):
    '''
    计算稠密光流的nn方法
    默认权重存储于'of_0918_ver_best.ckpt'
    或初始化时指定ckpt_path
    
    parameter:
    
    :param imgsize: (w,h),list
    :param ckpt\_path: string, path to weight dictionary

    '''
    #temp network
    def __init__(self,imgsize,ckpt_path = None, _optim=True):
        super(TianmoucOF_SpyNet, self).__init__()
        current_dir=os.path.dirname(__file__)
        
        if ckpt_path is None:
            ckpt_path = 'https://cloud.tsinghua.edu.cn/f/84ac6e32060443e2975d/?dl=1'
        status = check_url_or_local_path(ckpt_path)
        print('loading..:',ckpt_path)
        if status == 1:
            default_file_name = 'of_spy_ver_best.ckpt'
            if not os.path.exists(default_file_name):
                ckpt_path = download_file(url=ckpt_path,file_name=default_file_name)
            else:
                ckpt_path = default_file_name
            print('load finished')
                
        self.flowComp = SpyNet(dim=1+2+2)
        self.W, self.H = imgsize
        self.gridX, self.gridY = np.meshgrid(np.arange(self.W), np.arange(self.H))
        
        dict1 = torch.load(ckpt_path, map_location=torch.device('cpu'))
        dict1 = dict1['state_dict_OF']
        dict_flowComp = dict([])
        for key in dict1:
            new_key_list = key.split('.')[1:]
            new_key = ''
            for e in new_key_list:
                new_key += e + '.'
            new_key = new_key[:-1]
            if 'flowComp' in key:
                dict_flowComp[new_key] = dict1[key] 
        self.flowComp.load_state_dict(dict_flowComp,strict=True)
        self.eval()
        for param in self.flowComp.parameters():
            param.requires_grad = False
            
        main_version = int(torch.__version__[0])
        if main_version==2 and _optim:
            print('compiling model for pytorch version>= 2.0.0')
            self.flowComp = torch.compile(self.flowComp)
            print('compiled!')

    @torch.no_grad() 
    def forward_time_range(self, tsdiff: torch.Tensor,t1,t2,F0=None):
        '''
        Args:
          @tsdiff: [c,n,w,h], -1~1,torch，decoder的输出直接concate的结果
          
          @t1,t2 \in [0,n]  calculate the OF between t1-t2
          
        '''
        
        for param in self.flowComp.parameters():
            self.device = param.device
            break
        if len(tsdiff.shape)==3:
            tsdiff = tsdiff.unsqueeze(0)
        tsdiff = tsdiff.to(self.device)
        TD_0_t = torch.sum(tsdiff[:,0:1,t1:t2,...],dim=2)
        SD0 = tsdiff[:,1:,t1,...]
        SD1 = tsdiff[:,1:,t2,...]
        TD_t_0 = -1 * TD_0_t 
        # Part1. warp
        stime = time.time()
        Flow_1_0 = self.flowComp(TD_t_0, SD0, SD1) #输出值0~1
        etime = time.time()
        frameTime = etime - stime
        print(1/frameTime, 'fps')
        return Flow_1_0
