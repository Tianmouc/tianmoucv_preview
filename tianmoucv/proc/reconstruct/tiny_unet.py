import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

from .basic import batch_inference

from tianmoucv.tools import check_url_or_local_path,download_file
from tianmoucv.isp import upsampleTSD
from tianmoucv.proc.nn.utils import tdiff_split,spilt_and_adjust_td_batch
from tianmoucv.proc.nn.unet_modules import UNetRecon

class TianmoucRecon_tiny(nn.Module):
    '''
    重建网络 updated direct 2024-08-19
    '''
    def __init__(self,ckpt_path =None,_optim=True):
        super(TianmoucRecon_tiny, self).__init__()
        current_dir=os.path.dirname(__file__)
        
        if ckpt_path is None:
            ckpt_path = 'http://www.tianmouc.cn:38328/index.php/s/TzRQN96sq3Ag7EP/download/direct2024-08-19_extreme_32.7.ckpt'
            
        self.reconNet =  UNetRecon(7, 3)
        status = check_url_or_local_path(ckpt_path)
        print('loading..:',ckpt_path)
        if status == 1:
            default_file_name = 'tinyunet_best.ckpt'
            if not os.path.exists(default_file_name):
                ckpt_path = download_file(url=ckpt_path,file_name=default_file_name)
            else:
                ckpt_path = default_file_name
            print('load finished')
            
        dict_re = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict_ReconModel']
        dict_reconNet = dict([])
        for key in dict_re:
            new_key_list = key.split('.')[1:]
            new_key = ''
            for e in new_key_list:
                new_key += e + '.'
            new_key = new_key[:-1]
            if 'reconNet' in key:
                dict_reconNet[new_key] = dict_re[key]
        self.reconNet.load_state_dict(dict_reconNet,strict=True)

        self.eval()
        for param in self.reconNet.parameters():
            param.requires_grad = False
        main_version = int(torch.__version__[0])
        if main_version==2 and _optim:
            print('compiling model for pytorch version>= 2.0.0')
            self.reconNet = torch.compile(self.reconNet)
            print('compiled!')

    
    def __call__(self, sample, w=640,h=320,ifSingleDirection=False, bs=32):
        if ifSingleDirection:
            return self.forward_batch_direct(sample,bs=bs, w=w,h=h).float()
        else:
            return self.forward_batch_dual(sample,bs=bs, w=w,h=h).float()

    @torch.no_grad() 
    def forward_batch_direct(self, sample, bs=32, w=640,h=320):
        '''
            recontruct a batch
            @ tsdiff: [c,n,w,h], -1~1,torch
            @ F0:   [w,h,c],torch
        '''
        self.device = self.reconNet.up5.conv2.weight.device
        stime = time.time()
        Ft = batch_inference(sample,self.forward_batch,
                    model='direct',
                    h=h,
                    w=w,
                    device=self.device,
                    ifsingleDirection=True,
                    speedUpRate = 1, bs=bs)
        etime = time.time()
        frameTime = (etime-stime)/Ft.shape[0]
        print(1/frameTime/Ft.shape[0],'batch`ps',1/frameTime, 'fps in average')
        return Ft 

    @torch.no_grad() 
    def forward_batch_dual(self,sample,bs=32, w=640,h=320):
        '''
            recontruct a batch
            @ tsdiff: [c,n,w,h], -1~1,torch
            @ F0:   [w,h,c],torch
        '''
        self.device = self.reconNet.up5.conv2.weight.device
        stime = time.time()
        Ft = batch_inference(sample,self.forward_batch,
                    model='direct',
                    h=h,
                    w=w,
                    device=self.device,
                    ifsingleDirection=False,
                    speedUpRate = 1, bs=bs)
        etime = time.time()
        frameTime = (etime-stime)/Ft.shape[0]
        print(1/frameTime/Ft.shape[0],'batch`ps',1/frameTime, 'fps in average')
        return Ft 

    @torch.no_grad() 
    def forward_batch(self, F0, TFlow_0_1, SD0, SD1):
        '''
            recontruct a batch
            
            @ tsdiff: [c,n,w,h], -1~1,torch
            
            @ F0:   [w,h,c],torch
        '''
        Ft = self.reconNet(torch.cat([F0,TFlow_0_1,SD1],dim=1))#3+1
        return Ft,0,0,0   