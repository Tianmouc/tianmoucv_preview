import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

from .basic import laplacian_blending

from tianmoucv.tools import check_url_or_local_path,download_file
from tianmoucv.proc.nn.unet_modules import UNetRecon
from tianmoucv.isp import upsampleTSD
from tianmoucv.proc.nn.utils import tdiff_split

class TianmoucRecon_tiny(nn.Module):
    '''
    重建网络 updated direct 2024-05-15
    '''
    def __init__(self,ckpt_path =None,_optim=True):
        super(TianmoucRecon_tiny, self).__init__()
        current_dir=os.path.dirname(__file__)
        
        if ckpt_path is None:
            ckpt_path = 'https://cloud.tsinghua.edu.cn/f/dcbaea7004854939b5ec/?dl=1'
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

    def __call__(self, F0, tsdiff, t):
        if t == -1:
            return self.forward_batch_direct(F0,tsdiff).float()
        else:
            return self.forward_single_t(F0, tsdiff, t).float()

    @torch.no_grad() 
    def forward_single_t(self, F0, tsdiff, t):
        '''
          recontruct a frame
          
          @tsdiff: [c,n,w,h], -1~1,torch
          
          @F0:   [w,h,c],torch
          
          @t \in [0,n]
        '''
        self.device = self.reconNet.up5.conv2.weight.device
        
        F0 = F0.permute(2,0,1)
        c,h,w = F0.shape
        c2,n2,h2,w2 = tsdiff.shape
        if h!=h2:
            tsdiff = upsampleTSD(tsdiff)
            tsdiff = F.interpolate(tsdiff, size=(h,w), mode='bilinear')
            
        F0 = F0.unsqueeze(0).to(self.device)
        tsdiff = tsdiff.unsqueeze(0).to(self.device)
            
        SD1 = tsdiff[:,1:,t,...]
        TD_0_t = tsdiff[:,0:1,1:t,...]
        
        TD_0_t = tdiff_split(TD_0_t,cdim=1)#splie pos and neg

        I_1_rec = self.reconNet(torch.cat([F0,TD_0_t,SD1],dim=1))#3+1

        return I_1_rec 

    @torch.no_grad() 
    def forward_batch_direct(self, F0, tsdiff):
        '''
            recontruct a batch
            
            @ tsdiff: [c,n,w,h], -1~1,torch
            
            @ F0:   [w,h,c],torch
        '''
        self.device = self.reconNet.up5.conv2.weight.device
        F0 = F0.permute(2,0,1)
        c,h,w = F0.shape
        c2,n2,h2,w2 = tsdiff.shape
        if h!=h2:
            tsdiff = upsampleTSD(tsdiff)   
            tsdiff = F.interpolate(tsdiff, size=(h,w), mode='bilinear')

        F0 = F0.unsqueeze(0).to(self.device)
        tsdiff = tsdiff.unsqueeze(0).to(self.device)#[b,c,n,w,h]
            
        FO_b = torch.stack([F0[0,...]]*n2,dim=0)
        SD1_b = tsdiff[0,1:,:,...].permute(1,0,2,3)

        TD_0_t_b = torch.zeros([n2,2,h,w]).to(self.device)
        for n in range(1,n2):
            td_ = tsdiff[:,0:1,1:n+1,...]
            td = tdiff_split(td_,cdim=1)
            TD_0_t_b[n:n+1,...] = td
        
        stime = time.time()
        inputTensor = torch.cat([FO_b,TD_0_t_b,SD1_b],dim=1)
        I_1_rec = self.reconNet(inputTensor)#3+1
        etime = time.time()
        frameTime = (etime-stime)/n2
        print(1/frameTime/n2,'batch`ps',1/frameTime, 'fps in average')
        return I_1_rec 

    @torch.no_grad() 
    def forward_batch(self, F0, TFlow_0_1, SD0, SD1):
        '''
            recontruct a batch
            
            @ tsdiff: [c,n,w,h], -1~1,torch
            
            @ F0:   [w,h,c],torch
        '''
        self.device = self.reconNet.up5.conv2.weight.device
        stime = time.time()
        I_1_rec = self.reconNet(torch.cat([F0,TFlow_0_1,SD1],dim=1))#3+1
        etime = time.time()
        frameTime = (etime-stime)/n2
        print(1/frameTime/n2,'batch`ps',1/frameTime, 'fps in average')
        return I_1_rec 