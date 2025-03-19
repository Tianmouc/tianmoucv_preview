import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

from tianmoucv.isp import SD2XY,upsampleTSD
from tianmoucv.tools import check_url_or_local_path,download_file

from .basic import batch_inference

from tianmoucv.proc.nn.unet_modules import *
from tianmoucv.proc.nn.nature_code import UNet_Original,UNetRecon_Original,SpyNet
    
class TianmoucRecon_Original(nn.Module):

    def __init__(self,ckpt_path =None,_optim=True):
        super(TianmoucRecon_Original, self).__init__()
        current_dir=os.path.dirname(__file__)
        if ckpt_path is None:
            ckpt_path = 'http://www.tianmouc.cn:38328/index.php/s/YELLKofnnmNHwks/download/unet_nature_reconstruction.ckpt'
        
        self.flowComp = SpyNet(dim=1+2+2)
        self.syncComp = UNet_Original(8, 3)
        self.reconNet =  UNetRecon_Original(4,3)
        self.W, self.H = (640,320)
        self.gridX, self.gridY = np.meshgrid(np.arange(self.W), np.arange(self.H))

        status = check_url_or_local_path(ckpt_path)
        
        if status == 1:
            default_file_name = 'unet_nature_ver_best.ckpt'
            if not os.path.exists(default_file_name):
                ckpt_path = download_file(url=ckpt_path,file_name=default_file_name)
            else:
                ckpt_path = default_file_name
            print('load finished')
            
        self.load_model(ckpt=ckpt_path)
        self.eval()
        for param in self.reconNet.parameters():
            param.requires_grad = False
        main_version = int(torch.__version__[0])
        if main_version==2 and _optim:
            print('compiling model for pytorch version>= 2.0.0')
            self.reconNet = torch.compile(self.reconNet)
            print('compiled!')
            
    def load_model(self,ckpt=None):
        '''
            adaptive model parameter loading
            you can just load some of the model for pretraining
                parameter should be stored in 'state_dict_ReconModel'->module dict
        '''
        dict1 = torch.load(ckpt, map_location=torch.device('cpu'))
        dict1 = dict1['state_dict_ReconModel']

        dict_reconNet = dict([])
        dict_flowComp = dict([])
        dict_syncComp = dict([])
        for key in dict1:
            new_key_list = key.split('.')[1:]
            new_key = ''
            for e in new_key_list:
                new_key += e + '.'
            new_key = new_key[:-1]
            if 'AttnNet' in key:
                dict_reconNet[new_key] = dict1[key]
            if 'flowComp' in key:
                dict_flowComp[new_key] = dict1[key]
            if 'syncComp' in key:
                dict_syncComp[new_key] = dict1[key]
                
        if not "TdtoInitFlow" in dict_flowComp:
            print("no TdtoInitFlow")
            dict_flowComp["TdtoInitFlow.1.weight"] = torch.randn([2,2,3,3])
            dict_flowComp["TdtoInitFlow.4.weight"] = torch.randn([2,2,3,3])
        
        self.reconNet.load_state_dict(dict_reconNet,strict=True)
        self.flowComp.load_state_dict(dict_flowComp,strict=False)
        self.syncComp.load_state_dict(dict_syncComp,strict=True)

    
    def backWarp(self, img, flow):
        '''
            warp an image using inverse-direction optical flow
        '''
        # Extract horizontal and vertical flows.
        h,w = img.shape[-2:]
        self.W = w
        self.H = h
        self.gridX, self.gridY = np.meshgrid(np.arange(self.W), np.arange(self.H))

        MAGIC_num = 0.5
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        gridX = torch.tensor(self.gridX, requires_grad=False, device=flow.device)
        gridY = torch.tensor(self.gridY, requires_grad=False, device=flow.device)
        x = gridX.unsqueeze(0).expand_as(u).float() + u + MAGIC_num
        y = gridY.unsqueeze(0).expand_as(v).float() + v + MAGIC_num
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid,align_corners=False)
        return imgOut 
        
    @torch.no_grad() 
    def forward_batch(self, F0, TFlow_0_1, SD0, SD1,y=None):
        TFlow_1_0  = -1 * TFlow_0_1 
        # Part2. warp
        F_1_0 = self.flowComp(TFlow_1_0, SD0, SD1) #
        I_1_warp = self.backWarp(F0, F_1_0)

        # part3. time integration
        I_1_rec = self.reconNet(torch.cat([F0,TFlow_0_1],dim=1))#3+1

        # part4. fusion
        I_t_p = self.syncComp(torch.cat([I_1_rec,I_1_warp,SD1],dim=1))#3+3+2

        return I_t_p,F_1_0,I_1_rec,I_1_warp

    
    def __call__(self, sample, bs=26, h=320, w = 640):
        '''
        adjust bs if your graph mem is not enough
        '''
        self.device = self.reconNet.conv1.weight.device
        stime = time.time()
        Ft = batch_inference(sample,self.forward_batch,
                    model='unet_original',
                    h=h,
                    w=w,
                    device=self.device,
                    ifsingleDirection=False,
                    speedUpRate = 1, bs=bs)
        etime = time.time()
        n2 = 50
        frameTime = (etime-stime)/n2
        print(1/frameTime/n2,'batch`ps',1/frameTime, 'fps in average')
        return Ft