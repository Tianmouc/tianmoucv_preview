import os
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
import torch
import time

from .basic import *
from tianmoucv.isp import *
from tianmoucv.proc.nn.raft_models.raft_modified import RAFT_Mo
from tianmoucv.tools import check_url_or_local_path,download_file
from tianmoucv.proc.nn.utils import tdiff_split

def create_raft_model(dim1=2,dim2=2,istiny = False):
    import argparse
    parser = argparse.ArgumentParser([])
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 640])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--f_in_channel', type=int, default=dim1)
    parser.add_argument('--c_in_channel', type=int, default=dim2)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    default = ['--add_noise']
    if istiny:
        default.append('--small')
    return parser.parse_args(args=default)
    
class TianmoucOF_RAFT(nn.Module):
    '''
    计算稠密光流的nn方法
    默认权重存储于'weight/of_raft_ver_best.ckpt'
    或初始化时指定ckpt_path

    parameter:
        :param imgsize: (w,h),list
        :param ckpt\_path: string, path to weight dictionary

    '''
    #temp network
    def __init__(self,ckpt_path = None, train= False,_optim=True):
        super(TianmoucOF_RAFT, self).__init__()
        current_dir=os.path.dirname(__file__)
        
        if ckpt_path is None:
            ckpt_path = 'http://www.tianmouc.cn:38328/index.php/s/EjnowKZyXb3wzWR/download/of_raft2024-07-19_32.1.ckpt'
        status = check_url_or_local_path(ckpt_path)
        print('loading..:',ckpt_path)
        if status == 1:
            default_file_name = 'of_raft_ver_best.ckpt'
            if not os.path.exists(default_file_name):
                ckpt_path = download_file(url=ckpt_path,file_name=default_file_name)
            else:
                ckpt_path = default_file_name
            print('load finished')
            
        self.args = create_raft_model(2,2,istiny = False)
        self.flowComp = RAFT_Mo(self.args)
        self.H,self.W = (-1,-1)
        
        dict1 = torch.load(ckpt_path, map_location=torch.device('cpu'),weights_only=False)
        dict1 = dict1['state_dict_ReconModel']
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
        if not train:
            self.eval()
        for param in self.flowComp.parameters():
            param.requires_grad = False
            
        main_version = int(torch.__version__[0])
        if main_version==2 and _optim:
            print('compiling model for pytorch version>= 2.0.0')
            self.flowComp = torch.compile(self.flowComp)
            print('compiled!')

    @torch.no_grad() 
    def forward_time_range(self, tsdiff: torch.Tensor, t1, t2,iters=20,print_fps = True):
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
        
        TD_t_0 = tdiff_split(TD_0_t,cdim=1)

        for param in self.flowComp.parameters():
            self.device = param.device
            break
        if len(TD_0_t.shape)==3:
            TD_0_t = TD_0_t.unsqueeze(0)
            SD0 = SD0.unsqueeze(0)
            SD1 = SD1.unsqueeze(0)
        # Part1. warp
        stime = time.time()
        Flow_1_0 = self.flowComp(SD0.to(self.device), SD1.to(self.device), TD_t_0.to(self.device), iters=iters, flow_init=None, test_mode= True) #输出值0~1
        etime = time.time()
        frameTime = etime - stime
        if print_fps:
            print(1/frameTime, 'batch/fps')
        return Flow_1_0

    @torch.no_grad()       
    def backWarp(self, img, flow, dim=3):
        # Extract horizontal and vertical flows.
        if self.H!= img.shape[-2] or self.W!= img.shape[-1]:
            self.H, self.W = img.shape[-2:]
            self.gridX, self.gridY = np.meshgrid(np.arange(self.W), np.arange(self.H))
            self.gridX = torch.tensor(self.gridX, requires_grad=False, device=flow.device)
            self.gridY = torch.tensor(self.gridY, requires_grad=False, device=flow.device)
            
        self.gridX = self.gridX.to(flow.device)
        self.gridY = self.gridY.to(flow.device)
            
        h,w = flow.shape[-2:]
        if w < self.W:
            flow = F.interpolate(flow,(self.H,self.W),mode='bilinear')
        MAGIC_num = 0.5
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u + MAGIC_num
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v + MAGIC_num
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=dim).to(flow.device)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid,align_corners=False)
        return imgOut 
        
    @torch.no_grad() 
    def forward(self,TD_0_t: torch.Tensor,SD0: torch.Tensor,SD1: torch.Tensor,iters=20,print_fps = True):
        '''
        Args:
          @tsdiff: [c,n,w,h], -1~1,torch，decoder的输出直接concate的结果
          
          @t1,t2 \in [0,n]  calculate the OF between t1-t2
          
        '''
        for param in self.flowComp.parameters():
            self.device = param.device
            break
        if len(TD_0_t.shape)==3:
            TD_0_t = TD_0_t.unsqueeze(0)
            SD0 = SD0.unsqueeze(0)
            SD1 = SD1.unsqueeze(0)
            
        TD_t_0 = tdiff_split(TD_0_t,cdim=1)

        TD_t_0 = -1 * TD_0_t 
        # Part1. warp
        stime = time.time()
        flow_init = None
        Flow_1_0 = self.flowComp(SD0.to(self.device), SD1.to(self.device), TD_t_0.to(self.device), iters=iters, flow_init=flow_init, test_mode= True) #输出值0~1
        etime = time.time()
        frameTime = etime - stime
        if print_fps:
            print(1/frameTime, 'batch/fps')
        return Flow_1_0