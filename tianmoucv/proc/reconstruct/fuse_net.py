import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

from tianmoucv.isp import SD2XY,upsampleTSD
from tianmoucv.tools import check_url_or_local_path,download_file

from tianmoucv.proc.nn.unet_modules import *
from tianmoucv.proc.nn.raft_models.raft_modified import RAFT_Mo
from tianmoucv.proc.nn.utils import tdiff_split
from tianmoucv.proc.nn.unet_modules import UNetRecon
from tianmoucv.proc.opticalflow.sdraft_net import create_raft_model

from .basic import batch_inference
from .xmem_util import get_affinity,readout

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

class MergeNet_mem(nn.Module):
    #version primier: add short-term mem History_frames
    def __init__(self, inChannels=3,outChennel=3):
        super(MergeNet_mem, self).__init__()
        self.encoder1 = DownNet(3, fac=2)
        self.encoder2 = DownNet(3, fac=2)
        self.encoder3 = DownNet(5, fac=2)
        self.fusion = FuseNet_with_Mem(outChennel,fac=2)
        
    def forward(self,pathway1,pathway2,state_data):
        
        feature1 = self.encoder1(pathway1)
        feature2 = self.encoder2(pathway2)
        feature_state = self.encoder3(state_data)
        frame, ST4x, ST2x, ST1x, M8 ,M4, M2, M1, affinity_list = self.fusion(feature1,feature2,feature_state)
        return frame, M8 ,M4, M2, M1, affinity_list
    
    
class DownNet(nn.Module):
    def __init__(self, inChannels, fac = 1):
        super(DownNet, self).__init__()
        self.conv1 = nn.Sequential(nn.ReplicationPad2d([1,1,1,1]),
                                   nn.Conv2d(in_channels=inChannels, out_channels=32//fac, kernel_size=3),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=True))  
        self.down1 = down(32//fac, 64//fac, 3)
        self.down2 = down(64//fac, 128//fac, 3)
        self.down3 = down(128//fac, 256//fac, 3)
        self.down4 = down(256//fac, 256//fac, 3)

    def forward(self, x0):
        f32 = self.conv1(x0)
        f64 = self.down1(f32)
        f128 = self.down2(f64)
        f256 = self.down3(f128)
        ff = self.down4(f256)
        return ff,f256,f128,f64,f32
    
    
class FuseNet_with_Mem(nn.Module):
    def __init__(self,oc=3,fac=1):
        super(FuseNet_with_Mem, self).__init__()
        self.ConvIn = nn.Conv2d(in_channels=256*3//fac, out_channels=256//fac, kernel_size=1, stride=1, bias=False)
        self.AADBlk1 = FuseBlock(cin=256//fac, cout=128//fac, c_ef=256//fac, debug=False)
        self.AADBlk2 = FuseBlock(cin=128//fac, cout=64//fac, c_ef=128//fac)
        self.AADBlk3 = FuseBlock(cin=64//fac, cout=32//fac, c_ef=64//fac)
        self.AADBlk4 = FuseBlock(cin=32//fac, cout=16//fac, c_ef=32//fac, debug=False)
        self.Up2x = Interp(scale=2)
        self.ItStage1 = nn.Sequential(nn.ReplicationPad2d([1, 1, 1, 1]),
                                      nn.Conv2d(in_channels=16//fac, out_channels=oc, kernel_size=3),
                                      )

        self.mem_dict = nn.ParameterDict([])
        self.mem_dict[str(0)] = nn.Parameter(torch.randn([1,256//fac,128])*1,requires_grad=False) #[1,dim,status]
        self.mem_dict[str(1)] = nn.Parameter(torch.randn([1,256//fac,128])*1,requires_grad=False) #[1,dim,status]
        self.mem_dict[str(2)] = nn.Parameter(torch.randn([1,128//fac,128])*1,requires_grad=False) #[1,dim,status]
        self.mem_dict[str(3)] = nn.Parameter(torch.randn([1,64//fac,128])*1,requires_grad=False)  #[1,dim,status]
        self.mem_dict[str(4)] = nn.Parameter(torch.randn([1,32//fac,128])*1,requires_grad=False)  #[1,dim,status]
        
        self.downSampleList = nn.ModuleList()
        self.downSampleList.append(nn.Identity())
        self.downSampleList.append(nn.Conv2d(256//fac, out_channels=256//fac, kernel_size= 3, stride= 2, padding  = 1))
        self.downSampleList.append(nn.Conv2d(128//fac, out_channels=128//fac, kernel_size= 5, stride= 4, padding  = 1))
        self.downSampleList.append(nn.Conv2d(64//fac, out_channels=64//fac, kernel_size= 9, stride= 8, padding  = 1))
        self.downSampleList.append(nn.Conv2d(32//fac, out_channels=32//fac, kernel_size= 17, stride= 16, padding  = 1))

        self.upSampleList = nn.ModuleList()
        self.upSampleList.append(nn.Identity())
        self.upSampleList.append(nn.ConvTranspose2d(256//fac,256//fac,kernel_size=3,stride = 2,padding=1,output_padding=1))
        self.upSampleList.append(nn.ConvTranspose2d(128//fac,128//fac,kernel_size=5,stride = 4,padding=1,output_padding=1))
        self.upSampleList.append(nn.ConvTranspose2d(64//fac,64//fac,kernel_size=9, stride = 8,padding=1,output_padding=1))
        self.upSampleList.append(nn.ConvTranspose2d(32//fac,32//fac,kernel_size=17,stride = 16,padding=1,output_padding=1))
        
        self.Normlist = nn.ModuleList()
        self.Normlist.append(nn.LayerNorm(256//fac,1))
        self.Normlist.append(nn.LayerNorm(256//fac,1))
        self.Normlist.append(nn.LayerNorm(128//fac,1))
        self.Normlist.append(nn.LayerNorm(64//fac,1))
        self.Normlist.append(nn.LayerNorm(32//fac,1)) 

        self.print_once = True 
        self.count = 0
        self.mem_scale = 1


    def close_mem(self):
        self.mem_scale = 0

    def _debug_memory(self):
        return self.mem_dict

    def direct_mem_readout(self,z_,i):

        mem = self.mem_dict[str(i)]
        z = self.downSampleList[i](z_)

        b,dim,h,w= z.shape
        bm,dm,lm = mem.shape

        z = z.flatten(start_dim=2)
       
        key = torch.cat([mem]*b,dim=0)
        key = key.permute(0,2,1)      
        query = z.permute(0,2,1)       
        value = key
        query = self.Normlist[i](query)
        affinity = get_affinity(query, None, key, None)
        
        memory = readout(affinity, value)
        memory = memory.permute(0,2,1)
        memory = memory.view(b, dim, h, w)
                    
        memory = self.upSampleList[i](memory)

        return memory,affinity

    def get_mem_constrain(self):
        '''
        魔改
        embedding_loss0 加入unet mem
        让mem的std尽可能保留在1附近
        '''
        memstd = [torch.mean(torch.var(self.mem_dict[e],dim=1),dim=1) for e in self.mem_dict]
        mem_std_mean = 0
        for i in range(len(memstd)):
            mem_std_mean += memstd[i]/len(memstd)
        ones = torch.ones([1]).to(mem_std_mean.device)
        return MSE_LossFn(ones,mem_std_mean)


    def forward(self, z_e, z_f, z_state):
        
        affinity_list = []
        new_z_list = []
        self.count += 1
        
        for i in range(5):
            z_state_mem, affinity = self.direct_mem_readout(z_state[i],i)
            new_z_list.append(z_state[i] + z_state_mem)
            affinity_list.append(affinity.cpu().detach())
        
        ST16x = self.ConvIn(torch.cat([z_e[0], new_z_list[0], z_f[0]], dim=1))# 64
        ST8x,M8 = self.AADBlk1(self.Up2x(ST16x),z_e[1], z_f[1], new_z_list[1])# 32
        ST4x,M4 = self.AADBlk2(self.Up2x(ST8x), z_e[2], z_f[2], new_z_list[2])# 16
        ST2x,M2 = self.AADBlk3(self.Up2x(ST4x), z_e[3], z_f[3], new_z_list[3])# 8 
        ST1x,M1 = self.AADBlk4(self.Up2x(ST2x), z_e[4], z_f[4], new_z_list[4])# 4 
        ItStage1 = self.ItStage1(ST1x)

        if(self.print_once):
            self.print_once = False
                    
        return ItStage1, ST4x, ST2x, ST1x, M8 ,M4, M2, M1, affinity_list

    
    
class TianmoucRecon_mem(nn.Module):
    '''
    重建网络
    #20240515version
    '''
    def __init__(self,ckpt_path =None,_optim=True):
        super(TianmoucRecon_mem, self).__init__()
        current_dir=os.path.dirname(__file__)
        if ckpt_path is None:
            ckpt_path = 'http://www.tianmouc.cn:38328/index.php/s/PJFGZqfgREJiwrZ/download/unet_mem2024-08-21_extreme_34.49.ckpt'
        
        self.syncComp = MergeNet_mem(3)
        self.args = create_raft_model(2,2,istiny = False)
        self.flowComp = RAFT_Mo(self.args)
        self.reconNet =  UNetRecon(3+2+2, 3)

        status = check_url_or_local_path(ckpt_path)
        
        if status == 1:
            default_file_name = 'unet_mem_ver_best.ckpt'
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
        dict1 = torch.load(ckpt, map_location=torch.device('cpu'),weights_only=False)
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
            if 'reconNet' in key:
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
        I_1_warp = 0
        # Part1. warp
        flow_low, flow_up = self.flowComp(SD0, SD1, TFlow_1_0, iters=20, test_mode= True)
        I_1_warp = self.backWarp(F0, flow_up)
        Flow_1_0 = flow_up
        
        I_1_rec = self.reconNet(torch.cat([F0,TFlow_0_1,SD1],dim=1))#3+1
        # part3. merge
        intensity = torch.mean(F0,dim=1).unsqueeze(1)
        guidance = torch.cat([SD1,intensity,Flow_1_0],dim=1)
            
        I_t_p, M8 ,M4, M2, M1, affinity_list = self.syncComp(I_1_rec,I_1_warp,guidance)#3+3+2+1
        M_list = [M8.detach() ,M4.detach(), M2.detach(), M1.detach()]
        emb_loss = 0

        return I_t_p,Flow_1_0,I_1_rec,I_1_warp,M_list,emb_loss

    
    def __call__(self, sample, bs=26, h=320, w = 640):
        '''
        adjust bs if your graph mem is not enough
        '''
        self.device = self.reconNet.up5.conv2.weight.device
        stime = time.time()
        Ft = batch_inference(sample,self.forward_batch,
                    model='unet_mem',
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