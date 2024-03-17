import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from ..isp import fourdirection2xy,poisson_blend,upsampleTSD
from .modules import UNetRecon


def grayReconstructor(tsdiff,F0,F1,t,TDnoise=0,threshGate=4/255):
    '''
    AOP+COP合成灰度
    
    1. 校正TD的正向和负向差分的不一致性
    
    2. 计算AOP到COP的线性缩放系数
    
    3. SD使用泊松blending合成灰度
    
    4. 双向TD积累+SD灰度合成最终结果
    
    parameter:
        :param F0: [h,w,3],torch.Tensor
        :param F0: [h,w,3],torch.Tensor
        :param tsdiff: [3,T,h,w],torch.Tensor, 默认decoded结果的堆积
        :param TDnoise: 噪声矩阵 [h,w], torch.Tensor
        :param threshGate=4/255: 积累时的噪声阈值
        :param t: int

    '''
    gray0 = torch.mean(F0,dim=-1)
    gray1 = torch.mean(F1,dim=-1)
    TD_COP = gray1-gray0
    
    #adjust TD bias for tianmouc
    TD = tsdiff[0,:,...]     
    TD[abs(TD)<threshGate]=0
    
    possum  = torch.sum(TD[TD>0])
    negsum  = torch.sum(abs(TD[TD<0]))
    bias = (negsum-possum)/TD[TD>0].view(1,-1).shape[1]
    TD[TD>0] += bias
    
    TD = tsdiff[0,...] - TDnoise

    AOPDiff = torch.sum(TD[1:,...],dim=0)
    AOPDiff = F.interpolate(AOPDiff.unsqueeze(0).unsqueeze(0), size=TD_COP.shape, mode='nearest').squeeze(0).squeeze(0)
    AOP_COP_scale_neg = torch.sum(TD_COP[TD_COP<0])/torch.sum(AOPDiff[AOPDiff<0]) 
    AOP_COP_scale_pos = torch.sum(TD_COP[TD_COP>0])/torch.sum(AOPDiff[AOPDiff>0]) 

    TD[TD<0] *= AOP_COP_scale_neg 
    TD[TD>0] *= AOP_COP_scale_pos 
    forward_TD =  torch.sum(TD[1:t+1,...],dim=0)
    backward_TD =  torch.sum(TD[t+1:,...],dim=0)

    '''
    SDt = tsdiff[1:,t,...].permute(1,2,0) * (AOP_COP_scale_neg+AOP_COP_scale_pos)/2

    Ix,Iy = fourdirection2xy(SDt)
    gray = -poisson_blend(Ix,Iy,iteration=20)
    gray = F.interpolate(gray.unsqueeze(0).unsqueeze(0), 
                         size=(320,640), mode='bilinear').squeeze(0).squeeze(0)
    '''
    forward_TD  = F.interpolate(forward_TD.unsqueeze(0).unsqueeze(0), size=(320,640), mode='bilinear').squeeze(0).squeeze(0)
    backward_TD = F.interpolate(backward_TD.unsqueeze(0).unsqueeze(0), size=(320,640), mode='bilinear').squeeze(0).squeeze(0)
    
    hdr = (gray0+forward_TD + gray1 - backward_TD)/2
    
    return hdr


class Reconstrutor_NN(nn.Module):
    '''
    重建网络
    权重链接:https://drive.google.com/file/d/1eWF5mW7ccSjUY93gM7bGxgwGl5z1IdlM/view?usp=share_link
    '''

    def __init__(self,ckpt_path = None):
        super(Reconstrutor_NN, self).__init__()
        current_dir=os.path.dirname(__file__)
        if ckpt_path is None:
            #print('use shared weight:','https://drive.google.com/file/d/1eWF5mW7ccSjUY93gM7bGxgwGl5z1IdlM/view?usp=share_link')
            ckpt_path = os.path.join(current_dir,'weight/direct_0109_with_HDR_GTver_best.ckpt')
        self.reconNet =  UNetRecon(7, 3)
        try:
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
            self.easyrecon = True
        except:
            self.easyrecon = False
            self.reconNet = torch.jit.load(ckpt_path, map_location=torch.device('cpu')) 

        self.eval()
        for param in self.reconNet.parameters():
            param.requires_grad = False
        main_version = int(torch.__version__[0])
        if main_version==2:
            print('compiling model for pytorch version>= 2.0.0')
            self.reconNet = torch.compile(self.reconNet)
            print('compiled!')

    def __call__(self, F0, tsdiff, t):
        if self.easyrecon:
            return self.forward_batch(F0,tsdiff).float()
        else:
            return self.complex_forward_batch(F0,tsdiff).float()

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
        TD_0_t = torch.sum(tsdiff[:,0:1,1:t,...],dim=2)
        
        print(TD_0_t.shape,SD1.shape)

        I_1_rec = self.reconNet(torch.cat([F0,TD_0_t,SD1],dim=1))#3+1

        return I_1_rec 

    @torch.no_grad() 
    def forward_batch(self, F0, tsdiff):
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
        #TD_0_t_b = torch.zeros([n2,1,h,w]).to(self.device)
        #for n in range(1,n2):
        #    TD_0_t_b[n,...] = torch.sum(tsdiff[:,0:1,1:n,...],dim=2)
            
                
        TD_0_t_b = torch.zeros([n2,2,h,w]).to(self.device)
        for n in range(1,n2):
            td_ = tsdiff[:,0:1,1:n+1,...]
            td_pos = td_.clone()
            td_pos[td_pos<0] = 0
            td_pos = torch.sum(td_pos,dim=2)
            td_neg = td_.clone()
            td_neg[td_neg>0] = 0
            td_neg = torch.sum(td_neg,dim=2)
            td = torch.cat([td_pos,td_neg],dim=1)
            TD_0_t_b[n:n+1,...] = td
        
        stime = time.time()
        inputTensor = torch.cat([FO_b,TD_0_t_b,SD1_b],dim=1)
        I_1_rec = self.reconNet(inputTensor)#3+1
        etime = time.time()
        frameTime = (etime-stime)/n2
        print(1/frameTime/n2,'batch`ps',1/frameTime, 'fps')
        return I_1_rec 