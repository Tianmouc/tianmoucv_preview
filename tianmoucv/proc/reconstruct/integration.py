import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

from .basic import laplacian_blending
from tianmoucv.isp import SD2XY,upsampleTSD

def TD_integration(tsdiff,F0,F1,t, TD_BG_NOISE = 0, threshGate=4/255, dig_scaling= 1.5):
    '''
    AOP+COP合成灰度
    
    1. 校正TD的正向和负向差分的不一致性
    
    2. 计算AOP到COP的线性缩放系数
    
    3. laplacian_blending
    
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
    TD = tsdiff[0,1:,...]     

    TD -= TD_BG_NOISE
    TD[abs(TD)<threshGate]=0
    
    possum  = torch.sum(TD[TD>0])
    negsum  = torch.sum(abs(TD[TD<0]))
    bias = (negsum-possum)/TD[TD>0].view(1,-1).shape[1]
    TD[TD>0] += bias

    AOPDiff = torch.sum(TD[1:,...],dim=0)
    AOPDiff = F.interpolate(AOPDiff.unsqueeze(0).unsqueeze(0), size=TD_COP.shape, mode='bilinear').squeeze(0).squeeze(0)
    AOP_COP_scale_neg = torch.sum(TD_COP[TD_COP<0])/torch.sum(AOPDiff[AOPDiff<0]) 
    AOP_COP_scale_pos = torch.sum(TD_COP[TD_COP>0])/torch.sum(AOPDiff[AOPDiff>0]) 

    TD[TD<0] *= AOP_COP_scale_neg * dig_scaling
    TD[TD>0] *= AOP_COP_scale_pos  * dig_scaling

    forward_TD =  torch.sum(TD[0:t,...],dim=0)
    backward_TD =  torch.sum(TD[t:,...],dim=0)


    forward_TD  = F.interpolate(forward_TD.unsqueeze(0).unsqueeze(0), size=(320,640), mode='bilinear').squeeze(0).squeeze(0)
    backward_TD = F.interpolate(backward_TD.unsqueeze(0).unsqueeze(0), size=(320,640), mode='bilinear').squeeze(0).squeeze(0)
    
    hdr =  (gray0+forward_TD + gray1 - backward_TD)/2

    #hdr[hdr<0]=0
    #hdr[hdr>1]=1
    
    return hdr


def SD_integration(SDx:np.array, SDy:np.array)  -> np.array:
    '''
    SD直接积分累加重建，简单可视化用
    use mapped SDx and SDy to conduct direct integration
    '''
    canvas = np.zeros([SDx.shape[0],SDx.shape[1]+1])
    grayy = np.cumsum(SDy,axis=0)
    goody_first = grayy[:,0]
    canvas[:,0] = goody_first
    canvas[:,1:] = SDx
    gray = np.cumsum(canvas[:,:-1],axis=1)
    return gray