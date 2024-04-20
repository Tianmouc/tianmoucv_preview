import os
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
import torch
import time

from .basic import *
from tianmoucv.isp import *

# ===============================================================
# LK方法计算稠密光流 
# ===============================================================
def local_norm(Diff):
    '''
    梯度归一化
    
    parameter:
        :param SD: 待归一化项

    '''

    grad_norm = (Diff[0,...]**2 + Diff[1,...]**2 + 1e-18)**0.5 + 1e-9
    return Diff / torch.max(grad_norm)
        
def cal_optical_flow(SD,TD,win=5,stride=0,mask=None,ifInterploted = False):    
    '''
    LK方法计算稠密光流
    
    .. math:: [dx,dy]*[dI/dx,dI/dy]^T + dI/dt = 0

    parameter:
        :param SD: 原始SD，SD[0,1]: x,y方向上的梯度,[2,h,w],torch.Tensor
        :param TD: 原始SD，TD[0]: t方向上的梯度,[1,h,w],torch.Tensor
        :param win=5: 取邻域做最小二乘,邻域大小
        :param stride=0: 取邻域做最小二乘,计算步长
        :param mask=None: 特征点tensor,binary Tensor,[h,w]
        :param ifInterploted = False: 计算结果是否与COP等大

    '''
    I = SD.size(-2)
    J = SD.size(-1)
    i_step  = win//2
    j_step  = win//2
    if stride == 0:
        stride =  win//2
    flow = torch.zeros([2,I//stride,J//stride])
    
    #加权
    Ix = SD[0,...]
    Iy = SD[1,...]
    It = TD[0,...]

    musk = torch.abs(It)>4
    It *= musk
    for i in range(i_step,I-i_step-1,stride):
        for j in range(j_step,J-j_step-1,stride):
            dxdy = [0,0]
            #忽略一些边界不稠密的点
            if mask is not None and np.sum(mask[i-i_step:i+i_step+1,j-j_step:j+j_step+1]) < 5:
                continue
            #取一个小窗口
            Ix_win = Ix[i-i_step:i+i_step+1,j-j_step:j+j_step+1].reshape(1,-1)
            Iy_win = Iy[i-i_step:i+i_step+1,j-j_step:j+j_step+1].reshape(1,-1)
            It_win = It[i-i_step:i+i_step+1,j-j_step:j+j_step+1].reshape(1,-1)            
            A = torch.cat([Ix_win,Iy_win],dim=0).transpose(1,0)
            B = -1 * It_win.reshape(1,-1).transpose(1,0)
            AT_B = torch.matmul(A.transpose(1,0),B)
            
            if torch.sum(AT_B) == 0:
                flow[0,i//stride,j//stride] = 0
                flow[1,i//stride,j//stride] = 0
                continue
            
            AT_A = torch.matmul(A.transpose(1,0),A)
            
            try :
                dxdy = np.linalg.solve(AT_A.cpu().numpy(), AT_B.cpu().numpy())
                flow[0,i//stride,j//stride] = float(dxdy[0])
                flow[1,i//stride,j//stride] = float(dxdy[1])
            except Exception as e :
                pass
    if not ifInterploted:
        flow = F.interpolate(flow.unsqueeze(0), size=(I,J*2), mode='bilinear').squeeze(0)
    else:
        flow = F.interpolate(flow.unsqueeze(0), size=(I,J), mode='bilinear').squeeze(0)
    return flow

# ===============================================================
# HS方法计算稠密光流，效果更好
# ===============================================================
def recurrentOF(SD,TD,ifInterploted = False):
    '''
    HS方法计算稠密光流
    ![referemce](https://kns.cnki.net/kcms2/article/abstract?v=3uoqIhG8C475KOm_zrgu4h_jQYuCnj_co8vp4jCXSivDpWurecxFtEV8HAD0GySfgFWAxYnv5c-oQfA7zWjworscSCTy1fWb&uniplatform=NZKPT)
    
    .. math:: Energy_term = (u+v+\lambda)^2?
    .. math:: newu = u - Ix * (Ix*u + Iy * v + It) / (\lambda*\lambda + Ix*Ix + Iy*Iy)
    .. math:: newv = v - Iy * (Ix*u + Iy * v + It) / (\lambda*\lambda + Ix*Ix + Iy*Iy)

    parameter:
        :param SD: 原始SD，SD[0,1]: x,y方向上的梯度,[2,h,w],torch.Tensor
        :param TD: 原始SD，TD[0]: t方向上的梯度,[1,h,w],torch.Tensor
        :param ifInterploted = False: 计算结果是否与COP等大

    '''
    epsilon = 1e-8
    maxIteration = 50
    
    def uitter(u,v,Ix,Iy,It,lambdaL):
        newu = u - Ix * (Ix*u + Iy * v + It) / (lambdaL*lambdaL + Ix*Ix + Iy*Iy)
        return newu
    def vitter(u,v,Ix,Iy,It,lambdaL):
        newv = v - Iy * (Ix*u + Iy * v + It) / (lambdaL*lambdaL + Ix*Ix + Iy*Iy)
        return newv
        
    uitter_vector = np.vectorize(uitter)
    vitter_vector = np.vectorize(vitter)
        
    I = SD.size(-2)
    J = SD.size(-1)
    
    #加权
    Ix = SD[0,...].numpy()
    Iy = SD[1,...].numpy()
    It = TD[0,...].numpy()
    
    u = np.zeros([I,J])
    v = np.zeros([I,J])

    lambdaL = np.ones([I,J])
    
    for it in range(maxIteration):
        u_new = uitter_vector(u,v,Ix,Iy,It,lambdaL)
        v_new = vitter_vector(u,v,Ix,Iy,It,lambdaL)
        erroru = abs(u_new-u)
        errorv = abs(v_new-v)
        u = u_new
        v = v_new
        if np.max(erroru) < epsilon and np.max(errorv) < epsilon:
            break
    flow = torch.stack([torch.FloatTensor(u),torch.FloatTensor(v)],dim=0)
    
    if not ifInterploted:
        flow = F.interpolate(flow.unsqueeze(0), size=(I,J*2), mode='bilinear').squeeze(0)
    else:
        flow = F.interpolate(flow.unsqueeze(0), size=(I,J), mode='bilinear').squeeze(0)
    return flow


# ===============================================================
# 多尺度HS方法计算稠密光流，效果更好
# ===============================================================
def recurrentMultiScaleOF(SD,TD,ifInterploted = False,epsilon = 1e-8,maxIteration = 50,scales = 4,labmda=10):
    '''
    多尺度HS方法计算稠密光流，效果更好

    parameter:
        :param SD: 原始SD，SD[0,1]: x,y方向上的梯度,[2,h,w],torch.Tensor
        :param TD: 原始SD，TD[0]: t方向上的梯度,[1,h,w],torch.Tensor
        :param ifInterploted = False: 计算结果是否与COP等大
        :param epsilon = 1e-8: 收敛界
        :param maxIteration = 50: 最大迭代次数
        :param scales = 4: 尺度数量
        :param labmda=10: 惩罚因子,越大光流越平滑

    '''
    ld = labmda
    def uitter(u,v,Ix,Iy,It,lambdaL):
        newu = u - Ix * (Ix*u + Iy * v + It) / (lambdaL*lambdaL + Ix*Ix + Iy*Iy)
        return newu
    def vitter(u,v,Ix,Iy,It,lambdaL):
        newv = v - Iy * (Ix*u + Iy * v + It) / (lambdaL*lambdaL + Ix*Ix + Iy*Iy)
        return newv
        
    uitter_vector = np.vectorize(uitter)
    vitter_vector = np.vectorize(vitter)
        
    I = SD.size(-2)
    J = SD.size(-1)
    
    #加权
    Ix = SD[0,...].numpy()
    Iy = SD[1,...].numpy()
    It = TD[0,...].numpy()

    factor = 2**(scales-1)
    u = np.zeros([I,J])
    v = np.zeros([I,J])
    for s in range(scales):
        factor = 2**(scales-s-1)
        lambdaL = np.ones([J//factor,I//factor]) * ld
        
        #用金字塔算出的邻域结果做初始值
        u =  cv2.resize(u, [I//factor,J//factor], interpolation=cv2.INTER_LINEAR)
        v =  cv2.resize(v, [I//factor,J//factor], interpolation=cv2.INTER_LINEAR)
        Ixs =  cv2.resize(Ix, [I//factor,J//factor], interpolation=cv2.INTER_LINEAR)
        Iys =  cv2.resize(Iy, [I//factor,J//factor], interpolation=cv2.INTER_LINEAR)
        Its =  cv2.resize(It, [I//factor,J//factor], interpolation=cv2.INTER_LINEAR)
        continueFlag = False
        for it in range(maxIteration):
            if continueFlag:
                continue
            #print(u.shape,v.shape,Ixs.shape,Iys.shape,Its.shape,lambdaL.shape)
            u_new = uitter_vector(u,v,Ixs,Iys,Its,lambdaL)
            v_new = vitter_vector(u,v,Ixs,Iys,Its,lambdaL)
            erroru = abs(u_new-u)
            errorv = abs(v_new-v)
            u = u_new
            v = v_new
            if np.max(erroru) < epsilon and np.max(errorv) < epsilon:
                continueFlag = True
    
    flow = torch.stack([torch.FloatTensor(u),torch.FloatTensor(v)],dim=0)
        
    if not ifInterploted:
        flow = F.interpolate(flow.unsqueeze(0), size=(I,J*2), mode='bilinear').squeeze(0)
    else:
        flow = F.interpolate(flow.unsqueeze(0), size=(I,J), mode='bilinear').squeeze(0)
    return flow

    
