#与NN算法有关

import torch.nn as nn
import numpy as np
from scipy import signal
from PIL import Image
from tianmoucv import *
import cv2
import sys
import torch.nn.functional as F
import torch
from .isp import *

# ===============================================================
# bbox的IOU计算，无batch
# ===============================================================
def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0
    
    
#####################   
#sample input，固定Itter=25，TODO
#@F0:[1,3,w,h],0~1
#@F0:[1,3,w,h],0~1
#@F0:[1,3,w,h],-1~1
#####################
def warp_fast(sample,ReconModel,h=320,w=640,device=torch.device('cuda:0'),
              ifsingleDirection=False,speedUpRate = 1,batchSize=25):  
    F0 = sample['F0']
    F1 = sample['F1']
    tsdiff = sample['tsdiff']
    biasw = (640-w)//2
    biash = (320-h)//2
    timeLen = tsdiff.shape[2]
    #store results
    Ft = torch.zeros([timeLen,3,h,w])
    Ft_reco = torch.zeros([timeLen,3,h,w])
    Ft_warp = torch.zeros([timeLen,3,h,w])
    td_batch_inverse = torch.zeros([timeLen,1,h,w])
    tsdiff = tsdiff[:,:,:,biash:h+biash,biasw:w+biasw]
    tsdiff[abs(tsdiff)<2/128]=0
    F = torch.cat([F0[:,:,biash:h+biash,biasw:w+biasw]]*timeLen,dim=0)
    #tsdiff 有k帧，重建K-1帧，即1~K,第0帧由上一个group重建
    batch = int(np.ceil(timeLen/batchSize))
    print('reconstruction batch:',batch,batchSize,timeLen)
    for b in range(batch):
        biast = b * batchSize
        res = min(timeLen-biast,batchSize)
        print('this is :',b, 'batch, reconstruct:',biast+0,'~',biast+res-1)
      
        F_batch = torch.zeros([res,3,h,w])
        SD0_batch = torch.zeros([res,2,h,w])
        SD1_batch = torch.zeros([res,2,h,w])
        td_batch = torch.zeros([res,1,h,w])
        for rawt in range(res):#F0->F1-dt
            t = rawt*speedUpRate
            SD0_batch[rawt,...] = tsdiff[:,1:,0,...]
            SD1_batch[rawt,...] = tsdiff[:,1:,biast+t,...]
            F_batch[rawt,...] = F0[:,:,biash:h+biash,biasw:w+biasw]
            if biast+t == 0:
                td_batch[rawt,...] = 0
            else:
                td_batch[rawt,...] = torch.sum(tsdiff[:,0:1,1:biast+t,...],dim=2)   
        #print('concrete time:',biast+t)
        Ft1,_,I_1_rec,I_1_warp  = ReconModel(F_batch.to(device) , td_batch.to(device), 
                                     SD0_batch.to(device), SD1_batch.to(device))      
        if not ifsingleDirection:
            for rawt in range(res):#F0->F1-dt
                t = rawt*speedUpRate
                SD0_batch[rawt,...] = tsdiff[:,1:,-1,...]
                SD1_batch[rawt,...] = tsdiff[:,1:,biast+t,...]
                F_batch[rawt,...] = F1[:,:,biash:h+biash,biasw:w+biasw]
                td_batch[rawt,...] = torch.sum(tsdiff[:,0:1,biast+t:,...],dim=2) * -1   
            Ft2,_,I_2_rec,I_2_warp  = ReconModel(F_batch.to(device) , td_batch.to(device), 
                                     SD0_batch.to(device), SD1_batch.to(device))
            Ft1 = (Ft1+Ft2)/2
            I_1_rec = (I_1_rec+I_2_rec)/2
            I_1_warp = (I_1_warp+I_2_warp)/2

        Ft[biast:biast+res,...] = Ft1.clone()
        Ft_reco[biast:biast+res,...] = I_1_rec.clone()
        Ft_warp[biast:biast+res,...] = I_1_warp.clone()

    return Ft,F,tsdiff,Ft_reco,Ft_warp

def warp_fast_unet(sample,ReconModel,h=320,w=640,device=torch.device('cuda:0'),
              ifsingleDirection=False,speedUpRate = 1):  
    F0 = sample['F0']
    F1 = sample['F1']
    tsdiff = sample['tsdiff']
    recon = sample['gray']
    biasw = (640-w)//2
    biash = (320-h)//2
    batchSize = 25
    timeLen = tsdiff.shape[2]
    #store results
    Ft_batch = torch.zeros([timeLen-1,3,h,w])
    Ft_reco_batch = torch.zeros([timeLen-1,3,h,w])
    Ft_warp_batch = torch.zeros([timeLen-1,3,h,w])
    
    F_batch = torch.zeros([batchSize,3,h,w])
    SD0_batch = torch.zeros([batchSize,2,h,w])
    SD1_batch = torch.zeros([batchSize,2,h,w])
    recon1_batch = torch.zeros([batchSize,1,h,w])
    
    td_batch = torch.zeros([batchSize,1,h,w])
    td_batch_inverse = torch.zeros([25,1,h,w])
    tsdiff = tsdiff[:,:,:,biash:h+biash,biasw:w+biasw]
    recon = recon[:,:,:,biash:h+biash,biasw:w+biasw]
    tsdiff[abs(tsdiff)<2/128]=0
    #tsdiff 有k帧，重建K-1帧，即1~K,第0帧由上一个group重建
    batch =  int(np.ceil((timeLen-1)//batchSize))
    print('reconstruction batch:',batch,batchSize,timeLen)
    M_list = None
    for b in range(batch):
        biast = b * batchSize
        res = min(timeLen-biast,batchSize)
        print('this is :',b, 'batch, reconstruct:',biast+0,'~',biast+res-1)
        for rawt in range(res):#F0->F1-dt
            t = rawt*speedUpRate
            SD0_batch[rawt,...] = tsdiff[:,1:,biast+0,...]
            SD1_batch[rawt,...] = tsdiff[:,1:,biast+t,...]
            recon1_batch[rawt,...] = recon[:,:,biast+t,...]
            F_batch[rawt,...] = F0[:,:,biash:h+biash,biasw:w+biasw]
            if t == 0:
                td_batch[rawt,...] = 0
            else:
                td_batch[rawt,...] = torch.sum(tsdiff[:,0:1,1:biast+t,...],dim=2)   
        Ft1,_,I_1_rec,I_1_warp,M_list = ReconModel(F_batch.to(device) , td_batch.to(device), 
                                     SD0_batch.to(device), SD1_batch.to(device))      
        if not ifsingleDirection:
            for rawt in range(res):#F0->F1-dt
                t = rawt*speedUpRate
                SD0_batch[rawt,...] = tsdiff[:,1:,-1,...]
                SD1_batch[rawt,...] = tsdiff[:,1:,biast+t,...]
                recon1_batch[rawt,...] = recon[:,1:,biast+t,...]
                
                F_batch[rawt,...] = F1[:,:,biash:h+biash,biasw:w+biasw]
                td_batch[rawt,...] = torch.sum(tsdiff[:,0:1,biast+t+1:,...],dim=2) * -1   
            Ft2,_,I_2_rec,I_2_warp,M_list  = ReconModel(F_batch.to(device) , td_batch.to(device), 
                                     SD0_batch.to(device), SD1_batch.to(device))
            Ft1 = (Ft1+Ft2)/2
            I_1_rec = (I_1_rec+I_2_rec)/2
            I_1_warp = (I_1_warp+I_2_warp)/2
        Ft_batch[biast:biast+res,...] = Ft1.clone()
        Ft_reco_batch[biast:biast+res,...] = I_1_rec.clone()
        Ft_warp_batch[biast:biast+res,...] = I_1_warp.clone()
    return Ft_batch,F_batch,tsdiff,Ft_reco_batch,Ft_warp_batch,M_list

def warp_unet_dual(sample,ReconModel,h=320,w=640,device=torch.device('cuda:0'),batchSize=25):  
    
    F0 = sample['F0']
    F1 = sample['F1']
    tsdiff = sample['tsdiff']
    biasw = (640-w)//2
    biash = (320-h)//2
    timeLen = tsdiff.shape[2]
    #store results
    Ft = torch.zeros([timeLen,3,h,w])
    Ft_reco = torch.zeros([timeLen,3,h,w])
    Ft_warp = torch.zeros([timeLen,3,h,w])
    td_batch_inverse = torch.zeros([timeLen,1,h,w])
    tsdiff = tsdiff[:,:,:,biash:h+biash,biasw:w+biasw]
    tsdiff[abs(tsdiff)<2/128]=0
    F = torch.cat([F0[:,:,biash:h+biash,biasw:w+biasw]]*timeLen,dim=0)
    #tsdiff 有k帧，重建K-1帧，即1~K,第0帧由上一个group重建
    batch = int(np.ceil((timeLen-1)/batchSize))
    print('reconstruction batch:',batch,batchSize,timeLen)
    for b in range(batch):
        biast = b * batchSize
        res = min(timeLen-biast,batchSize)
        print('this is :',b, 'batch, reconstruct:',biast+0,'~',biast+res-1)
      
        F0_batch = torch.zeros([res,3,h,w])
        F1_batch = torch.zeros([res,3,h,w])
        SD0_batch = torch.zeros([res,2,h,w])
        SD1_batch = torch.zeros([res,2,h,w])
        SDt_batch = torch.zeros([res,2,h,w])
        TD_0_t_batch = torch.zeros([res,1,h,w])
        TD_1_t_batch = torch.zeros([res,1,h,w])
        t_batch = torch.zeros([res,1,1,1])

        for rawt in range(res):#F0->F1-dt
            t = rawt
            t_batch[rawt,...] = (biast+t)/timeLen
            SD0_batch[rawt,...] = tsdiff[:,1:,0,...]
            SDt_batch[rawt,...] = tsdiff[:,1:,biast+t,...]
            SD1_batch[rawt,...] = tsdiff[:,1:,-1,...]

            F0_batch[rawt,...] = F0[:,:,biash:h+biash,biasw:w+biasw]
            F1_batch[rawt,...] = F1[:,:,biash:h+biash,biasw:w+biasw]
            
            if biast+t == 0:
                TD_1_t_batch[rawt,...] = torch.sum(tsdiff[:,0:1,(biast+t)//2+1:,...],dim=2) * -1  #第0帧TD是没用的！
            if biast+t == timeLen-1:
                TD_0_t_batch[rawt,...] = torch.sum(tsdiff[:,0:1,1:(biast+t),...],dim=2)    
            else:
                TD_0_t_batch[rawt,...] = torch.sum(tsdiff[:,0:1,1:(biast+t),...],dim=2)    
                TD_1_t_batch[rawt,...] = torch.sum(tsdiff[:,0:1,(biast+t)+1:,...],dim=2) * -1  #第0帧TD是没用的！
            
           
        Ft_b, Flow_t_0, Flow_t_1, Frec_b, Fwarp0t_b,Fwarp1t_b = ReconModel(F0_batch.to(device),F1_batch.to(device),
                                                                           TD_0_t_batch.to(device), TD_1_t_batch.to(device),
                                                                           SD0_batch.to(device) ,SD1_batch.to(device),
                                                                           SDt_batch.to(device), t=t_batch.to(device))
        Ft[biast:biast+res,...] = Ft_b.clone()
        Ft_reco[biast:biast+res,...] = Frec_b.clone()
        Ft_warp[biast:biast+res,...] = Fwarp0t_b.clone()

    return Ft,F,tsdiff,Ft_reco,Ft_warp

# ===============================================================
# 利用uv输入做变形
# uv : [b,2,w,h]
# img: [b,3,w,h]
# ===============================================================
class backWarp(nn.Module):
    """
    A class for creating a backwarping object.
    This is used for backwarping to an image:
    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).
    ...
    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """
    def __init__(self, W, H, device):
        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)
        
    def forward(self, img, flow):
        # uv有奇怪的偏移
        MAGIC_NUM =  0.5
        # Extract horizontal and vertical flows.
        self.W = flow.size(3)
        self.H = flow.size(2)
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u + MAGIC_NUM
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v + MAGIC_NUM
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut



class opticalDetector_Maxone():
    
    def __init__(self,noiseThresh=8,distanceThresh=0.2):
        self.noiseThresh = noiseThresh
        self.th = distanceThresh
        self.accumU = 0
        self.accumV = 0
        
    def __call__(self,sd,td,ifInterploted = False):
        
        td[abs(td)<self.noiseThresh] = 0
        sd[abs(sd)<self.noiseThresh] = 0
       
        #rawflow = cal_optical_flow(sd,td,win=7,stride=3,mask=None,ifInterploted = ifInterploted)
        #rawflow = recurrentOF(sd,td,ifInterploted = ifInterploted)
        rawflow = recurrentMultiScaleOF(sd,td,ifInterploted = ifInterploted)
        
        flow = flow_to_image(rawflow.permute(1,2,0).numpy())
        
        flowup = np.zeros([flow.shape[0]*2,flow.shape[1]*2,3])
        flowup[1::2,1::2,:] = flow/255.0
        flowup[0::2,1::2,:] = flow/255.0
        flowup[1::2,0::2,:] = flow/255.0
        flowup[0::2,0::2,:] = flow/255.0

        #计算平均速度
        u = rawflow.permute(1,2,0).numpy()[:, :, 0]
        v = rawflow.permute(1,2,0).numpy()[:, :, 1]
        uv = [u,v]

        # case相关，去掉u是正的的那些背景光流
        distance = ((u)**2 + (v)**2) *(u<0)
        
        #和平均光流方向之差
        distance[distance>self.th] = 1
        distance[distance<self.th] = 0
        distanceup = np.zeros([flow.shape[0]*2,flow.shape[1]*2])

        # 膨胀
        kernel = np.ones((3,3),np.uint8)              
        distance = cv2.dilate(distance,kernel,iterations=3) 

        distanceup[1::2,1::2] = distance * 255.0
        distanceup[0::2,1::2] = distance * 255.0
        distanceup[1::2,0::2] = distance * 255.0
        distanceup[0::2,0::2] = distance * 255.0
        f = (distanceup).copy().astype(np.uint8)
        contours,hierarchy = cv2.findContours(f,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(f, contours, -1, (0, 255, 255), 2)
        #找到最大区域并填充
        area = []
        box = None
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))
        if len(area)>0:
            if np.max(area) < 1200:
                return None,distanceup,flowup
            max_idx = np.argmax(area)
            for i in range(max_idx - 1):
                cv2.fillConvexPoly(f, contours[max_idx - 1], 0)
            cv2.fillConvexPoly(f, contours[max_idx], 255)
            #求最大连通域的中心坐标
            maxcon = contours[max_idx]
            x1 = np.min(maxcon[:,:,0])  
            x2 = np.max(maxcon[:,:,0])  
            y1 = np.min(maxcon[:,:,1])  
            y2 = np.max(maxcon[:,:,1])  
            box = [x1,y1,x2,y2]
            #print(u[y1//2:y2//2,x1//2:x2//2]>0)
            #print(u[y1//2:y2//2,x1//2:x2//2],v[y1//2:y2//2,x1//2:x2//2])

        return box,distanceup,flowup