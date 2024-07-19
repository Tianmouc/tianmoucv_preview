import torch
import numpy as np
import cv2
from tianmoucv.proc.nn.utils import tdiff_split

def laplacian_blending_1c(Ix,Iy,gray,iteration=50):
    '''
    # 灰度重建-不直接调用
    # vectorized by Y. Lin
    # Function to apply lap blending to two images
    '''
    if gray is None:
        gray = torch.zeros_like(Ix)
    lap_blend = gray
    # Perform Poisson iteration
    for i in range(iteration):
        lap_blend_old = lap_blend.clone()
        # Update the Laplacian values at each pixel
        grad = 1/4 * (Ix[1:-1,2:] -  Iy[1:-1,1:-1] 
                    + Iy[2:,1:-1] -  Ix[1:-1,1:-1])
        lap_blend_old_tmp = 1/4 * (lap_blend_old[2:,1:-1] + lap_blend_old[0:-2,1:-1] 
                                 + lap_blend_old[1:-1,2:] + lap_blend_old[1:-1,0:-2])

        lap_blend[1:-1,1:-1] = lap_blend_old_tmp + grad
        # Check for convergence
        if torch.sum(torch.abs(lap_blend - lap_blend_old)) < 0.1:
            return lap_blend
    # Return the blended image
    return lap_blend

def laplacian_blending_1c_batch(Ix,Iy,gray=None,iteration=50):
    '''
    # 灰度重建-支持batch的网络训练用接口
    # vectorized by Y. Lin
    # Function to apply Poisson blending to two images
    '''
    if gray is None:
        gray = torch.zeros_like(Ix).to(Ix.device)
    lap_blend = gray
    # Perform Poisson iteration
    for i in range(iteration):
        lap_blend_old = lap_blend.clone()
        # Update the Laplacian values at each pixel
        grad = 1/4 * (Ix[:,0,1:-1,2:] -  Iy[:,0,1:-1,1:-1] 
                    + Iy[:,0,2:,1:-1] -  Ix[:,0,1:-1,1:-1])
        lap_blend_old_tmp = 1/4 * (lap_blend_old[:,0,2:,1:-1] + lap_blend_old[:,0,0:-2,1:-1] 
                                 + lap_blend_old[:,0,1:-1,2:] + lap_blend_old[:,0,1:-1,0:-2])

        lap_blend[:,0,1:-1,1:-1] = lap_blend_old_tmp + grad
        # Check for convergence
        if torch.sum(torch.abs(lap_blend - lap_blend_old)) < 0.1:
            return lap_blend
    # Return the blended image
    return lap_blend


def smooth_edges(img):
    # 自定义5x5 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 开运算，先腐蚀再膨胀
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    # 闭运算，先膨胀再腐蚀，使边缘更平滑
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    # 使用高斯滤波进一步平滑边缘
    blurred = cv2.GaussianBlur(closing, (5, 5), 0)
    return blurred
    
def genMask(gray,th = 24, maxV=255, minV = 0):
    '''
    生成过欠曝区域遮罩
    '''
    gap = maxV- minV
    mask_ts = (gray < (maxV-th)/gap).float() #( (gray < (maxV-th)/gap) * (gray > (minV+th)/gap) ).float()
    mask_np = mask_ts.cpu().numpy()
    mask_np_b = (mask_np * gap).astype(np.uint8)
    mask_np_b = smooth_edges(mask_np_b)
    mask_np = (mask_np_b>0.5)
    return torch.Tensor(mask_np).to(gray.device).bool()

def laplacian_blending(Ix,Iy,srcimg=None, iteration=20, mask_rgb=False, mask_th = 24):
    '''
    RGB/灰度 HDR 融合重建

    vectorized by Y. Lin
    Function to apply Poisson blending to two images
    :Ix,Iy: [h,w]，x和y方向的梯度
    :sciimg: [None],[h,w],[h,w,3]，分别进入不同模式
    '''
    if mask_rgb and not srcimg is None:
        mask = genMask(srcimg, th = mask_th, maxV=255, minV = 0)
    result = None
    
    if srcimg is None:
        result = laplacian_blending_1c(Ix,Iy,gray=srcimg,iteration=iteration)
    elif len(srcimg.shape)==2:
        img = srcimg.clone()
        result = laplacian_blending_1c(Ix,Iy,img,iteration=iteration)
    elif len(srcimg.shape)==3:
        img = srcimg.clone()
        for c in range(img.shape[-1]):
            target = img[...,c]
            img[...,c] = laplacian_blending_1c(Ix,Iy,target,iteration=iteration)
        result = img
    else:
        print('img shape:',srcimg.shape,' is illegal, [None],[H,W],[H,W,C] is supported')

    if mask_rgb and not result is None:
        result[mask] = srcimg[mask]
        
    return result

                                                                            
def batch_inference(model,sample,
                   h=320,
                   w=640,
                   device=torch.device('cuda:0'),
                   ifsingleDirection=False,
                   speedUpRate = 1, bs=1):  
    '''
    model need to implement "result = forward_batch(F_batch, td_batch,SD0_batch,SD1_batch)"
    '''
    with torch.no_grad():
        F0 = sample['F0'].to(device)
        F1 = sample['F1'].to(device)
        tsdiff = sample['tsdiff'].to(device)
        biasw = (640-w)//2
        biash = (320-h)//2
        timeLen = tsdiff.shape[2]
        #store results
        Ft_batch = torch.zeros([timeLen-1,3,h,w]).to(device)
        batchSize = bs
        tsdiff = tsdiff[:,:,:,biash:h+biash,biasw:w+biasw]
        
        print('tsdiff 有',timeLen,'帧，重建',timeLen-1,'帧，第-1帧由下一个group重建')
        print(timeLen,timeLen/batchSize,np.ceil(timeLen/batchSize))
        
        batch =  int(np.ceil(timeLen/batchSize))
        
        for b in range(batch):
            biast = b * batchSize
            res = min(timeLen-biast-1,batchSize)
            if res ==0:
                break
            F_batch = torch.zeros([res,3,h,w]).to(device)
            SD0_batch = torch.zeros([res,2,h,w]).to(device)
            SD1_batch = torch.zeros([res,2,h,w]).to(device)
            recon1_batch = torch.zeros([res,1,h,w]).to(device)
            td_batch = torch.zeros([res,2,h,w]).to(device)
            td_batch_inverse = torch.zeros([res,2,h,w]).to(device)
            for rawt in range(res):#F0->F1-dt
                t = rawt*speedUpRate
                SD0_batch[rawt,...] = tsdiff[:,1:,0,...]
                SD1_batch[rawt,...] = tsdiff[:,1:,biast+t,...]
                F_batch[rawt,...] = F0[:,:,biash:h+biash,biasw:w+biasw]
                    
                if t == 0 and b == 0:
                    td_batch[rawt,...] = 0
                else:
                    TD_0_t = tsdiff[:,0:1,1:t,...]
                    td = tdiff_split(TD_0_t,cdim=1)#splie pos and neg
                    td_batch[rawt,...] = td

            print('finished:',biast,'->',biast+res-1,' ALL:',Ft_batch.size(0))        
            Ft1 = model.forward_batch(F_batch, td_batch,SD0_batch,SD1_batch)
            if not ifsingleDirection and res>=1:
                for rawt in range(res):#F0->F1-dt
                    t = rawt*speedUpRate
                    SD0_batch[rawt,...] = tsdiff[:,1:,-1,...]
                    SD1_batch[rawt,...] = tsdiff[:,1:,biast+t,...]
                    F_batch[rawt,...] = F1[:,:,biash:h+biash,biasw:w+biasw]
                    TD_0_t = tsdiff[:,0:1,1:t,...]
                    td = tdiff_split(TD_0_t,cdim=1)#splie pos and neg
                    td_batch[rawt,...] = td

                print('finished:',biast+t,'->',-1,' ALL:',Ft_batch.size(0))
                Ft2  = model.forward_batch(F_batch, td_batch,SD0_batch,SD1_batch)
                Ft1 = (Ft1+Ft2)/2
                
            Ft_batch[biast:biast+res,...] = Ft1.clone()

        return Ft_batch
    
