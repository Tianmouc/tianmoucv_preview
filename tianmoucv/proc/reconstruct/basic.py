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


