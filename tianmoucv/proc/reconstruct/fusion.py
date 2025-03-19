import numpy as np
import cv2
import scipy.sparse.linalg as sparse_la
from scipy import sparse
import torch

def get_laplacian(Ix,Iy):
    result = np.zeros_like(Ix)
    result[1:-1,1:-1] = 1/4 * (Ix[1:-1,2:] -  Iy[1:-1,1:-1] 
                    + Iy[2:,1:-1] -  Ix[1:-1,1:-1])
    
    return result

def generate_matrix_b(source, Ix, Iy, mask):
    target = np.zeros_like(source)
    target_laplacian_flatten = get_laplacian(Ix,Iy).flatten()
    source_flatten = source.flatten()
    mask_flatten = mask.flatten()
    b = (mask_flatten) * target_laplacian_flatten + (1 - mask_flatten) * source_flatten
    return b

def generate_matrix_A(mask):
    data, cols, rows = [], [], []
    h, w = mask.shape[0], mask.shape[1]
    mask_flatten = mask.flatten()
    zeros = np.where(mask_flatten == 0)
    ones = np.where(mask_flatten == 1)
    # adding ones to data
    n = zeros[0].size
    data.extend(np.ones(n, dtype='float32').tolist())
    rows.extend(zeros[0].tolist())
    cols.extend(zeros[0].tolist())

    # adding 4s to data
    m = ones[0].size
    data.extend((np.ones(m, dtype='float32') * (4)).tolist())
    rows.extend(ones[0].tolist())
    cols.extend(ones[0].tolist())

    # adding -1s
    data.extend((np.ones(m, dtype='float32') * (-1)).tolist())
    rows.extend(ones[0].tolist())
    cols.extend((ones[0] - 1).tolist())

    data.extend((np.ones(m, dtype='float32') * (-1)).tolist())
    rows.extend(ones[0].tolist())
    cols.extend((ones[0] + 1).tolist())

    data.extend((np.ones(m, dtype='float32') * (-1)).tolist())
    rows.extend(ones[0].tolist())
    cols.extend((ones[0] - w).tolist())

    data.extend((np.ones(m, dtype='float32') * (-1)).tolist())
    rows.extend(ones[0].tolist())
    cols.extend((ones[0] + w).tolist())

    cols = [c if c > 0 else 0 for c in cols]
    return data, cols, rows


def solve_sparse_linear_equation(data, cols, rows, b, h, w):
    sparse_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(h * w, h * w), dtype='float32')
    f = sparse_la.spsolve(sparse_matrix, b)
    f = np.reshape(f, (h, w)).astype('float32')
    return f

def poisson_blending_solve_eq(source:np.array, Ix:np.array,Iy:np.array, thresh= 4/128):
    '''
    完整的poisson重建
    效果没有直接laplacian_blending叠加来得好
    https://github.com/nima7920/image-blending
    '''
    mask = ((abs(Ix) + abs(Iy)) > thresh).astype(np.uint8)
    h, w = source.shape[0], source.shape[1]
    source_r, source_g, source_b = cv2.split(source)
    data, cols, rows = generate_matrix_A(mask)
    b_b = generate_matrix_b(source_b, Ix,Iy, mask)
    b_g = generate_matrix_b(source_g, Ix,Iy, mask)
    b_r = generate_matrix_b(source_r, Ix,Iy, mask)
    blended_b = solve_sparse_linear_equation(data, cols, rows, b_b, h, w)
    blended_g = solve_sparse_linear_equation(data, cols, rows, b_g, h, w)
    blended_r = solve_sparse_linear_equation(data, cols, rows, b_r, h, w)
    result = cv2.merge((blended_b, blended_g, blended_r))
    return result

####################################################################################

def laplacian_blending_1c(Ix,Iy,gray,iteration=50):
    '''
    # 灰度重建-不直接调用
    # vectorized by Y. Lin
    # Function to apply lap blending to two images
    # 优化目标:最小化已知梯度u和位置区域f梯度的差异
    # 求极值能得到 \Delta f = div v
    # 迭代法数值求解 gray_0 = 1/4*( g_{N,S,W,E}  - div v）
    '''
    if gray is None:
        gray = torch.zeros_like(Ix)
    gray_iter = gray
    div_v = Ix[1:-1,2:] - Ix[1:-1,1:-1] + Iy[2:,1:-1] - Iy[1:-1,1:-1]
    # Perform Poisson iteration
    for i in range(iteration):
        gray_iter_old = gray_iter.clone()
        gray_iter[1:-1,1:-1] = 1/4 * (gray_iter[2:,1:-1] + 
                                      gray_iter[0:-2,1:-1] + 
                                      gray_iter[1:-1,2:] + 
                                      gray_iter[1:-1,0:-2] -  div_v)
        # Check for convergence
        if torch.sum(torch.abs(gray_iter - gray_iter_old)) < 0.1:
            return gray_iter
    # Return the blended image
    return gray_iter

def laplacian_blending_1c_batch(Ix,Iy,gray=None,iteration=50):
    '''
    # 灰度重建-支持batch的网络训练用接口
    # vectorized by Y. Lin
    # Function to apply Poisson blending to two images
    '''
    if gray is None:
        gray = torch.zeros_like(Ix)
    gray_iter = gray
    # Perform Poisson iteration
    div_v = 1/4 * (Ix[:,0,1:-1,2:] -  Iy[:,0,1:-1,1:-1] 
                    + Iy[:,0,2:,1:-1] -  Ix[:,0,1:-1,1:-1])
    for i in range(iteration):
        gray_iter_old = gray_iter.clone()
        gray_iter[1:-1,1:-1] = 1/4 * (gray_iter[:,0,2:,1:-1] + gray_iter[:,0,0:-2,1:-1] 
                                    + gray_iter[:,0,1:-1,2:] + gray_iter[:,0,1:-1,0:-2] -  div_v)

        # Check for convergence
        if torch.sum(torch.abs(gray_iter - gray_iter_old)) < 0.1:
            return gray_iter
    # Return the blended image
    return gray_iter


def smooth_edges(img):
    # 自定义5x5 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 开运算，先腐蚀再膨胀
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    # 闭运算，先膨胀再腐蚀，使边缘更平滑
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    # 使用高斯滤波进一步平滑边缘
    blurred = cv2.GaussianBlur(closing, (5, 5), sigmaX=1)
    return blurred
    
def genMask(gray,th = 24, maxV=255, minV = 0):
    '''
    生成过欠曝区域遮罩
    '''    
    gap = maxV- minV
    mask_ts = (gray < (maxV-th)/gap).float() #( (gray < (maxV-th)/gap) * (gray > (minV+th)/gap) ).float()
    mask_np = mask_ts.cpu().numpy()
    mask_np_gap = (mask_np * gap).astype(np.uint8)

    if gray.ndim== 3:
        mask_np_b = np.zeros_like(mask_np_gap)
        for c in range(3):
            mask_np_b[:,:,c] = smooth_edges(mask_np_gap[:,:,c])
    else:
        mask_np_b = smooth_edges(mask_np_gap)

    return torch.FloatTensor(mask_np_b).to(gray.device)


def poisson_blending(Ix,Iy,srcimg=None, iteration=20, mask_rgb=False, mask_th = 24, smooth=True):
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
        if smooth:
            mask /= 255.0
            result = (1-mask) * result + mask * srcimg
        else:
            mask = mask>0.5
            mask = mask.bool()
            result[mask] = srcimg[mask]

    return result

print_warning = True

def laplacian_blending(Ix,Iy,srcimg=None, iteration=20, mask_rgb=False, mask_th = 24):
    global print_warning
    if print_warning:
        print('this function\'s name has been changed to poisson_blending for a more accurate des')
        print_warning = False
    result = poisson_blending(Ix,Iy,srcimg=srcimg, iteration=iteration, mask_rgb=mask_rgb, mask_th = mask_th)
    return result

                                                                            