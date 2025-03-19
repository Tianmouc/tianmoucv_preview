# author: yihan lin
import cv2
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.nn as nn

import math

#===============================================================
# sobel
# ===============================================================
def sobel_operator(Ix, Iy):
    # 计算梯度幅值
    magnitude = torch.sqrt(torch.pow(Ix, 2) + torch.pow(Iy, 2))
    # 计算梯度角度
    angle = torch.atan2(Iy, Ix)
    return magnitude, angle


# ===============================================================
# 高斯卷积核
# ===============================================================
def gaussain_kernel(size=5,sigma=2):
    '''
    generate Gaussain blur kernel
    
    parameter:
        :param size: 特征的数量，int
        :param sigma: 高斯标准差，int

    '''
    size = int(size)
    if size % 2 == 0:
        size = size + 1
    m = (size - 1) / 2
    y, x = torch.meshgrid(torch.arange(-m, m + 1), torch.arange(-m, m + 1))
    kernel = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel


# ===============================================================
# 用现有高斯核做高斯模糊
# ===============================================================
def gaussian_smooth(inputTensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    '''
    用现有高斯核做高斯模糊
    
    parameter:
        :param inputTensor: 待处理矩阵,torch.Tensor
        :param kernel: 高斯模糊核,torch.Tensor
        :return: 处理后同尺寸模糊图,torch.Tensor

    '''
    padding_size = kernel.shape[-1] // 2
    input_padded = F.pad(inputTensor, (padding_size, padding_size, padding_size, padding_size), mode='reflect')
    kernel = kernel.to(inputTensor.device)
    return F.conv2d(input_padded, kernel, stride=1, padding=0)



# ===============================================================
# Harris角点，用Ix和Iy做计算，可以用SD的两个方向
# ===============================================================
def HarrisCorner(Ix:torch.Tensor,Iy:torch.Tensor, thresh = 0.1,k = 0.1,size=5,sigma=1,nmsSize=11):
    '''
    Harris 角点检测
    
    .. math:: R=det(H)−ktrace(H) 
    .. math:: \lambda1 + \lambda2 = \sum I_x^2 * \sum I_y^2 - \sum I_{xy} ^ 2
    .. math:: \lambda1 * \lambda2 = \sum I_{xy} ^ 2
    .. math:: R = det H - k trace H ^2= \lambda_1*\lambda_2 - k(\lambda_1 + \lambda_2)^ 2
    .. math::   = (\sum I_{X^2} * \sum I_{y^2} - \sum I_{xy} ^ 2) - k (\sum I_x^2 + \sum I_y^2)^2
    
    parameter:
        :param Ix: x方向梯度,[h,w],torch.Tensor
        :param Iy: y方向梯度,[h,w],torch.Tensor
        :param size: 高斯核尺寸,int
        :param th: 控制阈值,范围是0-1,对梯度来说应该设小一点,float
        :param nmsSize: 最大值筛选的范围
        :param k: 是一个经验参数,0-1,float

    '''
    # 1. sober filtering
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    # 3. corner detect
    kernel = gaussain_kernel(size,sigma)
    Ix2 = gaussian_smooth(Ix2.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    Iy2 = gaussian_smooth(Iy2.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    Ixy = gaussian_smooth(Ixy.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)

    R = (Ix2 * Iy2 - Ixy ** 2) - k * ((Ix2 + Iy2) ** 2)
    
    corner_list = []
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i,j] > thresh:
                i_min = max(i-nmsSize//2,0)
                i_max = min(i+nmsSize//2,R.shape[0]-1)
                j_min = max(j-nmsSize//2,0)
                j_max = min(j+nmsSize//2,R.shape[1]-1)
                neibor = R[i_min:i_max,j_min:j_max]
                if R[i,j] >= torch.max(neibor)-1e-5:
                    corner_list.append((i,j))

    print('[tianmoucv.HarrisCorner]detect ', len(corner_list),' corner points')
    return corner_list

    
# ===============================================================
# Shi-Tomasi角点，用Ix和Iy做计算，可以用SD的两个方向
# ===============================================================
def TomasiCorner(Ix:torch.Tensor, Iy:torch.Tensor, index=1000,size=5,sigma=2,nmsSize=11):
    '''
    Shi-Tomasi 角点检测
    在Harris角点检测的基础上，Shi和Tomasi 在1993的一篇论文《Good Features to track》中提出了基于Harris角点检测的Shi-Tomasi方法。
    经验参数需求更少，更快，但效果变差
    
    parameter:
        :param Ix: x方向梯度,[h,w],torch.Tensor
        :param Iy: y方向梯度,[h,w],torch.Tensor
        :param size: 高斯核尺寸,int
        :param index: 前N个点,int
        :param nmsSize: 最大值筛选的范围

    '''
    # 1. get difference image
    Ix[Ix<torch.max(Ix)*0.1]=0
    Iy[Iy<torch.max(Iy)*0.1]=0
    # 2. sober filtering
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    # 3. windowed
    kernel = gaussain_kernel(size,sigma)
    Ix2 = gaussian_smooth(Ix2.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    Iy2 = gaussian_smooth(Iy2.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    Ixy = gaussian_smooth(Ixy.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    # prepare output image
    out = np.zeros(Ix2.shape)
    # get R
    K = Ix2**2 + Iy2 **2 + Iy2*Ix2 + Ixy**2 + 1e-16
    R = Ix2 + Iy2 - torch.sqrt(K)
    # detect corner
    
    sorted_, _ = torch.sort(R.view(1,-1), descending=True)#descending为False，升序，为True，降序
    threshold = sorted_[0,index]
    
    corner_list = []
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i,j] > th:
                i_min = max(i-nmsSize//2,0)
                i_max = min(i+nmsSize//2,R.shape[0]-1)
                j_min = max(j-nmsSize//2,0)
                j_max = min(j+nmsSize//2,R.shape[1]-1)
                neibor = R[i_min:i_max,j_min:j_max]
                if R[i,j] >= torch.max(neibor)-1e-5:
                    corner_list.append((i,j))
    return corner_list

#===============================================================
# ******HOG****** 
# ===============================================================

def hog(Ix:torch.Tensor,Iy:torch.Tensor,kplist:list):
    '''
    hog 特征描述
    
    parameter:
        :param Ix: x方向梯度,[h,w],torch.Tensor
        :param Iy: y方向梯度,[h,w],torch.Tensor
        :param kplist: list of [x,y] 需要hog的坐标list, list

    '''
    cell_size = (8, 8)  # 单元格大小
    num_bins = 9  # 直方图通道数
    gradient_directions = np.arctan2(Ix, Iy)  # 梯度方向
    gradient_magnitude = np.sqrt(Ix**2 + Iy**2)  # 梯度幅度
    num_cells_x = int(Ix.shape[1] / cell_size[1])
    num_cells_y = int(Ix.shape[0] / cell_size[0])
    histograms = np.zeros((num_cells_y, num_cells_x, num_bins))

    discriptorList = []
    goodkp = []
    for kp in kplist:
        y , x = int(kp[0]), int(kp[1])

        cell_magnitude = gradient_magnitude[y-cell_size[0]:y+cell_size[0], x-cell_size[1]:x+cell_size[1]]
        cell_direction = gradient_directions[y-cell_size[0]:y+cell_size[0], x-cell_size[1]:x+cell_size[1]]
        histogram = torch.zeros(num_bins)
        if cell_magnitude.shape[0] < cell_size[0]*2 or cell_magnitude.shape[1] < cell_size[1]*2 :
            continue
            
        for i in range(cell_size[0]):
            for j in range(cell_size[1]):
                direction = cell_direction[i, j]
                magnitude = cell_magnitude[i, j]
                bin_index = int(num_bins * direction / (2 * math.pi))
                histogram[bin_index] += magnitude
        discriptorList.append(histogram)
        goodkp.append(kp)
    return goodkp,discriptorList


#===============================================================
# ******steadyHarrisCornerForSIFT****** 
# ===============================================================

def steadyHarrisCornerForSIFT(Ix:torch.Tensor,Iy:torch.Tensor,num_levels = 6, thresh = 0.1,k = 0.1,size=5,sigma=1,nmsSize=11):
        
    corner_list_pyramid = []
    current_shape = Ix.shape

    for i in range(num_levels):

        corner_list = HarrisCorner(Ix, Iy, thresh = thresh, k = k, size=size, sigma=sigma, nmsSize=nmsSize)

        corner_list = [(kp[0],kp[1]) for kp in corner_list]

        corner_list_pyramid.append(corner_list)
        
        if current_shape[0] > 1 or current_shape[1] > 1:
            # Use the 'valid' option to prevent zero-padding
            Ix = F.interpolate(Ix.unsqueeze(0).unsqueeze(0), 
                               size=(current_shape[1] // 2, current_shape[0] // 2), mode='bilinear').squeeze(0).squeeze(0)
            Iy = F.interpolate(Iy.unsqueeze(0).unsqueeze(0), 
                               size=(current_shape[1] // 2, current_shape[0] // 2), mode='bilinear').squeeze(0).squeeze(0)
            current_shape = Ix.shape

    final_corner_list = []
    for i in range(num_levels):
        current_corner_list = corner_list_pyramid[i]
        neibor_list = []
        if i > 0:
            neibor_list += corner_list_pyramid[i-1]
            for kp in current_corner_list:
                for kp_match in neibor_list:
                    if (kp[0]-kp_match[0])**2 + (kp[1]-kp_match[1])**2 < 4:
                        final_corner_list.append((kp[1],kp[0],i+1))
        if i < num_levels-1:
            neibor_list += corner_list_pyramid[i+1]
            for kp in current_corner_list:
                for kp_match in neibor_list:
                    if (kp[0]-kp_match[0])**2 + (kp[1]-kp_match[1])**2 < 4:
                        final_corner_list.append((kp[1],kp[0],i+1))
       
    print('[tianmoucv.steadyHarrisCornerForSIFT]Steady KP for SIFT:',len(final_corner_list))

    return final_corner_list


#===============================================================
# ******SIFT****** 
# ===============================================================


def compute_multiscale_sift_descriptor(Ix:torch.Tensor, Iy:torch.Tensor, kp_list: list, num_levels = 6):
    """
    计算 Harris 角点的 SIFT 特征描述子
    
    Args:
        Ix (Tensor): x 方向的梯度张量
        Iy (Tensor): y 方向的梯度张量
        kp_list (list): 每个元素为 (x, y, level) 的角点列表
        
    Returns:
        descriptors: 包含每个角点 SIFT 描述子的列表，形状为 [num_kp, 128]
    """
    # 定义参数
    bin_size = 4      # 每个方向直方图的小块大小
    hist_bins = 8     # 方向直方图的 bins 数量
    descriptor_size = 16*8  # 最终描述子长度
    
    # 将角点列表转换为张量
    kps = torch.tensor(kp_list, dtype=torch.float32)

    grad_angle_list = []
    grad_magnitude_list = []

    current_shape = Ix.shape

    Ix_ = Ix.clone()
    Iy_ = Iy.clone()
    
    for i in range(num_levels):
        # 计算梯度方向和幅度
        grad_angle = torch.atan2(Iy, Ix)  # 归一化到 [-pi, pi]
        grad_magnitude = torch.sqrt(Ix**2 + Iy**2)
        
        grad_angle_list.append(grad_angle)
        grad_magnitude_list.append(grad_magnitude)

        if current_shape[0] > 1 or current_shape[1] > 1:
            # Use the 'valid' option to prevent zero-padding
            Ix_= F.interpolate(Ix_.unsqueeze(0).unsqueeze(0), 
                               size=(current_shape[1] // 2, current_shape[0] // 2), mode='bilinear').squeeze(0).squeeze(0)
            Iy_ = F.interpolate(Iy_.unsqueeze(0).unsqueeze(0), 
                               size=(current_shape[1] // 2, current_shape[0] // 2), mode='bilinear').squeeze(0).squeeze(0)
            current_shape = Ix.shape
    
    descriptors = []
    kplist = []

    for i, (x, y, level) in enumerate(kp_list):

        # 提取周围区域的梯度信息
        grad_angle = grad_angle_list[level-1]
        grad_magnitude = grad_magnitude_list[level-1]
        
        region_angle = grad_angle[y - bin_size//2 : y + bin_size//2,
                                  x - bin_size//2 : x + bin_size//2]
        region_magnitude = grad_magnitude[y - bin_size//2 : y + bin_size//2,
                                          x - bin_size//2 : x + bin_size//2]
        # 转换到整数坐标
        x = int(round(x))
        y = int(round(y))
        
        # 检查边界条件
        if x < bin_size or y < bin_size or \
           x + bin_size > grad_angle.size(1) or y + bin_size > grad_angle.size(0):
            continue  # 跳过边缘点
            
        # 创建方向直方图
        hist = torch.zeros(hist_bins, device=Ix.device)
        cell_size = bin_size // 2
        
        for dy in range(cell_size):
            for dx in range(cell_size):
                # 计算每个小块的梯度方向和幅度
                angles = region_angle[dy*cell_size : (dy+1)*cell_size,
                                     dx*cell_size : (dx+1)*cell_size]
                magnitudes = region_magnitude[dy*cell_size : (dy+1)*cell_size,
                                            dx*cell_size : (dx+1)*cell_size]
                
                # 将角度归一化为 0-360 度
                angles = torch.fmod(angles + math.pi, 2*math.pi) * (180/math.pi)
                
                # 统计每个方向的幅度贡献
                for a, m in zip(angles.flatten(), magnitudes.flatten()):
                    bin_idx = int((a % 360) // (360/hist_bins))
                    hist[bin_idx] += m
                    
        # 归一化直方图
        if torch.sum(hist) > 1e-8:
            hist /= torch.sum(hist)
            
        # 将直方图扩展为 128 维描述子
        descriptor = []
        for i in range(4):
            for j in range(4):
                start_idx = (i*hist_bins//4 + j) % hist_bins
                end_idx = (start_idx + hist_bins//4) % hist_bins
                if start_idx < end_idx:
                    descriptor += hist[start_idx:end_idx].tolist()
                else:
                    descriptor += torch.cat([hist[start_idx:], hist[:end_idx]]).tolist()
                    
        # 归一化描述子
        descriptor = torch.tensor(descriptor)
        descriptor /= torch.norm(descriptor) + 1e-8
        if torch.max(descriptor) > 0.2:
            descriptor *= (0.2 / torch.max(descriptor))
            
        descriptors.append(descriptor)
        kplist.append([x*2**(level-1),y*2**(level-1)])
    
    return descriptors,kplist

