import cv2
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import signal
from PIL import Image

import torch
import torch.nn.functional as F
import torch.nn as nn


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
def HarrisCorner(Ix:torch.Tensor,Iy:torch.Tensor,k = 0.1,th = 0.5,size=5,sigma=1,nmsSize=11):
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

    # 1. get difference image
    Ix[Ix<torch.max(Ix)*0.02]=0
    Iy[Iy<torch.max(Iy)*0.02]=0
    # 2. sober filtering
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
   # 3. windowed
    kernel = gaussain_kernel(size,sigma)
    Ix2 = gaussian_smooth(Ix2.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    Iy2 = gaussian_smooth(Iy2.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    Ixy = gaussian_smooth(Ixy.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    # 4. corner detect
    out = np.zeros(Ix2.shape)
    R = (Ix2 * Iy2 - Ixy ** 2) - k * ((Ix2 + Iy2) ** 2)
    threshold =  float(torch.max(R)) * th
    R_Max = F.max_pool2d(R.unsqueeze(0).unsqueeze(0), kernel_size=nmsSize, 
                             stride=1, padding=nmsSize//2).squeeze(0).squeeze(0)
    idmap = (R >= threshold).int() * (R > R_Max*0.999).int()
    R = R[idmap>0]
        
    return idmap,R


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
    R_Max = F.max_pool2d(R.unsqueeze(0).unsqueeze(0), kernel_size=nmsSize, 
                             stride=1, padding=nmsSize//2).squeeze(0).squeeze(0)
    idmap = (R >= threshold).int() * (R > R_Max*0.999).int()
    return idmap,R


# ===============================================================
# ******in testing******  Harris3D角点，用Ix和Iy做计算，可以用SD的两个方向
# ===============================================================
def HarrisCorner3(Ix:torch.Tensor,Iy:torch.Tensor,It:torch.Tensor,k = 0.5,th = 0.95,size=5,sigma=2):
    '''
    Harris3D角点，用Ix和Iy做计算，可以用SD的两个方向
    
    如果小正方体沿z方向移动，那小正方体里的点云数量应该不变
    如果小正方体位于边缘上，则沿边缘移动，点云数量几乎不变，沿垂直边缘方向移动，点云数量改
    如果小正方体位于角点上，则有两个方向都会大幅改变点云数量
    拓展到3D中则使用法向量(包含法线和方向两个信息)
    
    .. math::    A = Ix * Ix
    .. math::    B = Iy * Iy
    .. math::    C = It * It
    .. math::    D = Ix * Iy
    .. math::    E = Ix * It
    .. math::    F = Iy * It
    .. math:: M= [[A F E];[F B D];[E D C]]
    
    Harris 角点检测
    similar,
    
    .. math:: R=det(M)−ktrace(M)^2

    '''
    # 1. get difference image
    Ix[Ix<torch.max(Ix)*0.1]=0
    Iy[Iy<torch.max(Iy)*0.1]=0
    Iy[It<torch.max(It)*0.1]=0
    grad_norm = (Ix**2 + Iy**2+ It**2)**0.5 + 1e-9
    Ix = Ix / torch.max(grad_norm)
    Iy = Iy / torch.max(grad_norm)
    It = It / torch.max(grad_norm)
    
    # 3. sober filtering
    A = Ix * Ix
    B = Iy * Iy
    C = It * It
    D = Ix * Iy
    E = Ix * It
    F = Iy * It
    
    size = int(size)
    if size % 2 == 0:
        size = size + 1
    m = (size - 1) / 2
    y, x = torch.meshgrid(torch.arange(-m, m + 1), torch.arange(-m, m + 1))
    kernel = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    A = gaussian_smooth(A.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    B = gaussian_smooth(B.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    C = gaussian_smooth(C.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    D = gaussian_smooth(D.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    E = gaussian_smooth(E.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    F = gaussian_smooth(F.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    
    # 4. corner detect
    detM = A*B*C+2*D*E*F
    traceM = A+B+C-A*D*D-B*E*E-C*F*F
    R = detM - k * traceM ** 2
    threshold =  float(torch.max(R)) * th
    idmap = R >= threshold
    return idmap




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
# ******简化版SIFT中的描述子，缺少多尺度****** 
# ===============================================================
def sift(Ix:torch.Tensor,Iy:torch.Tensor, keypoints:list):
    '''
    **简化版SIFT中的描述子，缺少多尺度**
    
    parameter:
        :param Ix: x方向梯度,[h,w],torch.Tensor
        :param Iy: y方向梯度,[h,w],torch.Tensor
        :param keypoints: list of [x,y] 需要sift的坐标list, list

    '''
    Ix = Ix.numpy().astype(np.float32)
    Iy = Iy.numpy().astype(np.float32)
    # 定义圆形区域的半径
    radius = 13
    descriptors = []
    count = 0
    # 获取关键点坐标
    goofkp = []
    for kp in keypoints:
        descriptorlist = []
        y, x = int(kp[0]), int(kp[1])
        Xneighbor = Ix[y-1:y+2,x-1:x+2]
        Yneighbor = Iy[y-1:y+2,x-1:x+2]
        mask = (Xneighbor!=0) & (Yneighbor!=0)
        magnitude, majorAngle = cv2.cartToPolar(Xneighbor[mask],Yneighbor[mask], angleInDegrees=True)
        if majorAngle is None:
            continue
        majorAngle = np.mean(majorAngle)
        magnitude = np.mean(magnitude)
        
        # 选取圆形区域
        pIx = Ix[y - radius : y + radius, x - radius : x + radius]
        pIy = Iy[y - radius : y + radius, x - radius : x + radius]
        shapeofIxy = pIx.shape
        if shapeofIxy[0] < radius*2 or shapeofIxy[1] < radius*2 :
            continue
        # 将圆形区域分成16个子块
        step = int(radius/2)
        for i in range(4):
            for j in range(4):
                # 计算子块内像素的梯度方向和梯度强度
                dx = pIx[i * step : (i + 1) * step, j * step : (j + 1) * step]
                dy = pIy[i * step : (i + 1) * step, j * step : (j + 1) * step]
                magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
                hist, _ = np.histogram(angle-majorAngle, bins=4, range=(0, 360), weights=magnitude)
                descriptorlist.append(torch.Tensor(hist))
        if(len(descriptorlist)>0):
            descriptors.append(torch.stack(descriptorlist,dim=0))
            count += 1
            goofkp.append(kp)

    return goofkp,descriptors