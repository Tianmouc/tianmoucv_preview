import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy import signal
from PIL import Image
from tianmoucv import *
import cv2
import sys
import torch.nn.functional as F
import torch
from scipy.optimize import linear_sum_assignment


# ===============================================================
# 一些基本工具
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


#=============================================================================================================================
# 检测器==============================================================================================================================

# ===============================================================
# Harris角点，用Ix和Iy做计算，可以用SD的两个方向
# ===============================================================
def HarrisCorner(Ix,Iy,k = 0.1,th = 0.5,size=5,sigma=1,nmsSize=11):
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
    idmap = (R >= threshold).int() * (R > R_Max-1e-5).int()
    R = R[idmap>0]
        
    return idmap,R
 
# ===============================================================
# Shi-Tomasi角点，用Ix和Iy做计算，可以用SD的两个方向
# ===============================================================

def TomasiCorner(Ix, Iy, index=1000,size=5,sigma=2,nmsSize=11):
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
    idmap = (R >= threshold).int() * (R > R_Max-1e-5).int()
    return idmap,R


# ===============================================================
# ******in testing******  Harris3D角点，用Ix和Iy做计算，可以用SD的两个方向
# ===============================================================
def HarrisCorner3(Ix,Iy,It,k = 0.5,th = 0.95,size=5,sigma=2):
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



#=============================================================================================================================
# 描述子==============================================================================================================================


#===============================================================
# ******HOG****** 
# ===============================================================

def hog(Ix,Iy,kplist):
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
def sift(Ix,Iy, keypoints):
    '''
    **简化版SIFT中的描述子，缺少多尺度**
    
    parameter:
        :param Ix: x方向梯度,[h,w],torch.Tensor
        :param Iy: y方向梯度,[h,w],torch.Tensor
        :param keypoints: list of [x,y] 需要sift的坐标list, list

    '''
    Ix = Ix.numpy().astype(np.float)
    Iy = Iy.numpy().astype(np.float)
    # 定义圆形区域的半径
    radius = 13
    descriptors = []
    count = 0
    # 获取关键点坐标
    goofkp = []
    for kp in keypoints:
        descriptorlist = []
        y, x = int(kp[0]), int(kp[1])
        
        #0831晚上修改：可能要取平均一下，噪声影响大，主方向比较重要
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

    #print(descriptors)
    return goofkp,descriptors

# ===============================================================
# ******描述子匹配****** 
# ===============================================================
#ratio=0.85:knn中前两个匹配的距离的比例
def feature_matching(des1, des2, ratio=0.85):
    """
    Match SIFT descriptors between two images.
    
    parameter:
        :param des1: kp list1,[x,y],list
        :param des2: kp list2,[x,y],list
        :param ratio: knn中前两个匹配的距离的比例,筛选匹配得足够好的点, float

    """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1.cpu().numpy(), des2.cpu().numpy(), k=2)
    good_matches = []
    if len(matches) > 0:
        for m,n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append([m])
    return good_matches



def mini_l2_cost_matching(des1, des2, num=20):
    """
    Match SIFT descriptors between two images.
    
    parameter:
        :param des1: kp list1,[x,y],list
        :param des2: kp list2,[x,y],list
        :param num: 筛选匹配得足够好的点, int

    """
    distance_matrix = np.zeros((len(des1), len(des2)))
    for i, ref_feature in enumerate(des1):
        for j, query_feature in enumerate(des2):
            distance_matrix[i, j] = np.linalg.norm(ref_feature - query_feature)
    matched_indices = linear_sum_assignment(distance_matrix)
    matches = []
    for i in range(len(matched_indices[0])):
        matches.append((matched_indices[0][i],matched_indices[1][i]))
    matches = matches[:num]
    return matches

# ===============================================================
# ******刚性对齐****** 
# ===============================================================
def align_images(image, kpList1, kpList2,matches, canvas=None):
    """
    单应性矩阵，刚性对齐
    
    parameter:
        :param image: np图像,np.array
        :param kpList1: kp list [x,y],list
        :param kpList2: kp list [x,y],list
        :param matches: 匹配点列表 [id1,id2],list

    """
    H = None
    src_pts = []
    dst_pts = []
    if(len(matches)>4):
        if isinstance(matches[0],tuple):
            src_pts = np.float32([kpList1[m[0]] for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpList2[m[1]] for m in matches]).reshape(-1, 1, 2)
        else:
            src_pts = np.float32([kpList1[m[0].queryIdx] for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpList2[m[0].trainIdx] for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(H)
    for i in range(len(src_pts)):
        y1 , x1 = int(src_pts[i][0][0]),int(src_pts[i][0][1])
        y2 , x2 = int(dst_pts[i][0][0]),int(dst_pts[i][0][1])
        if canvas is not None:
            cv2.line(canvas,(x1,y1),(x2+640,y2),(255,0,0))
        print(x1,',',y1,'---',x2,',',y2)
    w,h = image.shape[1],image.shape[0]
    
    imagewp = image
    if H is not None:
        imagewp = cv2.warpPerspective(image,H, (w,h))
    return imagewp,H

