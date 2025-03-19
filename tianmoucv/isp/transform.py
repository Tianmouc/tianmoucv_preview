import torch
import torch.nn.functional as F

import numpy as np
from scipy.spatial import ConvexHull
from typing import Union

# ===============================================================
# SD坐标变换
# ===============================================================
def SD2XY(sd_raw:torch.tensor) -> torch.tensor:
    '''
    input: [h,w,2]/[2,h,w]/[n,2,h,w]
    output: [h,2*w],[h,2*w] or [n,h,2*w],[n,h,2*w]
    坐标变换规则参照http://www.tianmouc.cn:40000/tianmoucv/introduction.html
    '''
    if len(sd_raw.shape) == 3:
        assert (sd_raw.shape[2]==2 or sd_raw.shape[0]==2)
        if sd_raw.shape[2] == 2:
            sd = sd_raw.permute(2,0,1).unsqueeze(0) #[h,w,c]->[1,c,h,w]
        else:
            sd = sd_raw.unsqueeze(0)
    else:
        assert (len(sd_raw.shape) == 4 and sd_raw.shape[1]==2)
        sd = sd_raw
        
    b,c,h,w = sd.shape
    sdul = sd[:,0:1,0::2,...]
    sdll = sd[:,0:1,1::2,...]
    sdur = sd[:,1:2,0::2,...]
    sdlr = sd[:,1:2,1::2,...]

    target_size = (h,w*2)
    sdul = F.interpolate(sdul, size=target_size, mode='bilinear', align_corners=False)
    sdll = F.interpolate(sdll, size=target_size, mode='bilinear', align_corners=False)
    sdur = F.interpolate(sdur, size=target_size, mode='bilinear', align_corners=False)
    sdlr = F.interpolate(sdlr, size=target_size, mode='bilinear', align_corners=False)

    sdx = ((sdul + sdll)/1.414 - (sdur + sdlr)/1.414)/2
    sdy = ((sdur - sdlr)/1.414 + (sdul - sdll)/1.414)/2

    if len(sd_raw.shape) == 3:
        return sdx.squeeze(0).squeeze(0), sdy.squeeze(0).squeeze(0)
    else:
        return sdx.squeeze(1), sdy.squeeze(1)


# ===============================================================
# SD坐标变换
# ===============================================================
def fourdirection2xy(sd: Union[np.array,torch.tensor]) -> Union[np.array,torch.tensor]:
    print('fourdirection2xy is decrepted, please use SD2XY')
    return SD2XY(sd)

# ===============================================================
# 自卷积
# ===============================================================
def selfConv(img,clv_w = 11):
    from scipy import ndimage
    cov_len = clv_w # 尽量选取比较大的卷积核
    avg_img = []
    for i in range(11): #循环次数不限，尽量不要太多
        px = np.random.randint(0, 256 - cov_len)
        py = np.random.randint(0, 256 - cov_len)
        cut_img = img[py:py+cov_len,px:px+cov_len]
        avg_img.append(np.array(cut_img).reshape(-1))
        avg_img = np.array(avg_img).mean(axis=0).reshape(cov_len, cov_len)
        avg_img = avg_img / avg_img.sum() # 加权平均
    avg_img = np.array(avg_img).mean(axis=0).reshape(cov_len, cov_len)
    avg_img = avg_img / avg_img.sum() # 加权平均
    return avg_img


# ===============================================================
# 可视化差分数据
# ===============================================================
def upsampleTSD(tsdiff):
    h,w = tsdiff.shape[-2:]
    w *= 2
    tsdiff_expand = torch.zeros([*tsdiff.shape[:-2],h,w])
    tsdiff_expand[...,::2,::2] = tsdiff[...,::2,:]
    tsdiff_expand[...,1::2,1::2] = tsdiff[...,1::2,:]
    for i in range(1,h,2):
        for j in range(0,w,2):
            sum_ele = 0
            count_ele = 0
            sum_ele += tsdiff_expand[...,i-1,j]
            count_ele += 1
            if j>1:
                sum_ele += tsdiff_expand[...,i,j-1]
                count_ele += 1
            if i<h-1:
                sum_ele += tsdiff_expand[...,i+1,j]
                count_ele += 1
            if j<w-1:
                sum_ele += tsdiff_expand[...,i,j+1]
                count_ele += 1      
            tsdiff_expand[...,i,j] = sum_ele/count_ele

    for i in range(0,h,2):
        for j in range(1,w,2):
            sum_ele = 0
            count_ele = 0
            if i>1:
                sum_ele += tsdiff_expand[...,i-1,j]
                count_ele += 1
            if j>1:
                sum_ele += tsdiff_expand[...,i,j-1]
                count_ele += 1
            if i<h-1:
                sum_ele += tsdiff_expand[...,i+1,j]
                count_ele += 1
            if j<w-1:
                sum_ele += tsdiff_expand[...,i,j+1]
                count_ele += 1      
            tsdiff_expand[...,i,j] = sum_ele/count_ele
    return tsdiff_expand


def compute_minimum_convex_hull(matrix):
    '''
    计算最小凸包
    '''
    # 获取非零元素的坐标
    nonzero_indices = np.transpose(np.nonzero(matrix))

    # 计算非凸闭包
    hull = ConvexHull(nonzero_indices)

    # 获取闭包的顶点
    vertices = nonzero_indices[hull.vertices]

    return vertices


def calculate_area(vertices):
    '''
    # 计算多边形的面积
    '''
    area = 0.5 * abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) -
                     np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))

    return area


def is_inside(vertices, point):
    '''
    # 判断点是否在多边形内部
    '''
    path = mpath.Path(vertices)
    inside = path.contains_point(point)

    return inside


def interpolate_zero_point(matrix, vertices, zero_point):
    '''
    # 找到零点周围的非零点，进行插值
    '''
    neighbors = []
    for i in range(zero_point[0] - 1, zero_point[0] + 2):
        for j in range(zero_point[1] - 1, zero_point[1] + 2):
            if (i, j) in vertices:
                neighbors.append((i, j))

    # 计算插值
    interpolated_value = np.mean([matrix[n[0], n[1]] for n in neighbors])

    return interpolated_value


