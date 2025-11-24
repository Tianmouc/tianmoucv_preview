import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate as tv_rotate
import math
import numpy as np
from scipy.spatial import ConvexHull
from typing import Union

# ===============================================================
# SD坐标变换
# ===============================================================

        
def fourdirection2xy(sd: Union[np.array,torch.tensor]) -> Union[np.array,torch.tensor]:
    print('fourdirection2xy is decrepted, please use SD2XY')
    return SD2XY(sd)
    
def SD2XY(sd_raw:torch.tensor) -> torch.tensor:
    '''
    input: [h,w,2]/[2,h,w]/[n,2,h,w]
    output: [h,2*w],[h,2*w] or [n,h,2*w],[n,h,2*w]
    坐标变换规则参照http://www.tianmouc.cn:40000/tianmoucv/introduction.html
    y 正方向是↓ x正方形是→
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

    sdx = (sdul + sdll - sdur - sdlr)/4
    sdy = (sdur - sdlr + sdul - sdll)/4

    if len(sd_raw.shape) == 3:
        return sdx.squeeze(0).squeeze(0), sdy.squeeze(0).squeeze(0)
    else:
        return sdx.squeeze(1), sdy.squeeze(1)

# ===============================================================
# SD上采样填
# ===============================================================

def upsampleTSD_conv(tsdiff):
    td = tsdiff[0:1,...]
    sd = tsdiff[1:,...]
    td_extend = upsample_cross_conv(td)
    sd_extend = upsample_horizental_conv(sd)
    return torch.cat([td,sd],dim=0)

def upsample_cross_conv(tensor):
    '''
    adjust the data space and upsampling, please refer to tianmoucv doc for detail
    '''
    # 获取输入Tensor的维度信息
    h,w = tensor.shape[-2:]
    tensor_expand = torch.zeros([*tensor.shape[:-2],h,w*2])
    tensor_expand[...,::2,::2] = tensor[...,::2,:]
    tensor_expand[...,1::2,1::2] = tensor[...,1::2,:]
    channels, T, height, width = tensor_expand.size()
    input_tensor = tensor_expand.view(channels*T, height, width).unsqueeze(1)
    # 定义卷积核
    kernel = torch.zeros(1, 1, 3, 3)
    kernel[:, :, 1, 0] = 1/4
    kernel[:, :, 1, 2] = 1/4
    kernel[:, :, 0, 1] = 1/4
    kernel[:, :, 2, 1] = 1/4
    # 对输入Tensor进行反射padding
    padded_tensor = F.pad(input_tensor, (1, 1, 1, 1), mode='reflect')
    # 将原tensor复制一份用于填充结果
    output_tensor = input_tensor.clone()
    # 将卷积结果填充回原tensor
    for c in range(channels*T):
        output = F.conv2d(padded_tensor[c:c+1,:,...], kernel, padding=0)
        output_tensor[c:c+1,: , 0:-1:2, 1:-1:2] = output[:, :, 0:-1:2, 1:-1:2]
        output_tensor[c:c+1,: , 1:-1:2, 0:-1:2] = output[:, :, 1:-1:2, 0:-1:2]
    return output_tensor[:,0,...].view(channels, T, height, width)
    
def upsample_horizental_conv(tensor):
    '''
    adjust the data space and upsampling, please refer to tianmoucv doc for detail
    '''
    h,w = tensor.shape[-2:]
    tensor_expand = torch.zeros([*tensor.shape[:-2],h,w*2])
    tensor_expand[...,::2,::2] = tensor[...,::2,:]
    tensor_expand[...,1::2,1::2] = tensor[...,1::2,:]
    channels, T, height, width = tensor_expand.size()
    input_tensor = tensor_expand.view(channels*T, height, width).unsqueeze(1)
    # 定义卷积核
    kernel = torch.zeros(1, 1, 3, 3)
    kernel[:, :, 1, 0] = 1/2
    kernel[:, :, 1, 2] = 1/2
    # 对输入Tensor进行反射padding
    padded_tensor = F.pad(input_tensor, (1, 1, 1, 1), mode='reflect')
    # 将原tensor复制一份用于填充结果
    output_tensor = input_tensor.clone()
    # 将卷积结果填充回原tensor
    for c in range(channels*T):
        output = F.conv2d(padded_tensor[c:c+1,:,...], kernel, padding=0)
        output_tensor[c:c+1,: , 0:-1:2, 1:-1:2] = output[:, :, 0:-1:2, 1:-1:2]
        output_tensor[c:c+1,: , 1:-1:2, 0:-1:2] = output[:, :, 1:-1:2, 0:-1:2]
    return output_tensor[:,0,...].view(channels, T, height, width)


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
# SD和tensor的旋转操作
# ===============================================================


def rotate_nd_tensor(tensor, angle):
    """
    旋转 [B, C, H, W] 张量的每个图像，并调整大小以刚好包络旋转后的内容。
    Args:
        tensor: 输入张量，形状 [B, C, H, W]
        angle: 旋转角度（度）

    Returns:
        旋转后的张量，形状 [B, C, H', W']，其中 H' 和 W' 是新尺寸
    """
    original_dim = tensor.dim()
    while tensor.dim() < 4:
        tensor = tensor[None]  # 在开头插入新维度
    B, C, H, W = tensor.shape
    angle_rad = math.radians(angle)
    # 对每个图像进行旋转
    rotated_tensor = tv_rotate(tensor, angle, expand=True)
    _, _, rot_H, rot_W = rotated_tensor.shape
    while rotated_tensor.dim() > original_dim:
        rotated_tensor = rotated_tensor[0,...] 
    return rotated_tensor

def rotate_xyd(SD, angle):
    """
    可以单独用于处理SD
    旋转 [B,C,H,W]
    """
    original_dim = SD.dim()
    assert (SD.shape[0] == 2 and SD.dim()==3) or (SD.shape[1] == 2 and SD.dim()==4)
    while SD.dim() < 4:
        SD = SD[None]  # 在开头插入新维度

    rotated_SD = rotate_nd_tensor(SD,angle)
    SDx = rotated_SD[:,0,...]
    SDy = rotated_SD[:,1,...]
    angle_rad = torch.deg2rad(torch.tensor(angle, device=SD.device))
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    x_rotated = SDx * cos_a + SDy * sin_a
    y_rotated = - SDx * sin_a + SDy * cos_a

    rotated_SD_corrected = torch.stack([x_rotated,y_rotated],dim=1)

    while rotated_SD_corrected.dim() > original_dim:
        rotated_SD_corrected = rotated_SD_corrected[0,...] 
    
    return rotated_SD_corrected


def rotate_txyd(txydiff, angle):
    """
    用于处理TD+XY的数据，可以直接作用于data reader给出的结果
    旋转 [B/T,C,H,W] angle是角度
    """
    assert (txydiff.shape[0] == 3 and txydiff.dim()==3) or (txydiff.shape[1] == 3 and txydiff.dim()==4)
    
    xydiff = rotate_xyd(torch.FloatTensor(txydiff[:,1:,...]),angle)
    td = rotate_nd_tensor(torch.FloatTensor(txydiff[:,0:1,...]),angle)
    rotate_txydiff = torch.cat([td,xydiff],dim=1)

    return rotate_txydiff


# ===============================================================
# SD和tensor的镜像操作
# ===============================================================

def flip_xyd(xydiff, dims=[2]):
    """
    翻转 [B,C,H,W] 的SD
    注：翻转TD 不需要额外处理
    """

    original_dim = xydiff.dim()
    assert (xydiff.shape[0] == 2 and xydiff.dim()==3) or (xydiff.shape[1] == 2 and xydiff.dim()==4)
    assert all(element == 2 or element == 3 for element in dims)
    
    while xydiff.dim() < 4:
        xydiff = xydiff[None]  # 在开头插入新维度

    flip_xyd = torch.flip(xydiff, dims)
    if 2 in dims:
        flip_xyd[:,0,...] = - flip_xyd[:,0,...]
    if 3 in dims:
        flip_xyd[:,1,...] = - flip_xyd[:,1,...]

    while flip_xyd.dim() > original_dim:
        flip_xyd = flip_xyd[0,...] 
    return flip_xyd



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


