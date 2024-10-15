import torch
import numpy as np
import cv2,sys
import torch
import torch.nn.functional as F
import torch.nn as nn

class denoise_defualt_args:

    def __init__(self,
                 thr_1 = 1,
                 thr_2 = 3, 
                 thr_3 = 3,
                 aop_gain=1,
                 self_calibration=False):
        '''
        self.aop_dark_dict = {'TD':[torch.zeros(160, 160) for _ in range(2)],
                              'SDL':[torch.zeros(160, 160) for _ in range(2*N)],
                              'SDR':[torch.zeros(160, 160) for _ in range(2*N)]}
        '''
        self.aop_dark_dict = {'TD':None,
                              'SDL':None,
                              'SDR':None}
        self.thr_1 = thr_1 * aop_gain
        self.thr_2 = thr_2 * aop_gain
        self.thr_3 = thr_3 * aop_gain
        self.gain = aop_gain
        self.self_calibration = self_calibration
        if self_calibration:
            print('[denoise_defualt_args]data reader will try to calibrate fpn using its own aop data, set to [True] for dark noise data')
        
    def print_info(self):
        print('------denoise_defualt_args----')
        if self.aop_dark_dict['TD'] is not None:
            print('aop_dark_dict,dict,dark noise for td,sdl,sdr (key,length,shape):',[(key,len(self.aop_dark_dict[key]),self.aop_dark_dict[key][0].shape) for key in self.aop_dark_dict] )
        else:
            print('denoise_args didnot provide dark noise, data reader will try to calibrate fpn using bright aop data(self_calibration=True), or use ZEROS(self_calibration=False)')
        print('thr_1,float,template threshold:',self.thr_1)
        print('thr_2,float,Threshold for 3x3 avg pool:',self.thr_2)
        print('thr_3,float,Threshold for hot pixel:',self.thr_3)
        print('------denoise_defualt_args----')

def custom_round(x):
#正信号向下取整，负信号向上取整
    return torch.where(x >= 0, torch.floor(x), torch.ceil(x))

def conv_and_threshold(input_tensor, kernel_size, threshold):
    # 粗滤波
    input_tensor_1 = input_tensor.unsqueeze(0).unsqueeze(0)  # 形状变为 (1, 1, h, w)
    input_tensor = torch.abs(input_tensor_1)

    kernel_size = (kernel_size//2)*2 + 1
    conv_kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32)

    # 定义卷积层
    conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernel_size, kernel_size), stride=1,
                           padding=kernel_size // 2, bias=False)

    # 设置卷积核
    with torch.no_grad():
        conv_layer.weight = nn.Parameter(conv_kernel.unsqueeze(0).unsqueeze(0))  # 形状变为 (1, 1, kH, kW)

    # 应用卷积层
    conv_output = conv_layer(input_tensor)

    # 设置阈值并应用
    mask = conv_output > threshold  # 创建一个布尔掩码
    result_tensor = torch.where(mask, input_tensor_1, torch.tensor(0.0))  # 根据掩码保留原始张量中的值或置0

    # 将结果张量从四维转换回二维
    result_tensor = result_tensor.squeeze(0).squeeze(0)  # 形状变回 (h, w)

    return result_tensor

def conv_and_threshold_1(input_tensor, kernel_size, threshold):
    # 细滤波

    input_tensor_1 = input_tensor.unsqueeze(0).unsqueeze(0)  # 形状变为 (1, 1, h, w)
    input_tensor = torch.abs(input_tensor_1)
    # 创建一个中间元素为0，其余为1的卷积核

    kernel_size = (kernel_size//2)*2 + 1
    conv_kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32)
    conv_kernel[kernel_size // 2, kernel_size // 2] = 0  # 设置中间元素为0

    # 定义卷积层
    conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernel_size, kernel_size), stride=1,
                           padding=kernel_size // 2, bias=False)

    # 设置卷积核
    with torch.no_grad():
        conv_layer.weight = nn.Parameter(conv_kernel.unsqueeze(0).unsqueeze(0))  # 形状变为 (1, 1, kH, kW)

    # 应用卷积层
    conv_output = conv_layer(input_tensor)

    # 对卷积结果取绝对值并进行求和
    mask = conv_output > threshold
    # 对每个像素值与其对应的绝对值之和进行比较
    result_tensor = torch.where(mask, input_tensor_1, torch.tensor(0.0))

    # 将结果张量从四维转换回二维
    result_tensor = result_tensor.squeeze(0).squeeze(0)  # 形状变回 (h, w)

    return result_tensor



