import torch
import numpy as np
import cv2,sys
import torch
import torch.nn.functional as F
import torch.nn as nn

class denoise_defualt_args:

    def __init__(self):
        self.aop_dark_dict = {'TD':[torch.zeros(160, 160) for _ in range(2)],
                              'SDL':[torch.zeros(160, 160) for _ in range(2)],
                              'SDR':[torch.zeros(160, 160) for _ in range(2)]}
        self.thr_1 = 2
        self.thr_2 = 6
        self.thr_3 = 6  
        self.gain = 1
        
    def print_info(self):
        print('------denoise_defualt_args----')
        print('aop_dark_dict,dict,dark noise for td,sdl,sdr (key,length,shape):',[(key,len(self.aop_dark_dict[key]),self.aop_dark_dict[key][0].shape) for key in self.aop_dark_dict] )
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


def dark_fpn(dataset_dark):
    # get dark
    TD_mean = [torch.zeros(160, 160) for _ in range(2)]
    SDL_mean = [torch.zeros(160, 160) for _ in range(2)]
    SDR_mean = [torch.zeros(160, 160) for _ in range(2)]
    TD_mean_odd = torch.zeros(160, 160)
    TD_mean_even = torch.zeros(160, 160)
    SDL_mean_odd = torch.zeros(160, 160)
    SDL_mean_even = torch.zeros(160, 160)
    SDR_mean_odd = torch.zeros(160, 160)
    SDR_mean_even = torch.zeros(160, 160)

    even_cnt = 0
    odd_cnt = 0
    # for index in tqdm(range(start, start + 1)):
    sample_dark = dataset_dark[1]
    N_rod = sample_dark['rawDiff'].shape[1]
    # One RGB (COP) have 25, 50 or more AOP(rod, diff) data, depends on the camera configuration
    for j in range(N_rod):
        rawDiff_dark = torch.Tensor(sample_dark['rawDiff'])
        SD_dark = rawDiff_dark[1:3, j , ...].clone()
        sd_l_dark = SD_dark[0,...]
        sd_r_dark = SD_dark[1,...]
        # TD record
        td_dark = rawDiff_dark[0, j, ...].clone()
        if j % 2 == 0:
            TD_mean_even += td_dark
            SDL_mean_even += sd_l_dark
            SDR_mean_even += sd_r_dark
            even_cnt += 1
        else:
            TD_mean_odd += td_dark
            SDL_mean_odd += sd_l_dark
            SDR_mean_odd += sd_r_dark
            odd_cnt += 1
    TD_mean[0] = custom_round(TD_mean_even / even_cnt)
    TD_mean[1] = custom_round(TD_mean_odd / odd_cnt)
    SDL_mean[0] = custom_round(SDL_mean_even / even_cnt)
    SDL_mean[1] = custom_round(SDL_mean_odd / odd_cnt)
    SDR_mean[0] = custom_round(SDR_mean_even / even_cnt)
    SDR_mean[1] = custom_round(SDR_mean_odd / odd_cnt)
    return TD_mean, SDL_mean, SDR_mean

