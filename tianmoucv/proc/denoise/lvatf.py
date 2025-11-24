import os
from PIL import Image
import numpy as np
import time
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.ndimage
from scipy.signal import wiener


def conv_and_threshold(input_tensor, kernel_size, threshold):
    # 将输入张量从二维转换为四维
    input_tensor_1 = input_tensor.unsqueeze(0).unsqueeze(0)  # 形状变为 (1, 1, h, w)
    input_tensor = torch.abs(input_tensor_1)
    # 创建一个全1的卷积核
    conv_kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32).to(input_tensor.device)
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


def conv_and_threshold_2(input_tensor, kernel_size, threshold):
    # 将输入张量从二维转换为四维

    input_tensor_1 = input_tensor.unsqueeze(0).unsqueeze(0)  # 形状变为 (1, 1, h, w)
    input_tensor = torch.abs(input_tensor_1)
    # 创建一个中间元素为0，其余为1的卷积核
    conv_kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32).to(input_tensor.device)
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


def local_var_test(input_tensor, kernel_size):
    if input_tensor.dim() == 2:
        tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 3:
        tensor = input_tensor.unsqueeze(0)

    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=tensor.device) / 9
    # kernel = kernel.to(tensor.device)

    local_mean = F.conv2d(tensor, kernel, padding=1)

    local_mean_square = F.conv2d(tensor ** 2, kernel, padding=1)

    local_variance = local_mean_square - local_mean ** 2

    # local_variance_np = local_variance.cpu().numpy()

    result_tensor = local_variance
    result_tensor = result_tensor.squeeze(0).squeeze(0)  # ÐÎ×´±ä»Ø (h, w)
    return result_tensor
def variance_to_threshold(variance, min_thr=3, max_thr=8):
    # 将方差线性映射到阈值范围 [3, 8]
    # variance_np = variance.detach().numpy()
    norm_variance = torch.clamp(variance, min=0, max=1)  # 将方差限制在0到1之间
    # norm_variance_np = norm_variance.detach().numpy()
    threshold_range = max_thr - min_thr
    dynamic_threshold = min_thr + (norm_variance) * threshold_range
    # dynamic_threshold_np = dynamic_threshold.detach().numpy()
    return dynamic_threshold


def sd_adaptive_filter(A, min_thr=3, max_thr=8, kernel_size=3):
    # 计算局部方差
    # t0 = time.time()
    local = local_var_test(A, kernel_size)
    # t1 = time.time()
    # 将方差映射到动态阈值
    dynamic_threshold = variance_to_threshold(local, min_thr, max_thr)
    # t2 = time.time()

    # dynamic_threshold_np = dynamic_threshold.numpy()
    TD_10 = (torch.abs(A) >= 1).float() * A
    TD_3 = conv_and_threshold(TD_10, kernel_size, dynamic_threshold)
    # t3 = time.time()

    TD_4 = conv_and_threshold_2(TD_3, kernel_size, 5)
    # t4 = time.time()
    # print(f"SD adaptive filter time {(t1 - t0)*1000:.4f} ms, {(t2-t1)*1000:.4f} ms, {(t3-t2)*1000:.4f} ms, {(t4-t3)*1000:.4f} ms")
    A_tensor = A
    label_array = torch.full(A_tensor.shape, -1, device=A_tensor.device)  # 初始化为 -1
    label_array[TD_4 != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array[A_tensor == 0] = 0  # A 中为 0 的区域设为 0
    return TD_4


def calculate_metrics(mode_1, mode_2):
    # 将输入的 numpy 数组转换为 torch 张量
    # mode_1 = torch.from_numpy(mode_1)

    # 计算 TP：mode_1 中为 1 且 mode_2 中也为 1 的元素个数
    TP = ((mode_1 == 1) & (mode_2 == 1)).sum().item()

    # 计算 TN：mode_1 中为 -1 且 mode_2 中也为 -1 的元素个数
    TN = ((mode_1 == -1) & (mode_2 == -1)).sum().item()

    # 计算 FP：mode_1 中为 -1 且 mode_2 中不为 -1 的元素个数
    FP = ((mode_1 == -1) & (mode_2 != -1)).sum().item()

    # 计算 FN：mode_1 中为 1 且 mode_2 中不为 1 的元素个数
    FN = ((mode_1 == 1) & (mode_2 != 1)).sum().item()

    return TP, TN, FP, FN


def local_var(input_tensor, kernel_size, th_mean, th_var):
    if input_tensor.dim() == 2:
        tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 3:
        tensor = input_tensor.unsqueeze(0)
    # ??3x3??????
    kernel = torch.ones(1, 1, kernel_size, kernel_size) / 9
    kernel = kernel.to(tensor.device)

    # ??????
    local_mean = F.conv2d(tensor, kernel, padding=1)

    # ?????????
    local_mean_square = F.conv2d(tensor ** 2, kernel, padding=1)

    # ??????
    local_variance = local_mean_square - local_mean ** 2
    # tensor_np = input_tensor.cpu().numpy()
    # local_mean_np = local_mean.cpu().numpy()
    # local_variance_np = local_variance.cpu().numpy()
    # 设置阈值并应用
    mask = (local_variance > th_var)  # 创建一个布尔掩码 (torch.abs(local_mean) > th_mean) &
    # mask_np = mask.cpu().numpy()
    result_tensor = torch.where(mask, tensor, torch.tensor(0.0))  # 根据掩码保留原始张量中的值或置0
    result_tensor = result_tensor.squeeze(0).squeeze(0)  # 形状变回 (h, w)
    return result_tensor


def variance_to_threshold_td(variance, min_thr=3, max_thr=8):
    # 将方差线性映射到阈值范围 [3, 8]
    # variance_np = variance.detach().numpy()
    norm_variance = torch.clamp(variance, min=0, max=2)  # 将方差限制在0到1之间
    # norm_variance_np = norm_variance.detach().numpy()
    threshold_range = max_thr - min_thr
    dynamic_threshold = min_thr + (norm_variance) * threshold_range / 2
    # dynamic_threshold_np = dynamic_threshold.detach().numpy()
    return dynamic_threshold


def td_adaptive_filter(A, min_thr=3, max_thr=8, kernel_size=3):
    # 计算局部方差
    # local = local_variance(A, kernel_size)
    local = local_var_test(A, kernel_size)

    # 将方差映射到动态阈值
    dynamic_threshold = variance_to_threshold_td(local, min_thr, max_thr)
    # dynamic_threshold_np = dynamic_threshold.numpy()
    TD_10 = (torch.abs(A) >= 1).float() * A
    TD_3 = conv_and_threshold(TD_10, kernel_size, dynamic_threshold)
    TD_4 = conv_and_threshold_2(TD_3, kernel_size, 5)
    A_tensor = A
    label_array = torch.full(A_tensor.shape, -1, device=A_tensor.device)   # 初始化为 -1
    label_array[TD_4 != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array[A_tensor == 0] = 0  # A 中为 0 的区域设为 0
    return TD_4

# batch version
def batch_local_var(input_batch, kernel_size):
    """
    input_batch: (B, H, W) »ò (B, C, H, W)
    Êä³ö: (B, C, H, W)
    """
    # Í³Ò»×ª»»Îª4DÕÅÁ¿
    if input_batch.dim() == 3:
        tensor = input_batch.unsqueeze(1)  # (B, 1, H, W)
    else:
        tensor = input_batch

    # Ê¹ÓÃ·ÖÀë¾í»ýºË±ÜÃâÖØ¸´´´½¨
    kernel = torch.ones(1, 1, kernel_size, kernel_size,
                       device=tensor.device, dtype=tensor.dtype) / (kernel_size**2)

    # ÅúÁ¿¾í»ý¼ÆËã
    local_mean = F.conv2d(tensor, kernel, padding=kernel_size//2, groups=tensor.size(1))
    local_mean_square = F.conv2d(tensor**2, kernel, padding=kernel_size//2, groups=tensor.size(1))
    return local_mean_square - local_mean**2

def batch_variance_to_threshold(variance_batch, min_thr, max_thr):
    """
    variance_batch: (B, C, H, W)
    Êä³ö: (B, C, H, W) ãÐÖµÕÅÁ¿
    """
    norm_variance = torch.clamp(variance_batch, min=0, max=1)
    return min_thr + norm_variance * (max_thr - min_thr)

def batch_conv_threshold(input_batch, kernel_size, threshold_batch):
    """
    input_batch: (B, C, H, W)
    threshold_batch: (B, C, H, W)
    """
    if input_batch.dim() == 3:
        inputtensor = input_batch.unsqueeze(1)  # (B, 1, H, W)
    else:
        inputtensor = input_batch

    B, C, H, W = inputtensor.shape
    abs_input = torch.abs(inputtensor)

    # Ô¤¶¨Òå¿É¸´ÓÃ¾í»ýºË
    kernel = torch.ones(1, 1, kernel_size, kernel_size,
                       device=inputtensor.device, dtype=inputtensor.dtype)

    # ·Ö×é¾í»ýÊµÏÖ²¢ÐÐ´¦Àí
    conv_output = F.conv2d(abs_input, kernel, padding=kernel_size//2, groups=C)

    # ¹ã²¥±È½ÏãÐÖµ
    mask = conv_output > threshold_batch
    return torch.where(mask, inputtensor, 0)

def batch_conv_threshold_2(input_batch, kernel_size, threshold):
    """
    input_batch: (B, C, H, W)
    threshold: scalar »ò (B, C) ¿É¹ã²¥ÕÅÁ¿
    """
    if input_batch.dim() == 3:
        inputtensor = input_batch.unsqueeze(1)  # (B, 1, H, W)
    else:
        inputtensor = input_batch

    B, C, H, W = inputtensor.shape
    abs_input = torch.abs(inputtensor)

    # ´´½¨ÖÐÐÄÎª0µÄ¾í»ýºË
    kernel = torch.ones(kernel_size, kernel_size,
                       device=inputtensor.device, dtype=inputtensor.dtype)
    kernel[kernel_size//2, kernel_size//2] = 0
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    # ·Ö×é¾í»ý
    conv_output = F.conv2d(abs_input, kernel, padding=kernel_size//2, groups=C)

    # Ö§³ÖãÐÖµ¹ã²¥
    if isinstance(threshold, (int, float)):
        threshold_tensor = torch.tensor(threshold, device=inputtensor.device)
    else:
        threshold_tensor = threshold.view(B, C, 1, 1)

    mask = conv_output > threshold_tensor
    return torch.where(mask, inputtensor, 0)

# ÐÞ¸ÄºóµÄÅúÁ¿´¦Àíº¯Êý
def batch_sd_adaptive_filter(A_batch, min_thr=3, max_thr=8, kernel_size=3):
    # ÅúÁ¿¼ÆËã¾Ö²¿·½²î
    local = batch_local_var(A_batch, kernel_size)

    # ÅúÁ¿¶¯Ì¬ãÐÖµ
    dynamic_threshold = batch_variance_to_threshold(local, min_thr, max_thr)

    # ÅúÁ¿¾í»ý´¦Àí
    TD_10 = (torch.abs(A_batch) >= 1).float() * A_batch
    TD_3 = batch_conv_threshold(TD_10, kernel_size, dynamic_threshold)
    TD_4 = batch_conv_threshold_2(TD_3, kernel_size, 5)
    TD_4 = TD_4.squeeze(1)
    # ÅúÁ¿Éú³É±êÇ©
    label_array = torch.full_like(A_batch, -1, device=A_batch.device)
    label_array[TD_4 != 0] = 1
    label_array[A_batch == 0] = 0

    return TD_4

#### ablation and baseline
# FTF fixed threshold filter 
def thr_filter_td(last_lsd, last_rsd, LSD_reconstructed, RSD_reconstructed, A_tensor, td_param):
    thr=2
    # A=torch.from_numpy(A)
    A_1=(torch.abs(A_tensor)>=thr).float()*A_tensor
    TD_4=A_1.clone()
    # A_tensor =A
    label_array = torch.full(TD_4.shape, -1, device=A_tensor.device)  # 初始化为 -1
    label_array[TD_4 != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    # The code `label_array` is likely a variable or an array in Python. Without seeing the specific
    # code inside the `label_array` variable or array, it is not possible to determine exactly what it
    # is doing. The code snippet you provided is not complete, so more context or code is needed to
    # provide a more accurate explanation.
    label_array[A_tensor == 0] = 0           # A 中为 0 的区域设为 0
    return TD_4, label_array
def thr_filter_sd(A_tensor,B_tensor, sd_param):
    thr=2
    # A=torch.from_numpy(A)
    A_1=(torch.abs(A_tensor)>=thr).float()*A_tensor
    SDL=A_1.clone()
    # A_tensor =A
    label_l_array = torch.full(SDL.shape, -1, device=A_tensor.device)  # 初始化为 -1
    label_l_array[SDL != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_l_array[A_tensor == 0] = 0           # A 中为 0 的区域设为 0

    B_1=(torch.abs(B_tensor)>=thr).float()*B_tensor
    SDR=B_1.clone()
    # A_tensor =A
    label_r_array = torch.full(SDR.shape, -1, device=A_tensor.device) # 初始化为 -1
    label_r_array[SDR != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_r_array[B_tensor == 0] = 0           # A 中为 0 的区域设为 0

    return SDL, label_l_array, SDR, label_r_array

def gradient_density_filter_sd(A_tensor, B_tensor, sd_param):
    result = torch.zeros(160, 320, device=A_tensor.device)  # 初始化目标张量

    # 重建SD阵列
    result[:, ::2] = A_tensor.clone()
    result[:, 1::2] = B_tensor.clone()

    TD_10 = (torch.abs(result) >= 1).float() * result
    TD_3 = conv_and_threshold(TD_10, 3, 3)
    TD_4 = conv_and_threshold_2(TD_3, 3, 3)
    TD_4 = TD_4.squeeze(1)
    # ÅúÁ¿Éú³É±êÇ©
    result = TD_4

    final_result =result
    # 重建SDL与SDR
    final_l = final_result[:, ::2]
    final_r = final_result[:, 1::2]

    # A_tensor = torch.from_numpy(A)
    label_array_l = torch.full(final_l.shape, -1, device=final_l.device)  # 初始化为 -1
    label_array_l[final_l != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array_l[A_tensor == 0] = 0

    # B_tensor = torch.from_numpy(B)
    label_array_r = torch.full(final_r.shape, -1, device=final_l.device)  # 初始化为 -1
    label_array_r[final_r != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array_r[B_tensor == 0] = 0

    return final_l, label_array_l, final_r, label_array_r
    
def gradient_density_filter_td(last_lsd, last_rsd, LSD_reconstructed, RSD_reconstructed, A_tensor, td_param):

    TD_10 = (torch.abs(A_tensor) >= 1).float() * A_tensor
    TD_3 = conv_and_threshold(TD_10, 3, 3)
    TD_4 = conv_and_threshold_2(TD_3, 3, 3)
    # A_tensor = A
    label_array = torch.full(A_tensor.shape, -1, device=A_tensor.device)   # 初始化为 -1
    label_array[TD_4 != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array[A_tensor == 0] = 0  # A 中为 0 的区域设为 0
    result = TD_4
    return result, label_array

def td_adaptive_filter(A, min_thr=3, max_thr=8, kernel_size=3):
    # 计算局部方差
    # local = local_variance(A, kernel_size)
    local = local_var_test(A, kernel_size)

    # 将方差映射到动态阈值
    dynamic_threshold = variance_to_threshold_td(local, min_thr, max_thr)
    # dynamic_threshold_np = dynamic_threshold.numpy()
    TD_10 = (torch.abs(A) >= 1).float() * A
    TD_3 = conv_and_threshold(TD_10, kernel_size, dynamic_threshold)
    TD_4 = conv_and_threshold_2(TD_3, kernel_size, 5)
    A_tensor = A
    label_array = torch.full(A_tensor.shape, -1, device=A_tensor.device)   # 初始化为 -1
    label_array[TD_4 != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array[A_tensor == 0] = 0  # A 中为 0 的区域设为 0
    return TD_4
def sd_adaptive_filter(A, min_thr=3, max_thr=8, kernel_size=3):
    # 计算局部方差
    # t0 = time.time()
    local = local_var_test(A, kernel_size)
    # t1 = time.time()
    # 将方差映射到动态阈值
    dynamic_threshold = variance_to_threshold(local, min_thr, max_thr)
    # t2 = time.time()

    # dynamic_threshold_np = dynamic_threshold.numpy()
    TD_10 = (torch.abs(A) >= 1).float() * A
    TD_3 = conv_and_threshold(TD_10, kernel_size, dynamic_threshold)
    # t3 = time.time()

    TD_4 = conv_and_threshold_2(TD_3, kernel_size, 5)
    # t4 = time.time()
    # print(f"SD adaptive filter time {(t1 - t0)*1000:.4f} ms, {(t2-t1)*1000:.4f} ms, {(t3-t2)*1000:.4f} ms, {(t4-t3)*1000:.4f} ms")
    A_tensor = A
    label_array = torch.full(A_tensor.shape, -1, device=A_tensor.device)  # 初始化为 -1
    label_array[TD_4 != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array[A_tensor == 0] = 0  # A 中为 0 的区域设为 0
    return TD_4


def sd_denoise_4dir_adp_oursDpA(A_tensor, B_tensor, sd_params):
    # 假设两个输入张量
    # tensor1 = torch.from_numpy(A)  # 第一个 160x160 张量
    # tensor2 = torch.from_numpy(B)  # 第二个 160x160 张量

    result = torch.zeros(160, 320, device=A_tensor.device)  # 初始化目标张量

    var_fil_ksize = sd_params["var_fil_ksize"] #3
    var_th = sd_params["var_th"] #0.5
    adapt_th_min = sd_params["adapt_th_min"] #3
    adapt_th_max = sd_params["adapt_th_max"]  #8
    # 重建SD阵列
    result[:, ::2] = A_tensor.clone()
    result[:, 1::2] = B_tensor.clone()

    result_copy = result

    result_1 = result_copy.clone()
    result_1[0::2, 0::2] = -result_1[0::2, 0::2]
    result_2 = result_copy.clone()
    result_2[0::2, 1::2] = -result_2[0::2, 1::2]
    result_3 = result_copy.clone()
    result_3[0::2, ...] = -result_3[0::2, ...]
    result_4 = result_copy.clone()
    result_4[..., 0::2] = -result_4[..., 0::2]

    result_1_pos = result_1 * (result_1 >= 1).float()
    result_1_neg = result_1 * (result_1 <= -1).float()
    result_2_pos = result_2 * (result_2 >= 1).float()
    result_2_neg = result_2 * (result_2 <= -1).float()
    result_3_pos = result_3 * (result_3 >= 1).float()
    result_3_neg = result_3 * (result_3 <= -1).float()
    result_4_pos = result_4 * (result_4 >= 1).float()
    result_4_neg = result_4 * (result_4 <= -1).float()
    t2 = time.time()

    inputs = torch.stack([
        result_1_pos, result_2_pos, result_3_pos, result_4_pos,
        result_1_neg, result_2_neg, result_3_neg, result_4_neg
    ], dim=0)  # shape: [8, C, H, W]
    # ÅúÁ¿´¦Àí
    batch_output = batch_sd_adaptive_filter(inputs, adapt_th_min, adapt_th_max)

    # ²ð½â½á¹û
    (ad_1_pos, ad_2_pos, ad_3_pos, ad_4_pos,
     ad_1_neg, ad_2_neg, ad_3_neg, ad_4_neg) = batch_output.unbind(0)

    # t3 = time.time()
    # print(f"SD: adapt Var {(t3-t2)*1000:.4f} ms")

    # 四方向融合
    ad_sum = ad_1_pos + ad_2_pos + ad_3_pos + ad_4_pos - ad_1_neg - ad_2_neg - ad_3_neg - ad_4_neg
    final_result = (ad_sum > 0).float() * result
    # 重建SDL与SDR
    final_l = final_result[:, ::2]
    final_r = final_result[:, 1::2]

    # A_tensor = torch.from_numpy(A)
    label_array_l = torch.full(final_l.shape, -1, device=final_l.device)  # 初始化为 -1
    label_array_l[final_l != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array_l[A_tensor == 0] = 0

    # B_tensor = torch.from_numpy(B)
    label_array_r = torch.full(final_r.shape, -1, device=final_l.device)  # 初始化为 -1
    label_array_r[final_r != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array_r[B_tensor == 0] = 0

    return final_l, label_array_l, final_r, label_array_r
def sd_denoise_onlyadp_oursA(A_tensor, B_tensor, sd_params):
    # 假设两个输入张量
    # tensor1 = torch.from_numpy(A)  # 第一个 160x160 张量
    # tensor2 = torch.from_numpy(B)  # 第二个 160x160 张量

    result = torch.zeros(160, 320, device=A_tensor.device)  # 初始化目标张量

    var_fil_ksize = sd_params["var_fil_ksize"] #3
    var_th = sd_params["var_th"] #0.5
    adapt_th_min = sd_params["adapt_th_min"] #3
    adapt_th_max = sd_params["adapt_th_max"]  #8
    # 重建SD阵列
    result[:, ::2] = A_tensor.clone()
    result[:, 1::2] = B_tensor.clone()



    inputs = torch.stack([
    result
    ], dim=0)  # shape: [8, C, H, W]
    # ÅúÁ¿´¦Àí
    batch_output = batch_sd_adaptive_filter(inputs, adapt_th_min, adapt_th_max)

    # ²ð½â½á¹û
    (result) = batch_output.unbind(0)[0]

    final_result =result
    # 重建SDL与SDR
    final_l = final_result[:, ::2]
    final_r = final_result[:, 1::2]

    # A_tensor = torch.from_numpy(A)
    label_array_l = torch.full(final_l.shape, -1, device=final_l.device)  # 初始化为 -1
    label_array_l[final_l != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array_l[A_tensor == 0] = 0

    # B_tensor = torch.from_numpy(B)
    label_array_r = torch.full(final_r.shape, -1, device=final_l.device)  # 初始化为 -1
    label_array_r[final_r != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array_r[B_tensor == 0] = 0

    return final_l, label_array_l, final_r, label_array_r


def sd_denoise_var_adp_oursVpA(A_tensor, B_tensor, sd_params):
    result = torch.zeros(160, 320, device=A_tensor.device)  # 初始化目标张量

    var_fil_ksize = sd_params["var_fil_ksize"] #3
    var_th = sd_params["var_th"] #0.5
    adapt_th_min = sd_params["adapt_th_min"] #3
    adapt_th_max = sd_params["adapt_th_max"]  #8
    # 重建SD阵列
    result[:, ::2] = A_tensor.clone()
    result[:, 1::2] = B_tensor.clone()

    # 方差阈值预去噪
    # t0 = time.time()
    result_copy = local_var(result, var_fil_ksize, 0.5, var_th)
    #result_copy = result
    inputs = torch.stack([
    result_copy
    ], dim=0)  # shape: [8, C, H, W]
    # ÅúÁ¿´¦Àí
    batch_output = batch_sd_adaptive_filter(inputs, adapt_th_min, adapt_th_max)

    # ²ð½â½á¹û
    (final_result) = batch_output.unbind(0)[0]

    final_l = final_result[:, ::2]
    final_r = final_result[:, 1::2]

    # A_tensor = torch.from_numpy(A)
    label_array_l = torch.full(final_l.shape, -1, device=final_l.device)  # 初始化为 -1
    label_array_l[final_l != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array_l[A_tensor == 0] = 0

    # B_tensor = torch.from_numpy(B)
    label_array_r = torch.full(final_r.shape, -1, device=final_l.device)  # 初始化为 -1
    label_array_r[final_r != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array_r[B_tensor == 0] = 0

    return final_l, label_array_l, final_r, label_array_r
    
def sd_denoise_onlyvar_oursV(A_tensor, B_tensor, sd_params):
    result = torch.zeros(160, 320, device=A_tensor.device)  # 初始化目标张量

    var_fil_ksize = sd_params["var_fil_ksize"] #3
    var_th = sd_params["var_th"] #0.5
    adapt_th_min = sd_params["adapt_th_min"] #3
    adapt_th_max = sd_params["adapt_th_max"]  #8
    # 重建SD阵列
    result[:, ::2] = A_tensor.clone()
    result[:, 1::2] = B_tensor.clone()

    final_result = local_var(result, var_fil_ksize, 0.5, var_th)


    final_l = final_result[:, ::2]
    final_r = final_result[:, 1::2]

    label_array_l = torch.full(final_l.shape, -1, device=final_l.device)  # 初始化为 -1
    label_array_l[final_l != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array_l[A_tensor == 0] = 0

    label_array_r = torch.full(final_r.shape, -1, device=final_l.device)  # 初始化为 -1
    label_array_r[final_r != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array_r[B_tensor == 0] = 0

    return final_l, label_array_l, final_r, label_array_r

## TD ##
def td_denoise_onlyvar_oursV(td_tensor, td_params):
    td_raw = td_tensor.float()#torch.from_numpy(td)
    td_orin = td_raw.clone()
    var_fil_ksize = td_params["var_fil_ksize"] #3
    var_th = td_params["var_th"] #0.5
    adapt_th_min = td_params["adapt_th_min"] #3
    adapt_th_max = td_params["adapt_th_max"]  #8
    
    td_orin = local_var(td_orin, var_fil_ksize, 0.5, var_th)
    # 四个方向引导结果融合
    rusult = td_orin

    A_tensor = td_raw
    label_array = torch.full(A_tensor.shape, -1, device=A_tensor.device)  # 初始化为 -1
    label_array[rusult != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array[A_tensor == 0] = 0

    return rusult, label_array
    
def td_denoise_onlyadp_oursA(sdl_1_denoised, sdr_1_denoised, sdl_2_denoised, sdr_2_denoised, td_tensor, td_params):
    td_raw = td_tensor.float()#torch.from_numpy(td)
    td_orin = td_raw.clone()
    var_fil_ksize = td_params["var_fil_ksize"] #3
    var_th = td_params["var_th"] #0.5
    adapt_th_min = td_params["adapt_th_min"] #3
    adapt_th_max = td_params["adapt_th_max"]  #8

    result = td_orin
    result = td_adaptive_filter(result, min_thr=adapt_th_min, max_thr=adapt_th_max, kernel_size=3)

    A_tensor = td_raw
    label_array = torch.full(A_tensor.shape, -1, device=A_tensor.device)  # 初始化为 -1
    label_array[result != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array[A_tensor == 0] = 0

    return result, label_array
    
def td_denoise_var_adp_oursVpANG(td_tensor, td_params):
    td_raw = td_tensor.float()#torch.from_numpy(td)
    td_orin = td_raw.clone()
    var_fil_ksize = td_params["var_fil_ksize"] #3
    var_th = td_params["var_th"] #0.5
    adapt_th_min = td_params["adapt_th_min"] #3
    adapt_th_max = td_params["adapt_th_max"]  #8
    
    td_orin = local_var(td_orin, var_fil_ksize, 0.5, var_th)
    # 四个方向引导结果融合
    rusult = td_orin
    # 自适应滤波
    rusult = td_adaptive_filter(rusult, min_thr=adapt_th_min, max_thr=adapt_th_max, kernel_size=3)

    # print(f"TD preprocess time {(t1-t0)*1000:.4f} ms {(t2-t1)*1000:.4f} ms, {(t3-t2)*1000:.4f} ms, {(t4-t3)*1000:.4f} ms, {(t5-t4)*1000:.4f} ms, \
    #       {(t6-t5)*1000:.4f} ms")

    A_tensor = td_raw
    label_array = torch.full(A_tensor.shape, -1, device=A_tensor.device)  # 初始化为 -1
    label_array[rusult != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array[A_tensor == 0] = 0

    return rusult, label_array
    
def td_denoise_guided_adp_oursApG(sdl_1_denoised, sdr_1_denoised, sdl_2_denoised, sdr_2_denoised, td_tensor, td_params):
    td_raw = td_tensor.float()#torch.from_numpy(td)
    td_orin = td_raw.clone()
    var_fil_ksize = td_params["var_fil_ksize"] #3
    var_th = td_params["var_th"] #0.5
    adapt_th_min = td_params["adapt_th_min"] #3
    adapt_th_max = td_params["adapt_th_max"]  #8
    t0 = time.time()
    # TD方差阈值滤波

    t1 = time.time()
    # 由TD计算TSD（L）
    tdl_orin = torch.empty_like(td_orin, device=td_orin.device)
    zero_tensor_1 = torch.zeros(80, 160).to(td_orin.device)
    zero_tensor_1[0:79, ...] = td_orin[2::2, :].clone()
    tdl_orin[::2, :] = td_orin[1::2, :] - td_orin[::2, :]
    tdl_orin[1::2, :] = zero_tensor_1 - td_orin[1::2, :]

    # 由TD计算TSD（R）
    tdr_orin = torch.empty_like(td_orin, device=td_orin.device)
    orin = torch.empty_like(td_orin, device=td_orin.device)
    orin[::2, :-1] = td_orin[::2, 1:]
    orin[1::2, ] = td_orin[1::2, ]
    orin[::2, -1] = 0
    zero_tensor_2 = torch.zeros(80, 160).to(td_orin.device)
    zero_tensor_2[0:79, ...] = orin[2::2, :].clone()
    tdr_orin[::2, :] = orin[1::2, :] - orin[::2, :]
    tdr_orin[1::2, :] = zero_tensor_2 - orin[1::2, :]
    t2 = time.time()
    # 去噪后的干净SD
    sdl_1_cali = sdl_1_denoised
    sdr_1_cali = sdr_1_denoised
    sdl_2_cali = sdl_2_denoised
    sdr_2_cali = sdr_2_denoised

    ###以下为SDL引导TSDL去噪
    tensor1 = sdl_1_cali
    tensor2 = sdl_2_cali
    # SDL区域1  SD（t-1）与SD（t）极性不相同的区域
    polarities_diff = torch.sign(tensor1) != torch.sign(tensor2)
    # SDL区域2  SD（t-1）与SD（t）极性不相同的区域极性相同且值为 0 的区域
    polarities_same_and_zero = (torch.sign(tensor1) == torch.sign(tensor2)) & (tensor1 == 0) & (tensor2 == 0)
    # SDL区域3  SD（t-1）与SD（t）极性相同且不为 0 的区域
    polarities_same_and_nonzero = (torch.sign(tensor1) == torch.sign(tensor2)) & (tensor1 != 0) & (tensor2 != 0)

    # 区域1去噪
    ref_tensor = (tensor2 - tensor1) * polarities_diff.float()
    cmp_tensor = tdl_orin * polarities_diff.float()
    ref_sign = torch.sign(ref_tensor)
    cmp_sign = torch.sign(cmp_tensor)
    mask = (ref_sign == cmp_sign)
    result_tensor = torch.where(mask, cmp_tensor, torch.zeros_like(cmp_tensor, device=td_orin.device))

    # 区域2去噪
    polarity_flag = torch.zeros_like(td_orin, dtype=torch.bool , device=td_orin.device)
    polarity_flag[::2, :] = ((td_orin[1::2, :]) == torch.sign(td_orin[::2, :]))
    polarity_flag[1::2, :] = torch.sign(zero_tensor_1) == torch.sign(td_orin[1::2, :])

    # TSDL去噪结果
    final_l = polarity_flag.float() * tdl_orin * polarities_same_and_zero.float() + result_tensor + tdl_orin * polarities_same_and_nonzero.float()
    t3 = time.time()
    ###以下为SDL引导TSDL去噪
    tensor1 = sdr_1_cali
    tensor2 = sdr_2_cali

    # SDL区域1  SD（t-1）与SD（t）极性不相同的区域
    polarities_diff = torch.sign(tensor1) != torch.sign(tensor2)
    # SDL区域2  SD（t-1）与SD（t）极性不相同的区域极性相同且值为 0 的区域
    polarities_same_and_zero = (torch.sign(tensor1) == torch.sign(tensor2)) & (tensor1 == 0) & (tensor2 == 0)
    # SDL区域3  SD（t-1）与SD（t）极性相同且不为 0 的区域
    polarities_same_and_nonzero = (torch.sign(tensor1) == torch.sign(tensor2)) & (tensor1 != 0) & (tensor2 != 0)

    ###区域1去噪
    # 计算两个张量的符号
    ref_tensor = (tensor2 - tensor1) * polarities_diff.float()
    cmp_tensor = tdr_orin * polarities_diff.float()
    ref_sign = torch.sign(ref_tensor)
    cmp_sign = torch.sign(cmp_tensor)
    # 创建布尔掩码，极性相同的位置为 True，极性不同的位置为 False
    mask = (ref_sign == cmp_sign)
    # 使用布尔掩码保留 cmp_tensor 中极性相同的像素，置零极性不同的像素
    result_tensor = torch.where(mask, cmp_tensor, torch.zeros_like(cmp_tensor))

    ###区域2去噪
    # 计算极性标志，比较相邻行的极性是否相同
    polarity_flag = torch.zeros_like(orin, dtype=torch.bool, device=td_orin.device)
    polarity_flag[::2, :] = ((orin[1::2, :]) == torch.sign(orin[::2, :]))
    polarity_flag[1::2, :] = torch.sign(zero_tensor_2) == torch.sign(orin[1::2, :])
    # TSDR去噪结果
    final_r = polarity_flag.float() * tdr_orin * polarities_same_and_zero.float() + result_tensor + tdr_orin * polarities_same_and_nonzero.float()
    t4 = time.time()
    # 由去噪后的TSD重建TD，统一TSD对应的两个TD都保留
    tsd_lu_orin = final_l[::2, :]
    tsd_ll_orin = final_l[1::2, :]
    tsd_ru_orin = final_r[::2, :]
    tsd_rl_orin = final_r[1::2, :]
    tsd_lu_to_td = torch.zeros(160, 160).to(td_tensor.device)
    tsd_ll_to_td = torch.zeros(160, 160).to(td_tensor.device)
    tsd_ru_to_td = torch.zeros(160, 160).to(td_tensor.device)
    tsd_rl_to_td = torch.zeros(160, 160).to(td_tensor.device)
    tsd_lu_to_td[::2, :] = tsd_lu_orin
    tsd_lu_to_td[1::2, :] = tsd_lu_orin
    tsd_ll_to_td[1::2, :] = tsd_ll_orin
    tsd_ll_to_td[2::2, :] = tsd_ll_orin[0:79, ...]
    tsd_ru_to_td[::2, 1:] = tsd_ru_orin[..., 0:159]
    tsd_ru_to_td[1::2, :] = tsd_ru_orin
    tsd_rl_to_td[2::2, 1:] = tsd_rl_orin[0:79, 0:159]
    tsd_rl_to_td[1::2, :] = tsd_rl_orin
    t5 = time.time()
    # 四个方向引导结果融合
    rusult = ((torch.abs(tsd_lu_to_td) > 0).float() + (torch.abs(tsd_ll_to_td) > 0).float() + (
                torch.abs(tsd_ru_to_td) > 0).float() + (torch.abs(tsd_rl_to_td) > 0).float()) * td_orin
    # 自适应滤波
    rusult = td_adaptive_filter(rusult, min_thr=adapt_th_min, max_thr=adapt_th_max, kernel_size=3)
    t6 = time.time()
    # print(f"TD preprocess time {(t1-t0)*1000:.4f} ms {(t2-t1)*1000:.4f} ms, {(t3-t2)*1000:.4f} ms, {(t4-t3)*1000:.4f} ms, {(t5-t4)*1000:.4f} ms, \
    #       {(t6-t5)*1000:.4f} ms")

    A_tensor = td_raw
    label_array = torch.full(A_tensor.shape, -1, device=A_tensor.device)  # 初始化为 -1
    label_array[rusult != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array[A_tensor == 0] = 0

    return rusult, label_array
    
##################################################################
### Final version of TD and SD denoise

def sd_denoise(A_tensor, B_tensor, sd_params):
    # 假设两个输入张量
    # tensor1 = torch.from_numpy(A)  # 第一个 160x160 张量
    # tensor2 = torch.from_numpy(B)  # 第二个 160x160 张量

    result = torch.zeros(160, 320, device=A_tensor.device)  # 初始化目标张量
    # result_1 = torch.zeros(160, 320)  # 初始化目标张量
    # result_2 = torch.zeros(160, 320)  # 初始化目标张量
    # result_3 = torch.zeros(160, 320)  # 初始化目标张量
    # result_4 = torch.zeros(160, 320)  # 初始化目标张量
    var_fil_ksize = sd_params["var_fil_ksize"] #3
    var_th = sd_params["var_th"] #0.5
    adapt_th_min = sd_params["adapt_th_min"] #3
    adapt_th_max = sd_params["adapt_th_max"]  #8
    # 重建SD阵列
    result[:, ::2] = A_tensor.clone()
    result[:, 1::2] = B_tensor.clone()

    # 方差阈值预去噪
    # t0 = time.time()
    result_copy = local_var(result, var_fil_ksize, 0.5, var_th)
    # delta_t = (time.time() - t0) * 1000
    # print(f"SD: local_var time {delta_t:.4f} ms")
    # t1 = time.time()
    # 四方向极性调整
    result_1 = result_copy.clone()
    result_1[0::2, 0::2] = -result_1[0::2, 0::2]
    result_2 = result_copy.clone()
    result_2[0::2, 1::2] = -result_2[0::2, 1::2]
    result_3 = result_copy.clone()
    result_3[0::2, ...] = -result_3[0::2, ...]
    result_4 = result_copy.clone()
    result_4[..., 0::2] = -result_4[..., 0::2]
    # 四方向正负信号提取
    # t11 = time.time()
    # print(f"SD: pre process time 1 {(t11 - t1) * 1000:.4f} ms")
    result_1_pos = result_1 * (result_1 >= 1).float()
    result_1_neg = result_1 * (result_1 <= -1).float()
    result_2_pos = result_2 * (result_2 >= 1).float()
    result_2_neg = result_2 * (result_2 <= -1).float()
    result_3_pos = result_3 * (result_3 >= 1).float()
    result_3_neg = result_3 * (result_3 <= -1).float()
    result_4_pos = result_4 * (result_4 >= 1).float()
    result_4_neg = result_4 * (result_4 <= -1).float()
    t2 = time.time()
    # print(f"SD: pre process time 2 {(t2-t11)*1000:.4f} ms")

    # 四方向正负信号自适应滤波
    ####################### Serial ###############################
    # ad_1_pos = sd_adaptive_filter(result_1_pos, min_thr=adapt_th_min, max_thr=adapt_th_max, kernel_size=3)
    # ad_2_pos = sd_adaptive_filter(result_2_pos, min_thr=adapt_th_min, max_thr=adapt_th_max, kernel_size=3)
    # ad_3_pos = sd_adaptive_filter(result_3_pos, min_thr=adapt_th_min, max_thr=adapt_th_max, kernel_size=3)
    # ad_4_pos = sd_adaptive_filter(result_4_pos, min_thr=adapt_th_min, max_thr=adapt_th_max, kernel_size=3)
    # ad_1_neg = sd_adaptive_filter(result_1_neg, min_thr=adapt_th_min, max_thr=adapt_th_max, kernel_size=3)
    # ad_2_neg = sd_adaptive_filter(result_2_neg, min_thr=adapt_th_min, max_thr=adapt_th_max, kernel_size=3)
    # ad_3_neg = sd_adaptive_filter(result_3_neg, min_thr=adapt_th_min, max_thr=adapt_th_max, kernel_size=3)
    # ad_4_neg = sd_adaptive_filter(result_4_neg, min_thr=adapt_th_min, max_thr=adapt_th_max, kernel_size=3)
    ####################### bacth ###############################
    inputs = torch.stack([
        result_1_pos, result_2_pos, result_3_pos, result_4_pos,
        result_1_neg, result_2_neg, result_3_neg, result_4_neg
    ], dim=0)  # shape: [8, C, H, W]
    # ÅúÁ¿´¦Àí
    batch_output = batch_sd_adaptive_filter(inputs, adapt_th_min, adapt_th_max)

    # ²ð½â½á¹û
    (ad_1_pos, ad_2_pos, ad_3_pos, ad_4_pos,
     ad_1_neg, ad_2_neg, ad_3_neg, ad_4_neg) = batch_output.unbind(0)

    # t3 = time.time()
    # print(f"SD: adapt Var {(t3-t2)*1000:.4f} ms")

    # 四方向融合
    ad_sum = ad_1_pos + ad_2_pos + ad_3_pos + ad_4_pos - ad_1_neg - ad_2_neg - ad_3_neg - ad_4_neg
    final_result = (ad_sum > 0).float() * result
    # 重建SDL与SDR
    final_l = final_result[:, ::2]
    final_r = final_result[:, 1::2]

    # A_tensor = torch.from_numpy(A)
    label_array_l = torch.full(final_l.shape, -1, device=final_l.device)  # 初始化为 -1
    label_array_l[final_l != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array_l[A_tensor == 0] = 0

    # B_tensor = torch.from_numpy(B)
    label_array_r = torch.full(final_r.shape, -1, device=final_l.device)  # 初始化为 -1
    label_array_r[final_r != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array_r[B_tensor == 0] = 0

    return final_l, label_array_l, final_r, label_array_r

def td_denoise_init(A_tensor, td_param):
    # A = torch.from_numpy(A).to(device)
    A_1 = (torch.abs(A_tensor) >= 1).float() * A_tensor
    sub_tensor = A_1.unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32).to(A_tensor.device) / 9.0  # 正则化使得总和为1
    smoothed_matrix = F.conv2d(sub_tensor, kernel, padding=1)
    smoothed_matrix = smoothed_matrix.squeeze(0).squeeze(0)
    sub_tensor = sub_tensor.squeeze(0).squeeze(0)
    mask_1 = torch.abs(smoothed_matrix) >= 0.5
    TD_10 = sub_tensor * mask_1
    TD_3 = conv_and_threshold(TD_10, 3, 5)
    TD_4 = conv_and_threshold_2(TD_3, 3, 5)

    # A_tensor = A
    label_array = torch.full(A_tensor.shape, -1).to(A_tensor.device)  # 初始化为 -1
    label_array[TD_4 != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array[A_tensor == 0] = 0  # A 中为 0 的区域设为 0
    return TD_4, label_array

def td_denoise(sdl_1_denoised, sdr_1_denoised, sdl_2_denoised, sdr_2_denoised, td_tensor, td_params):
    td_raw = td_tensor.float()#torch.from_numpy(td)
    td_orin = td_raw.clone()
    var_fil_ksize = td_params["var_fil_ksize"] #3
    var_th = td_params["var_th"] #0.5
    adapt_th_min = td_params["adapt_th_min"] #3
    adapt_th_max = td_params["adapt_th_max"]  #8
    t0 = time.time()
    # TD方差阈值滤波
    td_orin = local_var(td_orin, var_fil_ksize, 0.5, var_th)
    t1 = time.time()
    # 由TD计算TSD（L）
    tdl_orin = torch.empty_like(td_orin, device=td_orin.device)
    zero_tensor_1 = torch.zeros(80, 160).to(td_orin.device)
    zero_tensor_1[0:79, ...] = td_orin[2::2, :].clone()
    tdl_orin[::2, :] = td_orin[1::2, :] - td_orin[::2, :]
    tdl_orin[1::2, :] = zero_tensor_1 - td_orin[1::2, :]

    # 由TD计算TSD（R）
    tdr_orin = torch.empty_like(td_orin, device=td_orin.device)
    orin = torch.empty_like(td_orin, device=td_orin.device)
    orin[::2, :-1] = td_orin[::2, 1:]
    orin[1::2, ] = td_orin[1::2, ]
    orin[::2, -1] = 0
    zero_tensor_2 = torch.zeros(80, 160).to(td_orin.device)
    zero_tensor_2[0:79, ...] = orin[2::2, :].clone()
    tdr_orin[::2, :] = orin[1::2, :] - orin[::2, :]
    tdr_orin[1::2, :] = zero_tensor_2 - orin[1::2, :]
    t2 = time.time()
    # 去噪后的干净SD
    sdl_1_cali = sdl_1_denoised
    sdr_1_cali = sdr_1_denoised
    sdl_2_cali = sdl_2_denoised
    sdr_2_cali = sdr_2_denoised

    ###以下为SDL引导TSDL去噪
    tensor1 = sdl_1_cali
    tensor2 = sdl_2_cali
    # SDL区域1  SD（t-1）与SD（t）极性不相同的区域
    polarities_diff = torch.sign(tensor1) != torch.sign(tensor2)
    # SDL区域2  SD（t-1）与SD（t）极性不相同的区域极性相同且值为 0 的区域
    polarities_same_and_zero = (torch.sign(tensor1) == torch.sign(tensor2)) & (tensor1 == 0) & (tensor2 == 0)
    # SDL区域3  SD（t-1）与SD（t）极性相同且不为 0 的区域
    polarities_same_and_nonzero = (torch.sign(tensor1) == torch.sign(tensor2)) & (tensor1 != 0) & (tensor2 != 0)

    # 区域1去噪
    ref_tensor = (tensor2 - tensor1) * polarities_diff.float()
    cmp_tensor = tdl_orin * polarities_diff.float()
    ref_sign = torch.sign(ref_tensor)
    cmp_sign = torch.sign(cmp_tensor)
    mask = (ref_sign == cmp_sign)
    result_tensor = torch.where(mask, cmp_tensor, torch.zeros_like(cmp_tensor, device=td_orin.device))

    # 区域2去噪
    polarity_flag = torch.zeros_like(td_orin, dtype=torch.bool , device=td_orin.device)
    polarity_flag[::2, :] = ((td_orin[1::2, :]) == torch.sign(td_orin[::2, :]))
    polarity_flag[1::2, :] = torch.sign(zero_tensor_1) == torch.sign(td_orin[1::2, :])

    # TSDL去噪结果
    final_l = polarity_flag.float() * tdl_orin * polarities_same_and_zero.float() + result_tensor + tdl_orin * polarities_same_and_nonzero.float()
    t3 = time.time()
    ###以下为SDL引导TSDL去噪
    tensor1 = sdr_1_cali
    tensor2 = sdr_2_cali

    # SDL区域1  SD（t-1）与SD（t）极性不相同的区域
    polarities_diff = torch.sign(tensor1) != torch.sign(tensor2)
    # SDL区域2  SD（t-1）与SD（t）极性不相同的区域极性相同且值为 0 的区域
    polarities_same_and_zero = (torch.sign(tensor1) == torch.sign(tensor2)) & (tensor1 == 0) & (tensor2 == 0)
    # SDL区域3  SD（t-1）与SD（t）极性相同且不为 0 的区域
    polarities_same_and_nonzero = (torch.sign(tensor1) == torch.sign(tensor2)) & (tensor1 != 0) & (tensor2 != 0)

    ###区域1去噪
    # 计算两个张量的符号
    ref_tensor = (tensor2 - tensor1) * polarities_diff.float()
    cmp_tensor = tdr_orin * polarities_diff.float()
    ref_sign = torch.sign(ref_tensor)
    cmp_sign = torch.sign(cmp_tensor)
    # 创建布尔掩码，极性相同的位置为 True，极性不同的位置为 False
    mask = (ref_sign == cmp_sign)
    # 使用布尔掩码保留 cmp_tensor 中极性相同的像素，置零极性不同的像素
    result_tensor = torch.where(mask, cmp_tensor, torch.zeros_like(cmp_tensor))

    ###区域2去噪
    # 计算极性标志，比较相邻行的极性是否相同
    polarity_flag = torch.zeros_like(orin, dtype=torch.bool, device=td_orin.device)
    polarity_flag[::2, :] = ((orin[1::2, :]) == torch.sign(orin[::2, :]))
    polarity_flag[1::2, :] = torch.sign(zero_tensor_2) == torch.sign(orin[1::2, :])
    # TSDR去噪结果
    final_r = polarity_flag.float() * tdr_orin * polarities_same_and_zero.float() + result_tensor + tdr_orin * polarities_same_and_nonzero.float()
    t4 = time.time()
    # 由去噪后的TSD重建TD，统一TSD对应的两个TD都保留
    tsd_lu_orin = final_l[::2, :]
    tsd_ll_orin = final_l[1::2, :]
    tsd_ru_orin = final_r[::2, :]
    tsd_rl_orin = final_r[1::2, :]
    tsd_lu_to_td = torch.zeros(160, 160).to(td_tensor.device)
    tsd_ll_to_td = torch.zeros(160, 160).to(td_tensor.device)
    tsd_ru_to_td = torch.zeros(160, 160).to(td_tensor.device)
    tsd_rl_to_td = torch.zeros(160, 160).to(td_tensor.device)
    tsd_lu_to_td[::2, :] = tsd_lu_orin
    tsd_lu_to_td[1::2, :] = tsd_lu_orin
    tsd_ll_to_td[1::2, :] = tsd_ll_orin
    tsd_ll_to_td[2::2, :] = tsd_ll_orin[0:79, ...]
    tsd_ru_to_td[::2, 1:] = tsd_ru_orin[..., 0:159]
    tsd_ru_to_td[1::2, :] = tsd_ru_orin
    tsd_rl_to_td[2::2, 1:] = tsd_rl_orin[0:79, 0:159]
    tsd_rl_to_td[1::2, :] = tsd_rl_orin
    t5 = time.time()
    # 四个方向引导结果融合
    rusult = ((torch.abs(tsd_lu_to_td) > 0).float() + (torch.abs(tsd_ll_to_td) > 0).float() + (
                torch.abs(tsd_ru_to_td) > 0).float() + (torch.abs(tsd_rl_to_td) > 0).float()) * td_orin
    # 自适应滤波
    rusult = td_adaptive_filter(rusult, min_thr=adapt_th_min, max_thr=adapt_th_max, kernel_size=3)
    t6 = time.time()
    # print(f"TD preprocess time {(t1-t0)*1000:.4f} ms {(t2-t1)*1000:.4f} ms, {(t3-t2)*1000:.4f} ms, {(t4-t3)*1000:.4f} ms, {(t5-t4)*1000:.4f} ms, \
    #       {(t6-t5)*1000:.4f} ms")

    A_tensor = td_raw
    label_array = torch.full(A_tensor.shape, -1, device=A_tensor.device)  # 初始化为 -1
    label_array[rusult != 0] = 1  # LSD_reconstructed 为 true 的区域设为 1
    label_array[A_tensor == 0] = 0

    return rusult, label_array


_TMCV_SHOW_FLAG_LVFT = True
def LVFT(rawdiff, args):
    global _TMCV_SHOW_FLAG_LVFT
    try:
       sd_params = args['sd_params']
       td_params = args['td_params']
    except:
        sd_params = {"var_fil_ksize":3,
                     "var_th":0.5,
                     "adapt_th_min":3,
                     "adapt_th_max":8}
        td_params = {"var_fil_ksize":3,
                     "var_th":0.5,
                     "adapt_th_min":3,
                     "adapt_th_max":8}
    if _TMCV_SHOW_FLAG_LVFT:
        print('[LVFT DENOISE]sd_params:',sd_params)
        print('[LVFT DENOISE]td_params:',td_params)
        _TMCV_SHOW_FLAG_LVFT = False

    denoise_raw_diff = torch.zeros_like(rawdiff)    
    for t in range(rawdiff.shape[0]):
        td = rawdiff[0,t,...]    
        sdl = rawdiff[1,t,...]    
        sdr = rawdiff[2,t,...]    
        sdl_dn, _, sdr_dn, _ = sd_denoise_onlyvar_oursV(sdl, sdr, sd_params)
        td_dn,_ = td_denoise_onlyvar_oursV(td, td_params)

        denoise_raw_diff[0,t,...] = td_dn
        denoise_raw_diff[1,t,...] = sdl_dn
        denoise_raw_diff[2,t,...] = sdr_dn

    return denoise_raw_diff

_TMCV_SHOW_FLAG_LVAFT = True
def LVAFT(rawdiff, args):
    global _TMCV_SHOW_FLAG_LVAFT
    try:
       sd_params = args['sd_params']
       td_params = args['td_params']
    except:
        sd_params = {"var_fil_ksize":3,
                     "var_th":0.5,
                     "adapt_th_min":3,
                     "adapt_th_max":8}
        td_params = {"var_fil_ksize":3,
                     "var_th":0.5,
                     "adapt_th_min":3,
                     "adapt_th_max":8}
    if _TMCV_SHOW_FLAG_LVAFT:
        print('[LVAFT DENOISE]sd_params:',sd_params)
        print('[LVAFT DENOISE]td_params:',td_params)
        _TMCV_SHOW_FLAG_LVAFT = False

    denoise_raw_diff = torch.zeros_like(rawdiff)    
    for t in range(rawdiff.shape[0]):
        td = rawdiff[0,t,...]    
        sdl = rawdiff[1,t,...]    
        sdr = rawdiff[2,t,...]    
        sdl_dn, _, sdr_dn, _ = sd_denoise_var_adp_oursVpA(sdl, sdr, sd_params)
        td_dn,_ = td_denoise_var_adp_oursVpANG(td, td_params)

        denoise_raw_diff[0,t,...] = td_dn
        denoise_raw_diff[1,t,...] = sdl_dn
        denoise_raw_diff[2,t,...] = sdr_dn

    return denoise_raw_diff
    