#基础的一些isp操作与可视化函数，有些和算法的效果绑定
__author__ = 'Y. Lin'
__authorEmail__ = '532109881@qq.com'
import numpy as np
from scipy.signal import convolve2d
import cv2
import torch

from .awb import gray_world_awb, AutoWhiteBalance

##############################
#1. 图像基本处理
#2. 可视化
##############################
raw_awb = AutoWhiteBalance()

def default_rgb_isp(raw, blc = 0, gamma = 0.9, raw_input = True):
    '''
    默认的RGB RAW数据处理流程
    
    - 空洞填补
    - 去马赛克
    - 白平衡
    - 自适应降噪
    - 自动饱和度
    - 自动曲线
    - 自动归一化

    注意: 速度非常慢，仅供参考
    '''
    #fill hole
    if raw_input:
        raw = raw.astype(np.float32)
        raw = raw - blc
        raw[raw<0]=0
        raw = lyncam_raw_comp(raw)
        image_demosaic_withoutisp = exp_bayer_to_rgb_conv(raw)
        
        blc_avg = np.mean(blc)
        raw_after_awb = raw_awb(raw,method='GW',blc_avg=blc_avg)
        raw_after_awb[raw_after_awb>1023]=1023
        #adjust gamma
        raw_gamma = (raw_after_awb/1024.0)**(1/gamma)*1024.0

        #demosacing
        image_demosaic = exp_bayer_to_rgb_conv(raw_gamma)
    else:
        image_demosaic_withoutisp = (raw/1024.0)**(1/gamma)*1024.0
        image_demosaic = gray_world_awb(image_demosaic_withoutisp.copy(),HSB=1023)
    
    #adjust_saturation
    #saturation_factor = (96,128)
    #image = adjust_saturation(image,saturation_factor)
    #adjust_curve
    #curve_factor = 0.02
    #image = adjust_curve(image,curve_factor)

    image = image_demosaic.astype(np.float32)
    
    return image/1024.0,image_demosaic_withoutisp/1024.0
            

# ===============================================================
# ToneMapping
# ===============================================================
def ACESToneMapping(color, adapted_lum=1):
    '''
    https://zhuanlan.zhihu.com/p/21983679
    '''
    A = 2.51
    B = 0.03
    C = 2.43
    D = 0.59
    E = 0.14
    color *= adapted_lum
    return (color * (A * color + B)) / (color * (C * color + D) + E)



# ===============================================================
# adjust_curve
# ===============================================================
def Scurve(Y,curve_factor):
    '''
    曲线调整
    :param img: cv2.imread读取的图片数据
    :curve_factor: 增加对比度，值越大对比度越大
    :return: 返回的白平衡结果图片数据
    '''
    x = np.array(range(256))  # 创建0到255的数组
    s_curve = 255 / (1 + np.exp(-curve_factor * (x - 128)))  # 创建S形曲线
    Y = cv2.LUT(Y, s_curve.astype(np.uint8))
    return Y
    
def adjust_curve(image, curve_factor = 0.02):
    '''
    曲线调整
    '''
   # 将RGB图像转换为YCrCb颜色空间
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y = yuv_image[:,:,0]
    yuv_image[:,:,0] = Scurve(Y,curve_factor) #fuji_curve(Y)
    # 将YCrCb图像转换回RGB颜色空间
    #print('curve:',np.max(yuv_image))
    yuv_image[yuv_image>255.0]=255.0
    output_image = cv2.cvtColor(yuv_image, cv2.COLOR_YCrCb2BGR)

    return output_image



def adjust_saturation(image, saturation_factor = (128,256)):
    '''
    
    饱和度调整
     
    :param img: cv2.imread读取的图片数据
    :saturation_factor: 增加饱和度，saturation_factor越大饱和度越大
    :return: 返回的饱和度结果图片数据
    
    '''

    target,max_value = saturation_factor

    # 将图像转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = hsv_image.astype(np.float32)
    # 调整饱和度（增加饱和度可以使用更大的值，减少饱和度可以使用更小的值）
    V = hsv_image[:,:,1]
    
    factor = target / (1e-8+np.mean(V))
    V = V * factor

    V = max_value *(1 -np.exp(-V/max_value) )

    hsv_image[:,:,1] = V
    
    #print('st:',np.max(hsv_image))
    hsv_image = hsv_image.astype(np.uint8)
    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return result_image


# ===============================================================
# Tianmouc hole fill
# ===============================================================
def lyncam_raw_comp(raw):
    '''
    Tianmouc Bayer 填充

    Author: Taoyi Wang
    '''
    width = raw.shape[1] * 2
    height = raw.shape[0] * 1
    #raw_res = np.zeros((height, width), dtype=np.int16)
    #raw_res = cv2.resize(raw, (width, height),interpolation=cv2.INTER_LINEAR)
    raw_hollow = np.zeros((height, width), dtype=np.int16)
    test =   raw[0::4, 3::2]
    raw_hollow[0::4, 2::4], raw_hollow[2::4, 0::4] = raw[0::4, 0::2], raw[2::4, 0::2]
    raw_hollow[0::4, 3::4], raw_hollow[2::4, 1::4] = raw[0::4, 1::2], raw[2::4, 1::2]
    raw_hollow[1::4, 2::4], raw_hollow[3::4, 0::4] = raw[1::4, 0::2], raw[3::4, 0::2]
    raw_hollow[1::4, 3::4], raw_hollow[3::4, 1::4] = raw[1::4, 1::2], raw[3::4, 1::2]
    comp_kernal = np.array([[0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0]], dtype=np.float32) * 0.25
    raw_comp = np.zeros_like(raw_hollow)
    cv2.filter2D(raw_hollow, -1, comp_kernal, raw_comp, anchor= (-1, -1), borderType=cv2.BORDER_ISOLATED)
    raw_comp = raw_comp + raw_hollow;
    raw_comp = raw_comp.astype(np.uint16)
    return raw_comp

# ===============================================================
# Tianmouc Bayer 去马赛克
# ===============================================================
def exp_bayer_to_rgb_conv(bayer_image):
    '''
    # Define convolution kernels for each color channel
    # These kernels are designed to average the surrounding pixels
    # Kernel for Red and Blue channels (they are at the corners of the Bayer pattern)
    lyh testing        
    '''
    bayer_image = bayer_image.astype(np.float32)
    # Masks for each color in the BGGR pattern
    mask_b = np.zeros_like(bayer_image, dtype=np.float32)
    mask_g = np.zeros_like(bayer_image, dtype=np.float32)
    mask_r = np.zeros_like(bayer_image, dtype=np.float32)
    mask_b[0::2, 0::2] = 1  # Blue pixels
    mask_g[0::2, 1::2] = 1  # Green pixels on blue rows
    mask_g[1::2, 0::2] = 1  # Green pixels on red rows
    mask_r[1::2, 1::2] = 1  # Red pixels
    kernel_rb = np.array([[1, 12/5, 3, 12/5, 1],
                          [12/5, 6,    36/5, 6, 12/5],
                          [3, 36/5, 8, 36/5, 3],
                          [12/5, 6,    36/5, 6, 12/5],
                          [1, 12/5, 3, 12/5, 1]]) / 24
    # Kernel for Green channel (it's in the middle of the Bayer pattern)
    kernel_g = np.array([[0, 1, 0],
                         [1, 4, 1],
                         [0, 1, 0]]) / 4.0
    # Extract the height and width of the Bayer image
    height, width = bayer_image.shape
    # Initialize empty channels for R, G, B
    red_channel = np.zeros((height, width), dtype=np.float32)
    green_channel = np.zeros((height, width), dtype=np.float32)
    blue_channel = np.zeros((height, width), dtype=np.float32)
    # Apply convolution to simulate demosaicing
    # Note: We are using the 'same' border mode to ensure the output image has the same size
    blue_channel = cv2.filter2D(bayer_image * mask_b, -1, kernel_rb)
    green_channel = cv2.filter2D(bayer_image * mask_g, -1, kernel_g)
    red_channel = cv2.filter2D(bayer_image * mask_r, -1, kernel_rb)
    # Stack the R, G, B channels to form an RGB image
    rgb_image = cv2.merge((red_channel, green_channel, blue_channel))

    return rgb_image.astype(np.float32)


def demosaicing_npy(bayer=None, bayer_pattern='bggr', level=0 ,bitdepth=8):
    """
    Call this function to load raw bayer image
    :param bayer: input bayer image
    :param level: demosaicing level. 0: bilinear linear; 1: gradient
    :param bayer_type: bayer_type: : 0--RGrRGr...GbBGbB, 1--GrRGrR...BGbBGb...
    Author: Taoyi Wang
    """
    assert bayer is not None
    dtype = bayer.dtype
    max_v = (2 ** bitdepth) #- 1#bayer.max()
    bayer_cal = bayer.astype(np.float32)
    # 1st step: standard bilinear demosaicing process on bayer-patterned image
    conv_p = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32) * 1 / 4  # plus shaped interpolation
    conv_c = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float32) * 1 / 4  # cross shaped interpolation
    conv_ud = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float32) * 1 / 2  # up and down interpolation
    conv_lr = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=np.float32) * 1 / 2  # left and right interpolation
    p = convolve2d(bayer_cal, conv_p, boundary='symm', mode='same')
    c = convolve2d(bayer_cal, conv_c, boundary='symm', mode='same')
    ud = convolve2d(bayer_cal, conv_ud, boundary='symm', mode='same')
    lr = convolve2d(bayer_cal, conv_lr, boundary='symm', mode='same')
    # calculate gradient
    # center gradient v1 for plus shaped interpolation
    conv_grad_c1 = np.array([[0,    0,  -1, 0,  0],
                                [0,    0,  0,  0,  0],
                                [-1,   0,  4,  0,  -1],
                                [0,    0,  0,  0,  0],
                                [0,    0,  -1, 0,  0]]) * 1 / 8
    # center gradient v2 for cross shaped interpolation
    conv_grad_c2 = np.array([[0,    0,  -3/2,   0,  0],
                                [0,    0,  0,      0,  0],
                                [-3/2, 0,  6,      0,  -3/2],
                                [0,    0,  0,      0,  0],
                                [0,    0, -3/2,    0,  0]]) * 1 / 8
    # horizontal gradient for left and right interpolation
    conv_grad_h = np.array([[0, 0, 1 / 2, 0, 0],
                            [0, -1, 0, -1, 0],
                            [-1, 0, 5, 0, -1],
                            [0, -1, 0, -1, 0],
                            [0, 0, 1 / 2, 0, 0]]) * 1 / 8
    # vertical gradient for up and down interpolation
    conv_grad_v = conv_grad_h.T
    grad_c1 = convolve2d(bayer_cal, conv_grad_c1, boundary='symm', mode='same')
    grad_c2 = convolve2d(bayer_cal, conv_grad_c2, boundary='symm', mode='same')
    grad_h = convolve2d(bayer_cal, conv_grad_h, boundary='symm', mode='same')
    grad_v = convolve2d(bayer_cal, conv_grad_v, boundary='symm', mode='same')

    red = np.zeros_like(bayer_cal)
    ''' red[0::2, 0::2] = bayer_cal[0::2, 0::2]
    red[0::2, 1::2] = lr[0::2, 1::2]
    red[1::2, 0::2] = ud[1::2, 0::2]
    red[1::2, 1::2] = c[1::2, 1::2]'''

    green = np.zeros_like(bayer_cal)
    '''green[0::2, 0::2] = p[0::2, 0::2]
    green[0::2, 1::2] = bayer_cal[0::2, 1::2]
    green[1::2, 0::2] = bayer_cal[1::2, 0::2]
    green[1::2, 1::2] = p[1::2, 1::2]'''

    blue = np.zeros_like(bayer_cal)
    '''blue[0::2, 0::2] = c[0::2, 0::2]
    blue[0::2, 1::2] = ud[0::2, 1::2]
    blue[1::2, 0::2] = lr[1::2, 0::2]
    blue[1::2, 1::2] = bayer_cal[1::2, 1::2]'''

    if bayer_pattern == 'rggb':
        red[0::2, 0::2] = bayer[0::2, 0::2]
        red[0::2, 1::2] = lr[0::2, 1::2]
        red[1::2, 0::2] = ud[1::2, 0::2]
        red[1::2, 1::2] = c[1::2, 1::2]

        green[0::2, 0::2] = p[0::2, 0::2]
        green[0::2, 1::2] = bayer[0::2, 1::2]
        green[1::2, 0::2] = bayer[1::2, 0::2]
        green[1::2, 1::2] = p[1::2, 1::2]

        blue[0::2, 0::2] = c[0::2, 0::2]
        blue[0::2, 1::2] = ud[0::2, 1::2]
        blue[1::2, 0::2] = lr[1::2, 0::2]
        blue[1::2, 1::2] = bayer[1::2, 1::2]
        # add gradient compensation
        red[0::2, 1::2] += grad_h[0::2, 1::2]
        red[1::2, 0::2] += grad_v[1::2, 0::2]
        red[1::2, 1::2] += grad_c2[1::2, 1::2]

        green[0::2, 0::2] += grad_c1[0::2, 0::2]
        green[1::2, 1::2] += grad_c1[1::2, 1::2]

        blue[0::2, 0::2] += grad_c2[0::2, 0::2]
        blue[0::2, 1::2] += grad_v[0::2, 1::2]
        blue[1::2, 0::2] += grad_h[1::2, 0::2]
    elif bayer_pattern == 'grbg':
        red[0::2, 0::2] = lr[0::2, 0::2]
        red[0::2, 1::2] = bayer[0::2, 1::2]
        red[1::2, 0::2] = c[1::2, 0::2]
        red[1::2, 1::2] = ud[1::2, 1::2]
        
        green[0::2, 0::2] = bayer[0::2, 0::2]
        green[0::2, 1::2] = p[0::2, 1::2]
        green[1::2, 0::2] = p[1::2, 0::2]
        green[1::2, 1::2] = bayer[1::2, 1::2]
        
        blue[0::2, 0::2] = ud[0::2, 0::2]
        blue[0::2, 1::2] = c[0::2, 1::2]
        blue[1::2, 0::2] = bayer[1::2, 0::2]   
        blue[1::2, 1::2] = lr[1::2, 1::2]
        # add gradient compensation
        red[0::2, 0::2] += grad_h[0::2, 0::2]
        red[1::2, 1::2] += grad_v[1::2, 1::2]
        red[1::2, 0::2] += grad_c2[1::2, 0::2]
        
        green[0::2, 1::2] += grad_c1[0::2, 1::2]
        green[1::2, 0::2] += grad_c1[1::2, 0::2]
        
        blue[0::2, 0::2] += grad_v[0::2, 0::2]
        blue[0::2, 1::2] += grad_c2[0::2, 1::2]
        blue[1::2, 1::2] += grad_h[1::2, 1::2]
    elif bayer_pattern == 'bggr':
        blue[0::2, 0::2] = bayer[0::2, 0::2]
        blue[0::2, 1::2] = lr[0::2, 1::2]
        blue[1::2, 0::2] = ud[1::2, 0::2]
        blue[1::2, 1::2] = c[1::2, 1::2]

        green[0::2, 0::2] = p[0::2, 0::2]
        green[0::2, 1::2] = bayer[0::2, 1::2]
        green[1::2, 0::2] = bayer[1::2, 0::2]
        green[1::2, 1::2] = p[1::2, 1::2]

        red[0::2, 0::2] = c[0::2, 0::2]
        red[0::2, 1::2] = ud[0::2, 1::2]
        red[1::2, 0::2] = lr[1::2, 0::2]
        red[1::2, 1::2] = bayer[1::2, 1::2]
        # add gradient compensation
        blue[0::2, 1::2] += grad_h[0::2, 1::2]
        blue[1::2, 0::2] += grad_v[1::2, 0::2]
        blue[1::2, 1::2] += grad_c2[1::2, 1::2]

        green[0::2, 0::2] += grad_c1[0::2, 0::2]
        green[1::2, 1::2] += grad_c1[1::2, 1::2]

        red[0::2, 0::2] += grad_c2[0::2, 0::2]
        red[0::2, 1::2] += grad_v[0::2, 1::2]
        red[1::2, 0::2] += grad_h[1::2, 0::2]
    rgb = np.stack([red, green, blue], -1)
    rgb = np.clip(rgb, 0, max_v)
    return rgb


# ===============================================================
# 可视化差分数据
# bg_color:white/black
# ===============================================================
def vizDiff(diff,thresh=0,bg_color='white'):

    if bg_color == 'white':
        return vizDiff_WBG(diff,thresh=thresh)
    if bg_color == 'black':
        return vizDiff_BBG(diff,thresh=thresh)
    else:
        print('not implemented,bg_color:white/black')
        return None
    return rgb_diff
    
# ===============================================================
# 可视化差分数据(白底)
# ===============================================================
def vizDiff_WBG(diff,thresh=0):
    rgb_diff = 0
    w = h = 0
    if len(diff.shape)==2:
        w,h = diff.shape
    else:
        diff = diff[...,0]
        w,h = diff.shape
        
    rgb_diff = torch.ones([3,w,h]) * 255
    diff[abs(diff)<thresh] = 0
    rgb_diff[0,...][diff>0] = 0
    rgb_diff[1,...][diff>0] = diff[diff>0]
    rgb_diff[2,...][diff>0] = diff[diff>0]
    rgb_diff[0,...][diff<0] = -diff[diff<0]
    rgb_diff[1,...][diff<0] = 0
    rgb_diff[2,...][diff<0] = -diff[diff<0]
    return rgb_diff


# ===============================================================
# 可视化差分数据(黑底)
# ===============================================================
def vizDiff_BBG(diff,thresh=0):
    rgb_diff = 0
    w = h = 0
    if len(diff.shape)==2:
        w,h = diff.shape
    else:
        diff = diff[...,0]
        w,h = diff.shape
        
    rgb_diff = torch.zeros([3,w,h])
    diff[abs(diff)<thresh] = 0
    rgb_diff[1,...][diff>0] = diff[diff>0]
    rgb_diff[2,...][diff<0] = -diff[diff<0]
    
    return rgb_diff

