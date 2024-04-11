#基础的一些isp操作与可视化函数，有些和算法的效果绑定
__author__ = 'Y. Lin'
__authorEmail__ = '532109881@qq.com'
import torch
import numpy as np
from scipy.signal import convolve2d
import cv2

##############################
#1. 图像基本处理
#2. 可视化
##############################
def default_rgb_isp(raw,gamma = 1.2, curve_factor = 0.02, saturation_factor = (32,24,48), denoising = True, raw_input = True):
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
        raw = lyncam_raw_comp(raw)

        #adjust gamma
        raw_gamma = (raw/1024.0)**(1/gamma)*1024.0

        #antialiasing
        #aaf = AAF(raw_gamma)
        #raw_aaf = aaf.execute()

        #demosacing
        image_demosaic = demosaicing_npy(raw_gamma, 'bggr', 1, 10)
    else:
        image_demosaic = (raw/1024.0)**(1/gamma)*1024.0

    #AWB
    image = white_balance(image_demosaic.copy(),HSB=1023)
    
    image = (image/4.0).astype(np.uint8)
     #denoising(a little bit slow,cost 0.8s, while others cost 0.05s)
    if denoising:
        image = cv2.fastNlMeansDenoising(image)
   
    #adjust_saturation
    image = adjust_saturation(image,saturation_factor)
    
    #adjust_curve
    image = adjust_curve(image,curve_factor)

    #norm to 0-1
    image = image.astype(np.float32)/256.0
    
    return image,image_demosaic/1024.0
            

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

# ===============================================================
# 白平衡调整——灰度世界假设
#:param img: cv2.imread读取的图片数据
#:return: 返回的白平衡结果图片数据
# ===============================================================
def white_balance(img,HSB=256):
    '''
     白平衡调整——灰度世界假设
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
    K = (B_ave + G_ave + R_ave) / 3
    Kb, Kg, Kr = K / (B_ave+1e-8), K / (G_ave+1e-8), K / (R_ave+1e-8)
    Ba = (B * Kb)
    Ga = (G * Kg)
    Ra = (R * Kr)
    Ba[Ba>HSB] = HSB
    Ga[Ga>HSB] = HSB
    Ra[Ra>HSB] = HSB
    img[:, :, 0] = Ba
    img[:, :, 1] = Ga
    img[:, :, 2] = Ra
    #print('wb:',np.max(img))
    return img

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



def adjust_saturation(image, saturation_factor = (48,48,64)):
    '''
     饱和度调整
    :param img: cv2.imread读取的图片数据
    :saturation_factor:增加饱和度，saturation_factor越大饱和度越大
    :return: 返回的饱和度结果图片数据
    '''

    target,threshold,max_value = saturation_factor

    # 将图像转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = hsv_image.astype(np.float32)
    # 调整饱和度（增加饱和度可以使用更大的值，减少饱和度可以使用更小的值）
    V = hsv_image[:,:,1]
    
    factor = target / (1e-8+np.mean(V))
    V = V * factor

    mask = V > threshold
    V[mask] = max_value + -np.exp(-(V[mask]-threshold)/((max_value-threshold))) * (max_value-threshold)

    hsv_image[:,:,1] = V
    
    #print('st:',np.max(hsv_image))
    hsv_image = hsv_image.astype(np.uint8)
    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return result_image



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
##Author: Taoyi Wang
def demosaicing_npy(bayer=None, bayer_pattern='bggr', level=0 ,bitdepth=8):
    """
    Call this function to load raw bayer image
    :param bayer: input bayer image
    :param level: demosaicing level. 0: bilinear linear; 1: gradient
    :param bayer_type: bayer_type: : 0--RGrRGr...GbBGbB, 1--GrRGrR...BGbBGb...
  
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
# SD坐标变换
# ===============================================================
#axis mapping
#mapping ruls: tianmouc pixel pattern
def fourdirection2xy(sd):
    #print("warning: 0711version, Ix may be wrong direction")
    if len(sd.shape)==4:
        #input: [b,2,w,h]
        #output: [b,2,w,h]
        Ixy = torch.zeros(sd.shape).to(sd.device)
        sdul = sd[:,0,0::2,...]
        sdll = sd[:,0,1::2,...]
        sdur = sd[:,1,0::2,...]
        sdlr = sd[:,1,1::2,...]
        Ixy[:,0, ::2,...] = Ixy[:,0,1::2,...] = ((sdul + sdll)/1.414 - (sdur + sdlr)/1.414)/2
        Ixy[:,0,1::2,...] = Ixy[:,1, ::2,...] = ((sdur - sdlr)/1.414 + (sdul - sdll)/1.414)/2
        return Ixy
    else:
        #input: [w,h,2]
        #output: [w,h],[w,h]
        if sd.shape[-1] == 2:
            Ix = torch.zeros(sd.shape[:2]).to(sd.device)
            Iy = torch.zeros(sd.shape[:2]).to(sd.device)
            sdul = sd[0::2,...,0]
            sdll = sd[1::2,...,0]
            sdur = sd[0::2,...,1]
            sdlr = sd[1::2,...,1]
            Ix[::2,...] = Ix[1::2,...]= ((sdul + sdll)/1.414 - (sdur + sdlr)/1.414)/2
            Iy[1::2,...]= Iy[::2,...] = ((sdur - sdlr)/1.414 + (sdul - sdll)/1.414)/2
            return Ix,Iy
        else:
            Ix = torch.zeros(sd.shape[1:]).to(sd.device)
            Iy = torch.zeros(sd.shape[1:]).to(sd.device)
            sdul = sd[0,0::2,...]
            sdll = sd[0,1::2,...]
            sdur = sd[1,0::2,...]
            sdlr = sd[1,1::2,...]
            Ix[::2,...] = Ix[1::2,...]= ((sdul + sdll)/1.414 - (sdur + sdlr)/1.414)/2
            Iy[1::2,...]= Iy[::2,...] = ((sdur - sdlr)/1.414 + (sdul - sdll)/1.414)/2
            return Ix,Iy

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
# 可视化光流
# ===============================================================
def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 0.1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img



def laplacian_blending_1c(Ix,Iy,gray,iteration=50):
    '''
    # 灰度重建-不直接调用
    # vectorized by Y. Lin
    # Function to apply Poisson blending to two images
    '''
    
    if gray is None:
        gray = torch.zeros_like(Ix)
    lap_blend = gray
    # Perform Poisson iteration
    for i in range(iteration):
        lap_blend_old = lap_blend.clone()
        # Update the Laplacian values at each pixel
        grad = 1/4 * (Ix[1:-1,2:] -  Iy[1:-1,1:-1] 
                    + Iy[2:,1:-1] -  Ix[1:-1,1:-1])
        lap_blend_old_tmp = 1/4 * (lap_blend_old[2:,1:-1] + lap_blend_old[0:-2,1:-1] 
                                 + lap_blend_old[1:-1,2:] + lap_blend_old[1:-1,0:-2])

        lap_blend[1:-1,1:-1] = lap_blend_old_tmp + grad
        # Check for convergence
        if torch.sum(torch.abs(lap_blend - lap_blend_old)) < 0.1:
            #print("converged")
            break
    # Return the blended image
    return lap_blend

def laplacian_blending_1c_batch(Ix,Iy,gray,iteration=50):
    '''
    # 灰度重建-支持batch的网络训练用接口
    # vectorized by Y. Lin
    # Function to apply Poisson blending to two images
    '''
    if gray is None:
        gray = torch.zeros_like(Ix).to(Ix.device)
    lap_blend = gray
    # Perform Poisson iteration
    for i in range(iteration):
        lap_blend_old = lap_blend.clone()
        # Update the Laplacian values at each pixel
        grad = 1/4 * (Ix[:,0,1:-1,2:] -  Iy[:,0,1:-1,1:-1] 
                    + Iy[:,0,2:,1:-1] -  Ix[:,0,1:-1,1:-1])
        lap_blend_old_tmp = 1/4 * (lap_blend_old[:,0,2:,1:-1] + lap_blend_old[:,0,0:-2,1:-1] 
                                 + lap_blend_old[:,0,1:-1,2:] + lap_blend_old[:,0,1:-1,0:-2])

        lap_blend[:,0,1:-1,1:-1] = lap_blend_old_tmp + grad
        # Check for convergence
        if torch.sum(torch.abs(lap_blend - lap_blend_old)) < 0.1:
            #print("converged")
            break
    # Return the blended image
    return lap_blend


#兼容旧接口
def poisson_blend(Ix,Iy,iteration=50):
    return laplacian_blending_1c(Ix,Iy,None,iteration=iteration)


def genMask(gray,th = 24, maxV=255, minV = 0):

    gap = maxV- minV
    mask_ts = ( (gray < (maxV-th)/gap) * (gray > (minV+th)/gap) ).float()
    mask_np = mask_ts.cpu().numpy()
    mask_np_b = (mask_np * gap).astype(np.uint8)
    kernel = np.ones((5,5),np.uint8) * gap
    kernel[0,4] = kernel[4,0] = kernel[4,4] = kernel[0,0] = 0
    mask_np_b = cv2.erode(mask_np_b,kernel,iterations = 2)
    mask_np_b = cv2.dilate(mask_np_b,kernel,iterations = 2)
    mask_np = (mask_np_b>0.5)
    return torch.Tensor(mask_np).to(gray.device).bool()

def laplacian_blending(Ix,Iy,srcimg=None, iteration=20, mask_rgb=False, mask_th = 24):
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
        result = laplacian_blending_1c(Ix,Iy,iteration=iteration)
    elif len(srcimg.shape)==2:
        img = srcimg.clone()
        result = laplacian_blending_1c(Ix,Iy,img,iteration=iteration)
    elif len(srcimg.shape)==3:
        img = srcimg.clone()
        for c in range(img.shape[-1]):
            target = img[...,c]
            img[...,c] = laplacian_blending_1c(Ix,Iy,target, iteration=iteration)
        result = img
    else:
        print('img shape:',srcimg.shape,' is illegal, [None],[H,W],[H,W,C] is supported')

    if mask_rgb and not result is None:
        result[mask] = srcimg[mask]
        
    return result

# ===============================================================
# 光流用的色轮
# ===============================================================
def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

# ===============================================================
# 光流uv to RGB
# ===============================================================
def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8
    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0
    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))
    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))
    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)
    img = compute_color(u, v)
    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0
    return np.uint8(img)


# ===============================================================
# 可视化差分数据
# ===============================================================
def vizDiff(diff,thresh=0):
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