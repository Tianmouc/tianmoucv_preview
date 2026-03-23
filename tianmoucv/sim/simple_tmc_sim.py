import torch
import cv2
import os, sys
import numpy as np
import random
import math
import json
from tqdm import tqdm
import torchvision
from tianmoucv.isp import SD2XY

### Load parameters
file_path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(file_path,'simpleSim_params.json'), 'r') as f:
        sim_parama = json.load(f)

### parameters 
adc_bit_prec = sim_parama.get('adc_bit_prec', 8)
digital_type = torch.int8 if adc_bit_prec <= 8 else torch.int16 if adc_bit_prec <= 16 else torch.int32 if adc_bit_prec <= 24 else torch.float32
cop_cone_skip = 25 # the interval of RGB generation 
threshold = 0  # 3.001 / 255.0

def tdiff_split(td_,cdim = 0):
    td_pos = td_.clone()
    td_pos[td_pos<0] = 0
    td_neg = td_.clone()
    td_neg[td_neg>0] = 0
    td = torch.stack([td_pos,td_neg],dim=cdim)
    return td
    
def shift_rotate_tensor(tensor, shift_dx,shift_dy, rotate_degrees):

    transformed_tensor = torchvision.transforms.functional.affine(tensor, 
                                                                  translate=(shift_dx,shift_dy), 
                                                                  angle=rotate_degrees, 
                                                                  scale=1, 
                                                                  shear=0)
    return transformed_tensor
    
def push_to_fifo(tensor, x):
    """push to fifo, depth = self defined

    Args:
        tensor (torch.tensor): fifo tensor
        x (torch.tensor): new data

    Returns:
       torch.tensor : new fifo tensor
    """
    return torch.cat((tensor[1:], x))

def visualize_diff(diff_out, vis_gain, color='rg'):
    """diff data visualization

    Args:
        diff_out (_type_): _description_
        vis_gain (_type_): _description_
        color (str, optional): _description_. Defaults to 'rg'.

    Returns:
        _type_: _description_
    """
    height, width = diff_out.shape
   
    if color == 'rg' or color == 'rgwb':
        diff = (diff_out.astype(np.int32) * vis_gain).clip(min=-255, max=255)
        diff_vis = np.zeros((height, width, 3), dtype=np.uint8)
        diff_pos = diff * (diff > 0)
        diff_neg = -diff * (diff < 0)    
        diff_vis[..., 2] = diff_pos.astype(np.uint8)
        diff_vis[..., 1] = diff_neg.astype(np.uint8)

        return diff_vis

    elif color == 'gray':
        diff_vis = np.zeros((height, width), dtype=np.uint8)
        diff = (diff_out.astype(np.int32) * vis_gain).clip(min=-127, max=127)
        diff = diff + 127
        #diff = diff 
        diff_vis = diff.astype(np.uint8)
        return diff_vis
 

def diff_quant(diff, th):
    max_digi_num = 2 ** (adc_bit_prec- 1)
    lin_lsb = 1.0 / max_digi_num
    diff_th = (diff - th) * (diff > th) + (diff + th) * (diff < -th)
    diff_quantized = diff_th / lin_lsb
    diff_quantized = diff_quantized.clip(min=-max_digi_num, max=max_digi_num)
    diff_quantized = diff_quantized.ceil() * (diff > 0) + diff_quantized.floor() * (diff < 0)
    diff_quantized = diff_quantized.char() if digital_type == torch.int8 \
        else diff_quantized.short() if digital_type == torch.int16 \
        else diff_quantized.int() if digital_type == torch.int32 \
        else diff_quantized

    return diff_quantized

def diff_response(pix_v_out, threshold, sim_cnt, xy=False):
    '''
    :param pix_v_out, original input, please convert to float: 3D tensor, [2, H, W]
    :param threshold: dict, {'td': float, 'sd': float}
    :param sim_cnt: int, simulation counter, the number of frames simulated
    :return: td_quantized, sdl_quant, sdr_quant
    '''
    assert pix_v_out.shape[0] >= 2
    td_th = threshold['td']
    temp_diff = pix_v_out[1, :, :] - pix_v_out[0, :, :]
   # get  quantized TD
    td_quantized = diff_quant(temp_diff, td_th)
    
    ### SD
    sd_th = threshold['sd']
    sd_cal = pix_v_out[-1, :, :]
    sdl = torch.zeros_like(sd_cal)
    sdr = torch.zeros_like(sd_cal)
    sdl_proc =  sd_cal[1:, :]- sd_cal[:-1, :]
    sdl_proc[1::2, :] = -sdl_proc[1::2, :]
    sdl[:-1, :] = sdl_proc
    #sdr_cal = sd_cal[1:, :-1] - sd_cal[-1:, :1]
    interleved_sd = torch.zeros(size=(sd_cal.shape[0], sd_cal.shape[1]-1), dtype=sd_cal.dtype)
    interleved_sd[0::2] = sd_cal[0::2, 1:]
    interleved_sd[1::2] = sd_cal[1::2, :-1]
    sdr_proc = interleved_sd[1:, :] - interleved_sd[:-1, :]
    sdr_proc[1::2, :] = -sdr_proc[1::2, :]
    sdr[:-1, :-1] = sdr_proc
   # get quantized SD left and right
    sdl_quant = diff_quant(sdl, sd_th)
    sdr_quant = diff_quant(sdr, sd_th)
    
    if xy:
        sdl_quant = sdl_quant[:,::2]
        sdr_quant = sdr_quant[:,::2]
        sd = torch.stack([sdl_quant,sdr_quant],dim=0).float()
        sdx,sdy = SD2XY(sd)
        return td_quantized, sdx,sdy
    else:
        return td_quantized, sdl_quant, sdr_quant

def run_sim_singleimg(img,sensor_width= 640, sensor_height= 320, xy=True,
                      # 传感器噪声参数用于噪声增强
                      sensor_fixed_noise_prob=sim_parama.get('sensor_fixed_noise_prob', 0.0),
                      sensor_random_noise_prob=sim_parama.get('sensor_random_noise_prob', 0.0),
                      sensor_fixed_noise_mean_ch0=sim_parama.get('sensor_fixed_noise_mean_ch0', 0.2),
                      sensor_fixed_noise_std_ch0=sim_parama.get('sensor_fixed_noise_std_ch0', 0.5 / 128.0),
                      sensor_fixed_noise_mean_ch12=sim_parama.get('sensor_fixed_noise_mean_ch12', 0.0),
                      sensor_fixed_noise_std_ch12=sim_parama.get('sensor_fixed_noise_std_ch12', 0.4 / 128.0),
                      sensor_random_noise_std=sim_parama.get('sensor_random_noise_std', 1.0 / 128.0),
                      sensor_poisson_lambda=sim_parama.get('sensor_poisson_lambda', 4),
                      # 灰度输入扰动参数
                      gray_weight_jitter=sim_parama.get('gray_weight_jitter', 0.0),
                      gray_gain_min=sim_parama.get('gray_gain_min', 0.78),
                      gray_gain_max=sim_parama.get('gray_gain_max', 0.88),
                      # 仿真参数
                      sim_threshold_range=tuple(sim_parama.get('sim_threshold_range', [0.0, 0.0]))
                      ):

    # 输入： h,w,c的numpy矩阵
    # 输出： c,h,w的tensor 和 h，w的tensor
    
    # assum you have a png, 8bit rgb dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 采样仿真阈值
    sim_th = random.uniform(sim_threshold_range[0], sim_threshold_range[1]) if sim_threshold_range[1] > sim_threshold_range[0] else threshold
    
    # 2. 采样灰度输入扰动参数
    gray_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray_gain = 1.0
    if gray_weight_jitter > 0:
        delta = np.random.uniform(-gray_weight_jitter, gray_weight_jitter, size=3).astype(np.float32)
        gray_weights = np.clip(gray_weights + delta, 1e-6, None)
        gray_weights = gray_weights / np.sum(gray_weights)
        gray_gain = float(np.random.uniform(gray_gain_min, gray_gain_max))

    # 3. 采样传感器固定噪声图
    rod_height = sensor_height
    rod_width = sensor_width
    fixed_noise_map = None
    if sensor_fixed_noise_prob > 0 and random.random() < sensor_fixed_noise_prob:
        fixed_noise_map = np.zeros((3, rod_height, rod_width), dtype=np.float32)
        if sensor_fixed_noise_std_ch0 > 0:
            fixed_noise_map[0] = np.random.normal(loc=sensor_fixed_noise_mean_ch0, 
                                                scale=sensor_fixed_noise_std_ch0, 
                                                size=(rod_height, rod_width)).astype(np.float32)
        if sensor_fixed_noise_std_ch12 > 0:
            shared_map = np.random.normal(loc=sensor_fixed_noise_mean_ch12, 
                                        scale=sensor_fixed_noise_std_ch12, 
                                        size=(rod_height, rod_width)).astype(np.float32)
            fixed_noise_map[1] = shared_map
            fixed_noise_map[2] = shared_map
        fixed_noise_map = torch.from_numpy(fixed_noise_map).to(device)

    sim_cnt = 0
    img = cv2.resize(img, (sensor_width ,sensor_height))

    # 灰度扰动
    if gray_weight_jitter > 0:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_gray = np.tensordot(img_rgb, gray_weights, axes=([-1], [0]))
        img_gray = (img_gray * gray_gain * 255.0).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR) # 把扰动后的灰度图转回BGR以保持后面逻辑一致
    
    img_tensor = torch.FloatTensor(img).to(device)
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.permute(2,0,1)

    # 生成一个在1和10之间的随机整数
    random_x = int(random.uniform(-32,32))
    random_y = int(random.uniform(-32,32))
    random_degrees = random.uniform(-4,4)

    img_tensor_pre = shift_rotate_tensor(img_tensor.unsqueeze(0), random_x, random_y, random_degrees)
    img_tensor_pre = img_tensor_pre.squeeze(0)

    intensity_pre = torch.mean(img_tensor_pre,dim=0,keepdim=True)
    intensity = torch.mean(img_tensor,dim=0,keepdim=True)

    zero = torch.zeros([1,sensor_height, sensor_width]).to(device)
    rod_v_buf = torch.cat([zero,intensity_pre],dim=0)
 
    _, sdl0, sdr0 = diff_response(rod_v_buf, {'td': sim_th, 'sd': sim_th}, sim_cnt, xy=xy)

    rod_v_buf = torch.cat([intensity_pre,intensity],dim=0)
 
    td, sdl1, sdr1 = diff_response(rod_v_buf, {'td': sim_th, 'sd': sim_th}, sim_cnt, xy=xy)

    # 传感器噪声增强 (固定噪声 + 随机噪声)
    def apply_noise(td_in, sdl_in, sdr_in):
        td_in, sdl_in, sdr_in = td_in.float(), sdl_in.float(), sdr_in.float()
        if fixed_noise_map is not None:
            td_in = td_in + (fixed_noise_map[0] * 128.0)
            sdl_in = sdl_in + (fixed_noise_map[1] * 128.0)
            sdr_in = sdr_in + (fixed_noise_map[2] * 128.0)
        
        if sensor_random_noise_prob > 0 and random.random() < sensor_random_noise_prob:
            mu = max(float(sensor_poisson_lambda), 1e-6)
            std_target = max(float(sensor_random_noise_std), 0.0)
            if std_target > 0:
                scale = (std_target / np.sqrt(2.0 * mu)) * 128.0
                p_noise = torch.poisson(torch.full((3, rod_height, rod_width), mu, device=device))
                p_noise2 = torch.poisson(torch.full((3, rod_height, rod_width), mu, device=device))
                skellam_noise = (p_noise - p_noise2) * scale
                td_in = td_in + skellam_noise[0]
                sdl_in = sdl_in + skellam_noise[1]
                sdr_in = sdr_in + skellam_noise[2]
        return td_in, sdl_in, sdr_in

    _, sdl0, sdr0 = apply_noise(torch.zeros_like(td), sdl0, sdr0)
    td, sdl1, sdr1 = apply_noise(td, sdl1, sdr1)

    cdim = 0
    td =  tdiff_split(td,cdim=cdim)
    sd0 = torch.stack([sdl0,sdr0],dim=cdim)
    sd1 = torch.stack([sdl1,sdr1],dim=cdim)

    return img_tensor_pre.cpu(), img_tensor.cpu(), td.cpu(), sd0.cpu(), sd1.cpu()



