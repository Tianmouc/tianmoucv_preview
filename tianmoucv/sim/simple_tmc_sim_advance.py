# Author：Taoyi
import torch
import torch.nn.functional as F
import cv2
import os, sys
import numpy as np
import random
from tqdm import tqdm
import re
import json
import torchvision
from tianmoucv.isp import SD2XY, upsample_cross_conv
### Load parameters

file_path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(file_path,'simpleSim_params.json'), 'r') as f:
        sim_parama = json.load(f)

adc_bit_prec = sim_parama['adc_bit_prec']
dark_fpn_stat = sim_parama['dark_fpn_stat']
if adc_bit_prec == 8:
    for key in dark_fpn_stat:
        # 确保值是列表或NumPy数组，然后除以128
        dark_fpn_stat[key] = dark_fpn_stat[key] / 128
else:
    raise ValueError("Only support 8bit ADC now!")

digital_type = torch.int8 if adc_bit_prec <= 8 else torch.int16 if adc_bit_prec <= 16 else torch.int32 if adc_bit_prec <= 24 else torch.float32

def sort_filenames(filename):
    # 从文件名中提取数字部分
    pattern = r'[0-9]+'
    matches = re.findall(pattern, filename)
    if matches:
        return int(matches[0])
    else:
        return -1


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

def diff_response(pix_v_out, threshold, sim_cnt, device, fpn=None, xy=False):
    '''
    :param pix_v_out, original input, please convert to float: 3D tensor, [2, H, W]
    :param threshold: dict, {'td': float, 'sd': float}
    :param sim_cnt: int, simulation counter, the number of frames simulated
    :param device: torch.device, cuda or cpu
    :param fpn: dict, optional fixed pattern noise
    :param xy: bool, whether to return SDX and SDY
    :return: td_quantized, sdl_quant, sdr_quant (or sdx, sdy if xy=True)
    '''
    assert pix_v_out.shape[0] >= 2
    td_th = threshold['td']
    if sim_cnt == 0 :
        x_s, y_s = int(random.uniform(1,5)),int(random.uniform(1,5))
        response_org = pix_v_out[1, :, :].cpu().numpy()
        matShift = np.float32([[1,0,x_s],[0,1,y_s]])
        response_shift = cv2.warpAffine(response_org, matShift,(response_org.shape[1],response_org.shape[0]))
        torch_tensor = torch.from_numpy(response_shift)

        response_shift = torch_tensor.to(device)#torch.FloatTensor(response_shift, device=device)
        temp_diff = response_shift - response_org
    else:
        temp_diff = pix_v_out[1, :, :] - pix_v_out[0, :, :]
    
    # ADVANCED FPN for TD
    if fpn is not None:
        if sim_cnt % 2 == 0:
            td_fpn = fpn['td_even']
        else:
            td_fpn = fpn['td_odd']
        temp_diff += td_fpn
    
    # get quantized TD
    td_quantized = diff_quant(temp_diff, td_th)
    
    ### SD Calculation
    sd_th = threshold['sd']
    sd_cal = pix_v_out[-1, :, :]
    H, W = sd_cal.shape
    sdl = torch.zeros_like(sd_cal)
    sdr = torch.zeros_like(sd_cal)

    # Correct SD formulas: Minuends are ALWAYS odd rows
    # Due to the ROD layout (Even rows shifted left by 1 pixel):
    # - SDL calculation (vertical in ROD grid) is physically Center - Left
    # - SDR calculation (diagonal in ROD grid) is physically Center - Right
    # 
    if H > 1 and W > 1:
        # SDL Calculation (Physically Center - Left)
        # Even rows
        sdl[0:H-1:2, 0:W] = sd_cal[1:H:2, 0:W] - sd_cal[0:H-1:2, 0:W]
        # Odd rows
        sdl[1:H-1:2, 0:W] = sd_cal[1:H-1:2, 0:W] - sd_cal[2:H:2, 0:W]
        # SDR Calculation (Physically Center - Right)
        # Even rows
        sdr[0:H-1:2, 0:W-1] = sd_cal[1:H:2, 0:W-1] - sd_cal[0:H-1:2, 1:W]
        # Odd rows
        sdr[1:H-1:2, 0:W-1] = sd_cal[1:H-1:2, 0:W-1] - sd_cal[2:H:2, 1:W]
    
    ### ADVANCED NOISE SIMULATION FPN for SD
    if fpn is not None:
        if sim_cnt % 2 == 0:
            sdl_fpn = fpn['sdl_even']
            sdr_fpn = fpn['sdr_even']
        else:
            sdl_fpn = fpn['sdl_odd']
            sdr_fpn = fpn['sdr_odd']
        sdl += sdl_fpn
        sdr += sdr_fpn

    if xy:
        sd = torch.stack([sdl, sdr], dim=0).float()
        sdx, sdy = SD2XY(sd)
        sdx_quant = diff_quant(sdx, sd_th)
        sdy_quant = diff_quant(sdy, sd_th)
        return td_quantized, sdx_quant, sdy_quant
    else:
        # get quantized SD left and right
        sdl_quant = diff_quant(sdl, sd_th)
        sdr_quant = diff_quant(sdr, sd_th)
        return td_quantized, sdl_quant, sdr_quant

def get_fpn_from_stat(rod_height, rod_width, device):
    """
    Generate Fixed Pattern Noise (FPN) from dark_fpn_stat.
    """
    fixed_td_noise_odd = torch.normal(mean=dark_fpn_stat['td_odd_mean'], 
                                      std=dark_fpn_stat['td_odd_std'], 
                                      size=(rod_height, rod_width), device=device)
    fixed_sdl_noise_odd = torch.normal(mean=dark_fpn_stat['sdl_odd_mean'], 
                                       std=dark_fpn_stat['sdl_odd_std'], 
                                       size=(rod_height, rod_width), device=device)
    fixed_sdr_noise_odd = torch.normal(mean=dark_fpn_stat['sdr_odd_mean'],
                                       std=dark_fpn_stat['sdr_odd_std'], 
                                       size=(rod_height, rod_width), device=device)
    fixed_td_noise_even = torch.normal(mean=dark_fpn_stat['td_even_mean'],
                                       std=dark_fpn_stat['td_even_std'], 
                                       size=(rod_height, rod_width), device=device)
    fixed_sdl_noise_even = torch.normal(mean=dark_fpn_stat['sdl_even_mean'], 
                                        std=dark_fpn_stat['sdl_even_std'], 
                                        size=(rod_height, rod_width), device=device)
    fixed_sdr_noise_even = torch.normal(mean=dark_fpn_stat['sdr_even_mean'],
                                        std=dark_fpn_stat['sdr_even_std'],
                                        size=(rod_height, rod_width), device=device)
    fpn = {
        'td_odd': fixed_td_noise_odd,
        'sdl_odd': fixed_sdl_noise_odd,
        'sdr_odd': fixed_sdr_noise_odd,
        'td_even': fixed_td_noise_even,
        'sdl_even': fixed_sdl_noise_even,
        'sdr_even': fixed_sdr_noise_even
    }
    return fpn

def sample_sensor_fixed_noise(sensor_height, sensor_width, device, 
                               sensor_fixed_noise_prob, 
                               sensor_fixed_noise_mean_ch0, sensor_fixed_noise_std_ch0,
                               sensor_fixed_noise_mean_ch12, sensor_fixed_noise_std_ch12):
    rod_height = sensor_height // 2
    rod_width = sensor_width // 4
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
    return fixed_noise_map

def apply_gray_jitter(img, gray_weight_jitter, gray_gain_min, gray_gain_max):
    gray_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray_gain = 1.0
    if gray_weight_jitter > 0:
        delta = np.random.uniform(-gray_weight_jitter, gray_weight_jitter, size=3).astype(np.float32)
        gray_weights = np.clip(gray_weights + delta, 1e-6, None)
        gray_weights = gray_weights / np.sum(gray_weights)
        gray_gain = float(np.random.uniform(gray_gain_min, gray_gain_max))
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_gray = np.tensordot(img_rgb, gray_weights, axes=([-1], [0]))
        img_gray = (img_gray * gray_gain * 255.0).clip(0, 255).astype(np.uint8)
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def get_rod_img(img_gray_tensor, rod_height, rod_width, device):
    if img_gray_tensor.dim() == 3: # [C, H, W]
        img_gray_tensor = torch.mean(img_gray_tensor, dim=0) # [H, W]
    
    etron_img_rod = torch.zeros(size=(rod_height, rod_width), device=device)
    etron_img_bin = (img_gray_tensor[0::2, 0::2] + img_gray_tensor[1::2, 0::2] + 
                     img_gray_tensor[0::2, 1::2] + img_gray_tensor[1::2, 1::2]) / 4
    
    etron_img_rod[0::2, :] = etron_img_bin[0::2, 0::2]
    etron_img_rod[1::2, :] = etron_img_bin[1::2, 1::2]
    return etron_img_rod

def apply_sensor_noise(td, sdl, sdr, fixed_noise_map, 
                       sensor_random_noise_prob, sensor_poisson_lambda, sensor_random_noise_std,
                       rod_height, rod_width, device):
    td, sdl, sdr = td.float(), sdl.float(), sdr.float()
    if fixed_noise_map is not None:
        td = td + (fixed_noise_map[0] * 128.0)
        sdl = sdl + (fixed_noise_map[1] * 128.0)
        sdr = sdr + (fixed_noise_map[2] * 128.0)
    
    if sensor_random_noise_prob > 0 and random.random() < sensor_random_noise_prob:
        mu = max(float(sensor_poisson_lambda), 1e-6)
        std_target = max(float(sensor_random_noise_std), 0.0)
        if std_target > 0:
            scale = (std_target / np.sqrt(2.0 * mu)) * 128.0
            p1 = torch.poisson(torch.full((3, rod_height, rod_width), mu, device=device))
            p2 = torch.poisson(torch.full((3, rod_height, rod_width), mu, device=device))
            skellam_noise = (p1 - p2) * scale
            td = td + skellam_noise[0]
            sdl = sdl + skellam_noise[1]
            sdr = sdr + skellam_noise[2]
    return td, sdl, sdr

def run_sim(datapath,sensor_width, sensor_height,  device, display = False, save = False, 
            save_path = os.path.join(os.environ.get('HOME'), "temp"),
            interp=False,
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
            sim_threshold_range=tuple(sim_parama.get('sim_threshold_range', [0.005, 0.02]))
            ):

    print('读取:',datapath,'下的所有图像数据，按图像名称排序')
    print('存储到:',save_path,'')

    # 1. 采样仿真阈值
    sim_th = random.uniform(sim_threshold_range[0], sim_threshold_range[1])
    
    # 2. 采样传感器固定噪声图
    fixed_noise_map = sample_sensor_fixed_noise(sensor_height, sensor_width, device, 
                                                sensor_fixed_noise_prob, 
                                                sensor_fixed_noise_mean_ch0, sensor_fixed_noise_std_ch0,
                                                sensor_fixed_noise_mean_ch12, sensor_fixed_noise_std_ch12)

    # assum you have a png, 8bit rgb dataset
    # if use some other dataset, please write your own code
    flist = sorted(os.listdir(datapath), key=sort_filenames)
    cop_cone_skip = 10 # the interval of RGB generation 
    sim_cnt = 0
    rod_height = sensor_height // 2
    rod_width = sensor_width // 4
    rod_v_buf = torch.zeros(size=(2, rod_height, rod_width), dtype=torch.float32, device=device)
    ### ADVANCED NOISE SIM, Fixed pattern noise odd and ven
    fpn = get_fpn_from_stat(rod_height, rod_width, device)
    # fixed_td_noise_odd_np = fpn['td_odd'].cpu().numpy() * 128
    if save:
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        rgb_save_path = os.path.join(save_path, 'rgb')
        tsdiff_save_path = os.path.join(save_path, 'tsdiff')
        gt_with_noise_save_path = os.path.join(save_path, 'noisy_gt')
        viz_path = os.path.join(save_path, 'viz')
        if os.path.exists(rgb_save_path) == False:
            os.makedirs(rgb_save_path)
        if os.path.exists(tsdiff_save_path) == False:
            os.makedirs(tsdiff_save_path)
        if os.path.exists(gt_with_noise_save_path) == False:
            os.makedirs(gt_with_noise_save_path)
        if os.path.exists(viz_path) == False:
            os.makedirs(viz_path)
        # Here save Fixed pattern noise
        all_fpn = np.stack([fpn['td_odd'].cpu().numpy(),
                            fpn['sdl_odd'].cpu().numpy(), fpn['sdr_odd'].cpu().numpy(),
                            fpn['td_even'].cpu().numpy(), fpn['sdl_even'].cpu().numpy(), 
                            fpn['sdr_even'].cpu().numpy()], axis=-1)
        np.save(os.path.join(save_path, "fpn.npy"), all_fpn)

    for img_file in tqdm(flist):
        if img_file.endswith(".png") or img_file.endswith(".jpg"): 
            # You can add more endings, liek .bmp, .tiff, etc.
            
            img = cv2.imread(os.path.join(datapath, img_file))
            img = cv2.resize(img, (sensor_width ,sensor_height))
            # 灰度扰动
            img_gray = apply_gray_jitter(img, gray_weight_jitter, gray_gain_min, gray_gain_max)
            img_gray_tensor = torch.FloatTensor(img_gray).to(device)
            if sim_cnt % cop_cone_skip == 0:
                rgb = torch.ShortTensor(img).to(device)
            ### ADVANCED NOISE SIM,
            # First, possion noise
            img_gray_tensor = torch.poisson(img_gray_tensor)

            img_diff_sim = img_gray_tensor / 255.0
            # if you use 10bit, or higher precision img, please divide by the max value!
            
            etron_img_rod = get_rod_img(img_diff_sim, rod_height, rod_width, device)
            
            ### ADVANCED SIM for NOISE: Add Norm read noise!
            etron_img_rod = etron_img_rod + torch.normal(mean=0, std=0.008, size=(rod_height, rod_width), device=device)
            # etron_img_rod 
            
            img_diff_sim = etron_img_rod.unsqueeze(0)
            rod_v_buf = push_to_fifo(rod_v_buf, img_diff_sim)
            td, sdl, sdr = diff_response(rod_v_buf, {'td': sim_th, 'sd': sim_th}, sim_cnt, device, fpn=fpn)

            # 传感器噪声增强 (固定噪声 + 随机噪声)
            td, sdl, sdr = apply_sensor_noise(td, sdl, sdr, fixed_noise_map, 
                                              sensor_random_noise_prob, sensor_poisson_lambda, sensor_random_noise_std,
                                              rod_height, rod_width, device)
            
            if interp:
                tsdiff = torch.stack([td, sdl, sdr], dim=0) # [3, H, W]
                # upsample_cross_conv expects [C, T, H, W]
                tsdiff_expand = upsample_cross_conv(tsdiff.unsqueeze(1)).squeeze(1) # [3, H, W*2]
                # bilinear upsample to full size
                tsdiff_full = F.interpolate(tsdiff_expand.unsqueeze(0), size=(sensor_height, sensor_width), 
                                            mode='bilinear', align_corners=False).squeeze(0)
                td, sdl, sdr = tsdiff_full[0], tsdiff_full[1], tsdiff_full[2]
                # Update sizes for visualization if needed
                cur_rod_h, cur_rod_w = sensor_height, sensor_width // 2
            else:
                cur_rod_h, cur_rod_w = rod_height, rod_width

            if display or save:
                rgb_np = rgb.cpu().numpy().astype(np.uint8)
                td_np = td.cpu().numpy()
                sdl_np = sdl.cpu().numpy()
                sdr_np = sdr.cpu().numpy()
                
                tsdiff_raw = np.stack([td_np, sdl_np, sdr_np], axis=-1)

                td_np_viz = visualize_diff(td_np, 8, color='rg')
                sdl_np_viz = visualize_diff(sdl_np, 8, color='rg')
                sdr_np_viz = visualize_diff(sdr_np, 8, color='rg')
                red_line = np.zeros((cur_rod_h, 2, 3), dtype=np.uint8)
                diff_concat = np.concatenate((td_np_viz, red_line, sdl_np_viz, red_line,
                                sdr_np_viz),
                               axis=1)
                
                ### for visualization
                sd_merge_viz = cv2.addWeighted(sdl_np_viz, 0.5, sdr_np_viz, 0.5, 0)
                tsd_viz = np.concatenate(
                    (cv2.resize(td_np_viz, (cur_rod_w * 2, cur_rod_h)),
                    cv2.resize(sd_merge_viz, (cur_rod_w * 2, cur_rod_h))),
                            axis=1)
                rgb_tsd_viz = np.concatenate((rgb_np, tsd_viz), axis=0)
                
                if display:
                    cv2.imshow("all", rgb_tsd_viz)
                    cv2.waitKey(1)
                    
                if save:
                    if sim_cnt % cop_cone_skip == 0:
                        cv2.imwrite(os.path.join(rgb_save_path, img_file), rgb_np)
                    cv2.imwrite(os.path.join(viz_path, f"sim_viz_{sim_cnt}.png"), rgb_tsd_viz)
                    np.save(os.path.join(tsdiff_save_path, f"tsdiff_{sim_cnt}.npy"), tsdiff_raw)
                    np.save(os.path.join(gt_with_noise_save_path, f"noisyGT_{sim_cnt}.npy"), etron_img_rod.cpu().numpy())
            # sim counter
            sim_cnt += 1

def run_sim_singleimg(img_target=None, img_ref=None ,sensor_width= 640, sensor_height= 320, xy=False, interp=False,     # assum you have a png, 8bit rgb dataset
                      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
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
    # 1. 采样仿真阈值
    sim_th = random.uniform(sim_threshold_range[0], sim_threshold_range[1]) if sim_threshold_range[1] > sim_threshold_range[0] else sim_parama.get('threshold', 0.0)
    
    # 2. 采样传感器固定噪声图
    fixed_noise_map = sample_sensor_fixed_noise(sensor_height, sensor_width, device, 
                                                sensor_fixed_noise_prob, 
                                                sensor_fixed_noise_mean_ch0, sensor_fixed_noise_std_ch0,
                                                sensor_fixed_noise_mean_ch12, sensor_fixed_noise_std_ch12)

    sim_cnt = 0
    rod_height = sensor_height // 2
    rod_width = sensor_width // 4
    fpn = get_fpn_from_stat(rod_height, rod_width, device)
    
    img_target_tensor = None
    img_ref_tensor = None

    if img_target is not None:
        img_target = cv2.resize(img_target, (sensor_width ,sensor_height))
        img_gray_target = apply_gray_jitter(img_target, gray_weight_jitter, gray_gain_min, gray_gain_max)
        img_target = cv2.cvtColor(img_gray_target, cv2.COLOR_GRAY2BGR)
        img_target_tensor = torch.FloatTensor(img_target).to(device) / 255.0
        img_target_tensor = img_target_tensor.permute(2,0,1)

    if img_ref is not None:
        img_ref = cv2.resize(img_ref, (sensor_width ,sensor_height))
        img_gray_ref = apply_gray_jitter(img_ref, gray_weight_jitter, gray_gain_min, gray_gain_max)
        img_ref = cv2.cvtColor(img_gray_ref, cv2.COLOR_GRAY2BGR)
        img_ref_tensor = torch.FloatTensor(img_ref).to(device) / 255.0
        img_ref_tensor = img_ref_tensor.permute(2,0,1)

    # Generate missing frame if needed
    if img_target_tensor is None and img_ref_tensor is not None:
        # Generate target (current) from ref (previous)
        random_x = int(random.uniform(-32,32))
        random_y = int(random.uniform(-32,32))
        random_degrees = random.uniform(-4,4)
        img_target_tensor = shift_rotate_tensor(img_ref_tensor.unsqueeze(0), random_x, random_y, random_degrees).squeeze(0)
    elif img_ref_tensor is None and img_target_tensor is not None:
        # Generate ref (previous) from target (current)
        random_x = int(random.uniform(-32,32))
        random_y = int(random.uniform(-32,32))
        random_degrees = random.uniform(-4,4)
        img_ref_tensor = shift_rotate_tensor(img_target_tensor.unsqueeze(0), random_x, random_y, random_degrees).squeeze(0)
    elif img_target_tensor is None and img_ref_tensor is None:
        raise ValueError("Either img_target or img_ref must be provided to run_sim_singleimg")

    # ROD采样逻辑
    intensity_ref = get_rod_img(img_ref_tensor, rod_height, rod_width, device).unsqueeze(0)
    intensity_target = get_rod_img(img_target_tensor, rod_height, rod_width, device).unsqueeze(0)

    # First frame SD only (using ref-ref to make TD=0)
    rod_v_buf = torch.cat([intensity_ref, intensity_ref], dim=0)
    _, sdl0, sdr0 = diff_response(rod_v_buf, {'td': sim_th, 'sd': sim_th}, 0, device, fpn=fpn, xy=xy)

    # Second frame TD+SD (using ref-target)
    rod_v_buf2 = torch.cat([intensity_ref, intensity_target], dim=0)
    td, sdl1, sdr1 = diff_response(rod_v_buf2, {'td': sim_th, 'sd': sim_th}, 1, device, fpn=fpn, xy=xy)

    # 传感器噪声增强 (固定噪声 + 随机噪声)
    _, sdl0, sdr0 = apply_sensor_noise(torch.zeros_like(td), sdl0, sdr0, fixed_noise_map, 
                                       sensor_random_noise_prob, sensor_poisson_lambda, sensor_random_noise_std,
                                       rod_height, rod_width, device)
    td, sdl1, sdr1 = apply_sensor_noise(td, sdl1, sdr1, fixed_noise_map, 
                                        sensor_random_noise_prob, sensor_poisson_lambda, sensor_random_noise_std,
                                        rod_height, rod_width, device)
    if interp:
        if not xy:
            # First frame SD (no TD)
            tsdiff0 = torch.stack([torch.zeros_like(td), sdl0, sdr0], dim=0)
            tsdiff0_expand = upsample_cross_conv(tsdiff0.unsqueeze(1)).squeeze(1)
            tsdiff0_full = F.interpolate(tsdiff0_expand.unsqueeze(0), size=(sensor_height, sensor_width), 
                                         mode='bilinear', align_corners=False).squeeze(0)
            sdl0, sdr0 = tsdiff0_full[1], tsdiff0_full[2]
            
            # Second frame TD + SD
            tsdiff1 = torch.stack([td, sdl1, sdr1], dim=0)
            tsdiff1_expand = upsample_cross_conv(tsdiff1.unsqueeze(1)).squeeze(1)
            tsdiff1_full = F.interpolate(tsdiff1_expand.unsqueeze(0), size=(sensor_height, sensor_width), 
                                         mode='bilinear', align_corners=False).squeeze(0)
            td, sdl1, sdr1 = tsdiff1_full[0], tsdiff1_full[1], tsdiff1_full[2]
        else:
            # xy=True, just directly upsample as requested
            def up_direct(t):
                if t.dim() == 2: t = t.unsqueeze(0).unsqueeze(0)
                elif t.dim() == 3: t = t.unsqueeze(0)
                return F.interpolate(t, size=(sensor_height, sensor_width), 
                                     mode='bilinear', align_corners=False).squeeze()

            # td still uses upsample_cross_conv based on tianmoucData reference
            td_expand = upsample_cross_conv(td.view(1, 1, rod_height, rod_width)).squeeze()
            td = F.interpolate(td_expand.view(1, 1, rod_height, rod_width*2), size=(sensor_height, sensor_width), 
                               mode='bilinear', align_corners=False).squeeze()
            sdl0 = up_direct(sdl0)
            sdr0 = up_direct(sdr0)
            sdl1 = up_direct(sdl1)
            sdr1 = up_direct(sdr1)

    cdim = 0
    sd0 = torch.stack([sdl0, sdr0], dim=cdim)
    sd1 = torch.stack([sdl1, sdr1], dim=cdim)

    return img_ref_tensor.cpu(), img_target_tensor.cpu(), td.cpu(), sd0.cpu(), sd1.cpu()

