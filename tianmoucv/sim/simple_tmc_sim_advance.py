# Author：Taoyi
import torch
import cv2
import os, sys
import numpy as np
import random
from tqdm import tqdm
import re
import json
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
        diff_vis[..., 0] = diff_pos.astype(np.uint8)
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

def diff_response(pix_v_out, fpn, threshold, sim_cnt, device):
    '''
    :param pix_v_out, original input, please convert to float: 3D tensor, [2, H, W]
    :param threshold: dict, {'td': float, 'sd': float}
    :param sim_cnt: int, simulation counter, the number of frames simulated
    :param device: torch.device, cuda or cpu
    :return: td_quantized, sdl_quant, sdr_quant
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
        temp_diff = response_shift - pix_v_out[1, :, :]
    else:
        temp_diff = pix_v_out[1, :, :] - pix_v_out[0, :, :]
    # ADVANCED FPN
    if sim_cnt % 2 == 0:
        td_fpn = fpn['td_even']
    else:
        td_fpn = fpn['td_odd']
    temp_diff += td_fpn
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
    ### ADVANCED NOISE SIMULATION  FPN
    if sim_cnt % 2 == 0:
        sdl_fpn = fpn['sdl_even']
        sdr_fpn = fpn['sdr_even']
    else:
        sdl_fpn = fpn['sdl_odd']
        sdr_fpn = fpn['sdr_odd']
    sdl += sdl_fpn
    sdr += sdr_fpn
   # get quantized SD left and right
    sdl_quant = diff_quant(sdl, sd_th)
    sdr_quant = diff_quant(sdr, sd_th)
    
    return td_quantized, sdl_quant, sdr_quant

def run_sim(datapath,sensor_width, sensor_height,  device, display = False, save = False, 
            save_path = os.path.join(os.environ.get('HOME'), "temp")):

    print('读取:',datapath,'下的所有图像数据，按图像名称排序')
    print('存储到:',save_path,'')
    
    # assum you have a png, 8bit rgb dataset
    # if use some other dataset, please write your own code
    flist = sorted(os.listdir(datapath), key=sort_filenames)
    cop_cone_skip = 10 # the interval of RGB generation 
    sim_cnt = 0
    rod_height = sensor_height // 2
    rod_width = sensor_width // 4
    rod_v_buf = torch.zeros(size=(2, rod_height, rod_width), dtype=torch.float32, device=device)
    ### ADVANCED NOISE SIM, Fixed pattern noise odd and ven
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
    # fixed_td_noise_odd_np = fixed_td_noise_odd.cpu().numpy() * 128
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
        all_fpn = np.stack([fixed_td_noise_odd.cpu().numpy(),
                            fixed_sdl_noise_odd.cpu().numpy(), fixed_sdr_noise_odd.cpu().numpy(),
                            fixed_td_noise_even.cpu().numpy(), fixed_sdl_noise_even.cpu().numpy(), 
                            fixed_sdr_noise_even.cpu().numpy()], axis=-1)
        np.save(os.path.join(save_path, "fpn.npy"), all_fpn)

    for img_file in tqdm(flist):
        if img_file.endswith(".png") or img_file.endswith(".jpg"): 
            # You can add more endings, liek .bmp, .tiff, etc.
            
            img = cv2.imread(os.path.join(datapath, img_file))
            img = cv2.resize(img, (sensor_width ,sensor_height))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray_tensor = torch.FloatTensor(img_gray).to(device)
            if sim_cnt % cop_cone_skip == 0:
                rgb = torch.ShortTensor(img).to(device)
            ### ADVANCED NOISE SIM,
            # First, possion noise
            img_gray_tensor = torch.poisson(img_gray_tensor)

            img_diff_sim = img_gray_tensor / 255.0
            # if you use 10bit, or higher precision img, please divide by the max value!
            
            etron_img_rod = torch.zeros(size=(rod_height,rod_width), device=device)
            etron_img_bin =  (img_diff_sim[0::2, 0::2] + 
                              img_diff_sim[1::2, 0::2] + 
                              img_diff_sim[0::2, 1::2] + 
                              img_diff_sim[1::2, 1::2]) / 4
            
            etron_img_rod[0::2, :] = etron_img_bin[0::2, 0::2]
            etron_img_rod[1::2, :] = etron_img_bin[1::2, 1::2]
            
            ### ADVANCED SIM for NOISE: Add Norm read noise!
            etron_img_rod = etron_img_rod + torch.normal(mean=0, std=0.008, size=(rod_height, rod_width), device=device)
            # etron_img_rod 
            
            img_diff_sim = etron_img_rod.unsqueeze(0)
            rod_v_buf = push_to_fifo(rod_v_buf, img_diff_sim)
            td, sdl, sdr = diff_response(rod_v_buf, fpn,{'td': 0.01, 'sd': 0.01}, sim_cnt, device)

            if display or save:
                rgb_np = rgb.cpu().numpy().astype(np.uint8)
                td_np = td.cpu().numpy()
                sdl_np = sdl.cpu().numpy()
                sdr_np = sdr.cpu().numpy()
                
                tsdiff_raw = np.stack([td_np, sdl_np, sdr_np], axis=-1)

                td_np_viz = visualize_diff(td_np, 8, color='rg')
                sdl_np_viz = visualize_diff(sdl_np, 8, color='rg')
                sdr_np_viz = visualize_diff(sdr_np, 8, color='rg')
                red_line = np.zeros((rod_height, 2, 3), dtype=np.uint8)
                diff_concat = np.concatenate((td_np_viz, red_line, sdl_np_viz, red_line,
                                sdr_np_viz),
                               axis=1)
                
                ### for visualization
                sd_merge_viz = cv2.addWeighted(sdl_np_viz, 0.5, sdr_np_viz, 0.5, 0)
                tsd_viz = np.concatenate(
                    (cv2.resize(td_np_viz, (rod_width * 2, rod_height)),
                    cv2.resize(sd_merge_viz, (rod_width * 2, rod_height))),
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

