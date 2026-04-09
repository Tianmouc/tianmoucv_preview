import numpy as np
from torch.utils import data
import os
from PIL import Image
import torch
import random
import cv2
from tianmoucv.data import TianmoucDataReader

import glob
import math

def read_exposure_time(cone_folder):
    """在cone下的txt文件中读取曝光时间"""
    txt_list = glob.glob(os.path.join(cone_folder, '*.txt'))
    # txt文件缺失 ->
    if not txt_list:
        raise FileNotFoundError(f"未在 {cone_folder} 中找到任何 txt 文件")
    info_txt = txt_list[0]
    with open(info_txt, 'r') as f:
        for line in f:
            if 'Exp Time:' in line:
                try:
                    exp_time = int(line.split('Exp Time:')[1].split('us')[0].strip())
                    return exp_time
                except:
                    continue
    raise ValueError(f"未能从 {info_txt} 中解析出 Exp Time")


def calc_blur_length(exp_time):
    """计算模糊帧数"""
    return math.ceil(exp_time / 1320)

class TianmouDeblurDataset(data.Dataset):
    
    def __init__(self, TMC_path, return_voxel=True, return_frame=True, return_gt_frame=False):
        
        # self.tianmouc_path = opt['TMC_path']  # 下一级必须是cone rod
        self.tianmouc_path = TMC_path
    
        self.return_voxel = return_voxel
        self.return_frame = return_frame
        self.return_gt_frame = return_gt_frame


        # ==========================================================
        # 自动读取 cone/ 下的曝光时间 → 自动算 blur_len
        # ==========================================================
        cone_folder = os.path.join(self.tianmouc_path, "cone")

        exp_time = read_exposure_time(cone_folder)
        self.blur_len = calc_blur_length(exp_time)

        print(f"blur_len = {self.blur_len}（Exp Time = {exp_time}us）")
        # ==========================================================



        self.tmc_dataset = TianmoucDataReader(path=self.tianmouc_path, N=1, camera_idx=0)
        self.total_samples = len(self.tmc_dataset)
        

    def __getitem__(self, index):

        if index < 0 or index >= self.total_samples:
            raise IndexError("Index out of range.")
        
        sample = self.tmc_dataset[index]

        aop_idx = 0
        if self.blur_len == 12:  #取中心靠左sd
            aop_idx = 6
        elif self.blur_len == 11:
            aop_idx = 6
        elif self.blur_len == 10:
            aop_idx = 5
        elif self.blur_len == 9:
            aop_idx = 5
        elif self.blur_len == 8:
            aop_idx = 4
        elif self.blur_len == 7:
            aop_idx = 4
        elif self.blur_len == 6:
            aop_idx = 3
        elif self.blur_len == 5:
            aop_idx = 3
        
        item = {}

        ################# F0 #################
        if self.return_frame:
            F0 = sample["F0"].numpy()
            F0 = F0.astype(np.float32)  # (H, W, 3)  # 0-1 float32
            F0 = F0[:, ::-1, :]  # 左右翻转

            F0 = F0.transpose((2, 0, 1))  # (3, H, W)
            item['frame'] = torch.from_numpy(F0.copy()).float()  #不允许有负stride，故copy

        ################# TSD #################
        if self.return_voxel:
            tsd_all = sample["tsdiff"].numpy()  # (3, aop_len, H, W)  aop_len = 26 (?

            if self.blur_len == 12:
                t_indices = np.arange(2, 13)
            elif self.blur_len == 11:
                t_indices = np.arange(2, 12)
            elif self.blur_len == 10:
                t_indices = np.arange(2, 11)
            elif self.blur_len == 9:
                t_indices = np.arange(2, 10)
            elif self.blur_len == 8:
                t_indices = np.arange(2, 9)
            elif self.blur_len == 7:
                t_indices = np.arange(2, 8)
            elif self.blur_len == 6:
                t_indices = np.arange(2, 7)
            elif self.blur_len == 5:
                t_indices = np.arange(2, 6)
                

            td_array = tsd_all[0, t_indices, :, :][:, :, ::-1].astype(np.float32)  # (12, H, W)  不要0，要1-12 [-1, 1] 左右翻转    
             
            item['td_voxel'] = torch.from_numpy(td_array.copy()).float()

            sdl = tsd_all[1, aop_idx, :, :][:, ::-1] # (H, W)  左右翻转
            sdr = tsd_all[2, aop_idx, :, :][:, ::-1] # (H, W)  左右翻转
            sd_array = np.stack([sdl, sdr], axis=0).astype(np.float32)  # (2, H, W)
            item['sd_voxel'] = torch.from_numpy(sd_array.copy()).float()

        item['seq'] = index

        return item

    def __len__(self):
        return self.total_samples


    @staticmethod
    def collate_fn(data):
        """自定义 collate_fn 以适应批处理。"""
        collated = {}
        for key in data[0].keys():
            collated[key] = [item[key] for item in data]
        return collated


