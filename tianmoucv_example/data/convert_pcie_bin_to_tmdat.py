import numpy as np
import os
import struct
import cv2,sys
import torch

from tianmoucv.rdp_usb.rod_decode_pybind_usb import rod_decoder_py as rdc
import time

def check_and_create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"文件夹 {path} 已创建。")
    else:
        print(f"文件夹 {path} 已存在。")


train='/data/lyh/tianmoucData/tianmoucReconDataset/train/'
train_aim = '/data/lyh/tianmoucData/tianmoucReconDataset_usb/train/'
dirlist = os.listdir(train)
traindata = [train + e for e in dirlist]
train_aim_data = [train_aim + e for e in dirlist]

val='/data/lyh/tianmoucData/tianmoucReconDataset/test/'
val_aim = '/data/lyh/tianmoucData/tianmoucReconDataset_usb/test/'
dirlist = os.listdir(val)
valdata = [val + e for e in dirlist]
val_aim_data = [val_aim + e for e in dirlist]
key_list = []

convert_dict = dict([])

print('---------------------------------------------------')
for i in range(len(traindata)):
    sampleset = traindata[i]
    aimsampleset = train_aim_data[i]
    print('---->',sampleset,'有：',len(os.listdir(sampleset)),'个样本')
    for e in os.listdir(sampleset):
        key = sampleset + '/' + e
        value = aimsampleset + '/' + e
        convert_dict[key] = value
        
for i in range(len(valdata)):
    sampleset = valdata[i]
    aimsampleset = val_aim_data[i]
    print('---->',sampleset,'有：',len(os.listdir(sampleset)),'个样本')
    for e in os.listdir(sampleset):
        key = sampleset + '/' + e
        value = aimsampleset + '/' + e
        convert_dict[key] = value

for key in convert_dict:
    print(key,convert_dict[key])
    path = convert_dict[key]
    check_and_create_folder(path)
    check_and_create_folder(path + '/cone')
    check_and_create_folder(path + '/rod')
    
for key in convert_dict:
    dataset_top = key
    save_path_cone = convert_dict[key] + '/cone/cone_compact.tmdat'
    save_path_rod = convert_dict[key] + '/rod/rod_compact.tmdat'
    cone_eff_size = 102400 + 16; # fixed size of cone
    
    rdc.cone_pcie2usb_conv(dataset_top, cone_eff_size, save_path_cone)
    
    img_per_file = 2
    one_frm_size = 0x9e00
    size = one_frm_size * img_per_file
    rdc.rod_pcie2usb_conv(dataset_top, img_per_file, size,  one_frm_size, save_path_rod)
    print('Finished:',key)
    # rod_compact_pcie2usb(dataset_top,  img_per_file, size,  one_frm_size, save_file_path);