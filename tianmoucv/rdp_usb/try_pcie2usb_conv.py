import numpy as np
import os
import struct
import cv2,sys
import torch

import rod_decoder_py as rdc
import time

dataset_top = "/data/taoyi/dataset/Lyncam/2023_02_16_zy/flicker/120r_fast_1.5k2.6k_30Hz"


cone_eff_size = 102400 + 16; # fixed size of cone
save_path_cone = dataset_top +  "/cone_compact.tmdat"
save_path_rod = dataset_top +  "/rod_compact.tmdat"

rdc.cone_pcie2usb_conv(dataset_top, cone_eff_size, save_path_cone)

img_per_file = 2
one_frm_size = 0x9e00
size = one_frm_size * img_per_file
rdc.rod_pcie2usb_conv(dataset_top, img_per_file, size,  one_frm_size, save_path_rod)
# rod_compact_pcie2usb(dataset_top,  img_per_file, size,  one_frm_size, save_file_path);