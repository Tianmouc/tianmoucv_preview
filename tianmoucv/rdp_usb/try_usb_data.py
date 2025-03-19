import numpy as np
import os
import struct
import cv2,sys
import torch

# import rod_decoder_py as rdc
from tianmoucv.rdp_usb import rod_decoder_py as rdc
import time

dataset_path = "/data/taoyi/dataset/Tianmouc/noise/1240ms_test/opt_dark_gain1"
rod_data_path = os.path.join(os.path.join(dataset_path, "rod"), "tianmouc_raw_data_866033.tmdat")
cone_data_path = os.path.join(os.path.join(dataset_path, "cone"), "tianmouc_raw_data_862505.tmdat")

rod_img_per_file = 1
rod_width = 160
rod_height = 160

frm_start_pos = 0

rod_ts_list, rod_cnt_list = [],[]
t0 = time.time()
r_ptr_list = rdc.construct_frm_list(rod_data_path,rod_ts_list,rod_cnt_list)
#print(rod_ts_list)
t1 = time.time()
# print("Rod construct_frm_list time: ", t1 - t0)
t0 = time.time()
c_ts_list, c_cnt_list = [],[]
c_ptr_list = rdc.construct_frm_list(cone_data_path, c_ts_list, c_cnt_list)
t1 = time.time()
# print("Cone construct_frm_list time: ", t1 - t0)
#print(ret)

temp_diff_np = np.zeros((rod_img_per_file, rod_width * rod_height), dtype=np.int8)
spat_diff_left_np = np.zeros((rod_img_per_file, rod_width * rod_height), dtype=np.int8)
spat_diff_right_np = np.zeros((rod_img_per_file, rod_width * rod_height), dtype=np.int8)
pkt_size_np = np.zeros([rod_img_per_file], dtype=np.int32)
pkt_size_td = np.zeros([rod_img_per_file], dtype=np.int32)
pkt_size_sd = np.zeros([rod_img_per_file], dtype=np.int32) 
rod_img_timestamp_np = np.zeros([rod_img_per_file], dtype=np.uint64)
rod_fcnt_np = np.zeros([rod_img_per_file], dtype=np.uint32)
rod_adcprec_np = np.zeros([rod_img_per_file], dtype=np.int32)
ret_code = rdc.get_one_rod_fullinfo(rod_data_path, frm_start_pos,
                                    temp_diff_np, spat_diff_left_np, spat_diff_right_np,
                                    pkt_size_np, pkt_size_td, pkt_size_sd,
                                    rod_img_timestamp_np, rod_fcnt_np, rod_adcprec_np,
                                    rod_height, rod_width)

# 如果你想把生成的天眸图，改写为原始的天眸数据，由于可能已经丢失原始帧头信息，请保证重写的帧头的时间戳和帧计数是正确的，否则会导致数据解析错误
rtimestamp = rod_img_timestamp_np.item()
rcnt = rod_fcnt_np.item()
header_elements = [
    0xED800008,
    0xED000000 + ((rtimestamp >> 48) & 0xFFFFF),
    0xED000000 + ((rtimestamp >> 24) & 0xFFFFFF),
    0xED000000 + ((rtimestamp) & 0xFFFFF),
    0xED000000 + ((rcnt >> 24) & 0xFFFFF),
    0xED000000 + ((rcnt) & 0xFFFFF),
    0xED000000,
    0xED000000,
    0xED000000,
    0xED000000,
    0xED000000,
    0xED000000,
    0xED000000,
    0xED000000,
    0xED000000,
    0xED000000
]

# 创建 numpy 数组
header = np.array(header_elements, dtype=np.uint32)
old_pkt_first = 0x08000AFF # 随便写一个即可，理论上，不影响数据解析
# 用TD，SD的numpy矩阵，生成的header和old_pkt_first，生成天眸数据格式（单帧）
encoded_pkt = rdc.rod_encoder_np(temp_diff_np, spat_diff_left_np, spat_diff_right_np, header, old_pkt_first, rod_height, rod_width)
frame_size = len(encoded_pkt)
encoded_pkt[6] = 0xED000000 + frame_size
# header_elements[6] = 0xED000000 + frame_size
# 接下来就可以把encoded_pkt写入.tmdat文件了
# 自行组帧。。注意
# print(spat_diff_left_np)
print(pkt_size_np, pkt_size_td, pkt_size_sd, rod_img_timestamp_np, rod_fcnt_np, rod_adcprec_np)
cone_width = 320
cone_height = 320
cone_raw = np.zeros((1, cone_width * cone_height), dtype=np.int16)


c_img_timestamp_np = np.zeros([1], dtype=np.uint64)
c_fcnt_np = np.zeros([1], dtype=np.int32)
c_adcprec_np = np.zeros([1], dtype=np.int32)
ret_code = rdc.get_one_cone_fullinfo(cone_data_path, frm_start_pos,
                                    cone_raw,
                                    c_img_timestamp_np, c_fcnt_np, c_adcprec_np,
                                    cone_height, cone_width)



print(cone_raw)