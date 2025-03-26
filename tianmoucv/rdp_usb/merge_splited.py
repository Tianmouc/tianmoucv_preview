import re
from collections import defaultdict
import numpy as np
import os
import struct
import cv2,sys
# import rod_decoder_py as rdc ### if you use local rdc
from tianmoucv.rdp_usb import rod_decoder_py as rdc ### if you use tianmoucv rdc

# Dataset for read
dataset_top = "/data/taoyi/dataset/Tianmouc/metacam/yapeng/"
dataset_name = "test_metacam_encode"
# here you can set your dataset name
save_path_cone = os.path.join(dataset_top, dataset_name, "cone")
save_path_rod = os.path.join(dataset_top, dataset_name, "rod")
if not os.path.exists(save_path_cone):
    os.makedirs(save_path_cone)
if not os.path.exists(save_path_rod):
    os.makedirs(save_path_rod)
cone_fname = "cone_encoded.tmdat"
rod_fname = "rod_encoded.tmdat"

cone_path = os.path.join(save_path_cone, cone_fname)
rod_path = os.path.join(save_path_rod, rod_fname)
if os.path.exists(cone_path):
    os.remove(cone_path)
if os.path.exists(rod_path):
    os.remove(rod_path)
    
# 假设这是OCR或手动读取到的文件列表（从图片中提取）
file_list = os.listdir(os.path.join(dataset_top, dataset_name))

# 正则匹配
f0raw_pattern = re.compile(r'(\d{3})_00_F0raw\.npy')
tsdraw_pattern = re.compile(r'(\d{3})_(\d{2})_tsdraw\.npy')

# 结果字典
result = defaultdict(list)

# 先找出所有F0raw
f0raw_set = set()
for f in file_list:
    match = f0raw_pattern.match(f)
    if match:
        f0raw_set.add(match.group(1))

# 遍历tsdraw，归类到对应的外层编号
for f in file_list:
    match = tsdraw_pattern.match(f)
    if match:
        outer, inner = match.groups()
        if outer in f0raw_set:
            result[f"{outer}_00_F0raw.npy"].append(f)
### You must set the tm fps!
tmc_fps = 757 # 1515, 3030, 3300, 10000
rod_adcprec = 8 
if rod_adcprec == 8 and tmc_fps > 1515:
    print("[ERROR] Set wrong TMC fps {} and precision{} ".format(tmc_fps, rod_adcprec))
    exit(-1)
if tmc_fps == 757:
    r_interval = 132 
elif tmc_fps == 1515:
    r_interval = 66
elif tmc_fps == 3030:
    r_interval = 33
elif tmc_fps == 3300:
    r_interval = 30
elif tmc_fps == 10000:
    r_interval = 10
else:
    print("Set wrong TMC fps error")
    exit(-1)
# 打印结果
cone_height, cone_width = 320, 320
rod_height, rod_width = 160, 160
ctimestamp = 1
c_interval = 330
rtimestamp = 1
cfcnt = 1
rfcnt = 1

for k in sorted(result.keys()):
    print(f"{k}:")
    cone_raw = np.load(os.path.join(dataset_top, dataset_name, k))
    header_elements_cone = [
        0xFA800001,
        0xFA000000 + ((ctimestamp >> 48) & 0xFFFFF),
        0xFA000000 + ((ctimestamp >> 24) & 0xFFFFFF),
        0xFA000000 + ((ctimestamp) & 0xFFFFF),
        0xFA000000 + ((cfcnt >> 24) & 0xFFFFF),
        0xFA000000 + ((cfcnt) & 0xFFFFF),
        0x0000C810,
        0xFA000000,
        0xFA000000,
        0xFA000000,
        0xFA000000,
        0xFA000000,
        0xFA000000,
        0xFA000000,
        0xFA000000,
        0xFA000000
    ]

    # print(cone_raw)
    header = np.array(header_elements_cone, dtype=np.uint32)
    # 用TD，SD的numpy矩阵，生成的header和old_pkt_first，生成天眸数据格式（单帧）
    encoded_raw = rdc.cone_encoder_np(cone_raw, header, cone_height, cone_width)
    with open(cone_path, 'ab') as f:
        f.write(encoded_raw.tobytes())
    ctimestamp += 3300
    cfcnt += 1
    for v in sorted(result[k]):
        print(f"  {v}")
        rod_all = np.load(os.path.join(dataset_top, dataset_name, v))
        temp_diff_np = rod_all[ :, :, 0].astype(np.int8)
        spat_diff_left_np = rod_all[ :, :, 1].astype(np.int8)
        spat_diff_right_np = rod_all[ :, :, 2].astype(np.int8)
        header_elements = [
            0xED800000 + rod_adcprec,
            0xED000000 + ((rtimestamp >> 48) & 0xFFFFF),
            0xED000000 + ((rtimestamp >> 24) & 0xFFFFFF),
            0xED000000 + ((rtimestamp) & 0xFFFFF),
            0xED000000 + ((rfcnt >> 24) & 0xFFFFF),
            0xED000000 + ((rfcnt) & 0xFFFFF),
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
        encoded_pkt = rdc.rod_encoder_np(temp_diff_np, spat_diff_left_np, spat_diff_right_np, header, old_pkt_first, rod_height, rod_width)
        # print(header)
        hex_array = [hex(element) for element in encoded_pkt[0:24]]
        print(hex_array)
        with open(rod_path, 'ab') as f:
            f.write(encoded_pkt.tobytes())
        rtimestamp += r_interval
        rfcnt += 1