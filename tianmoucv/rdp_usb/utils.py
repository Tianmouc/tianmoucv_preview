from collections import defaultdict
import numpy as np
import os
import struct
import cv2,sys
import os
import numpy as np
from tianmoucv.rdp_usb import rod_decoder_py as rdc

def create_tmdat_folder_structure(target_path):

    save_path_cone = os.path.join(target_path, "cone")
    save_path_rod = os.path.join(target_path, "rod")
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

    return cone_path,rod_path

def covert_to_tmdat(cone_list, rod_list , target_path, 
                    mode = 0, # 1515, 3030, 3300, 10000
                    rod_adcprec = 8,
                    if_output=True):
    '''
    @ sample: 存储数据的字典
    @ if_output：False如果只需要计算带宽
    '''
    
    # 打印结果
    cone_height, cone_width = 320, 320
    rod_height, rod_width = 160, 160
    ctimestamp = 1
    c_interval = 330
    rtimestamp = 1
    cfcnt = 0
    rfcnt = 0
    
    ### You must set the tm fps!
    assert mode in [0,1]
    assert rod_adcprec in [2,4,8]

    tmc_fps = -1
    r_interval = -1
    
    if rod_adcprec == 8:
        if mode == 0:
            tmc_fps = 757
            r_interval = 132 
        else:
            tmc_fps = 1515
            r_interval = 66

    if rod_adcprec == 4:
        if mode == 0:
            tmc_fps = 1515
            r_interval = 66
        else:
            tmc_fps = 3030
            r_interval = 33

    if rod_adcprec == 2:
        if mode == 0:
            tmc_fps = 3300
            r_interval = 30
        else:
            tmc_fps = 10000
            r_interval = 10

    cone_path,rod_path = create_tmdat_folder_structure(target_path)

    cone_total_size = 0
    rod_total_size = 0
    cone_max_size = 0
    rod_max_size = 0

    # 检查所有数据的合法性
    # TODO：检查数据值域是否越界
    for rod in rod_list:
        assert rod.shape[0] == rod_height and rod.shape[1] == rod_width
    for cone_raw in cone_list:
        assert cone_raw.shape[0] == cone_height and cone_raw.shape[1] == cone_width
    # 检查帧率比例
    print('target number:', len(cone_list), len(cone_list) )

    # 开始转换
    for cone_raw in cone_list:
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

        size_in_bytes = encoded_raw.nbytes
        cone_total_size += size_in_bytes

        if size_in_bytes > cone_max_size:
            cone_max_size = size_in_bytes

        if if_output:
            with open(cone_path, 'ab') as f:
                f.write(encoded_raw.tobytes())
        ctimestamp += 3300
        cfcnt += 1

        print('Finished:',cfcnt,' cone /',len(cone_list),' clips')


    for rod_all in rod_list:

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

        size_in_bytes_rod = 0
        if rod_adcprec in [4,8]:
            size_in_bytes_rod = len(encoded_pkt) * 3 
        else:
            size_in_bytes_rod = len(encoded_pkt) * 4 
        rod_total_size += size_in_bytes_rod

        if size_in_bytes_rod > rod_max_size:
            rod_max_size = size_in_bytes_rod
        
        # print(header)
        hex_array = [hex(element) for element in encoded_pkt[0:24]]
        # print(hex_array)
        if if_output:
            with open(rod_path, 'ab') as f:
                f.write(encoded_pkt.tobytes())
        rtimestamp += r_interval
        rfcnt += 1

        if rfcnt % 25 == 0:
            print('Finished:', rfcnt, ' rod /',len(cone_list),' clips')

        
    output_bandwidth_meta = {}
    output_bandwidth_meta['cone_total_size'] = cone_total_size
    output_bandwidth_meta['rod_total_size'] = rod_total_size
    output_bandwidth_meta['cone_avg_band_width'] = cone_total_size / cfcnt * 30.3
    output_bandwidth_meta['cone_ideal_band_width'] = cone_total_size / cfcnt * 30.3/16*10
    output_bandwidth_meta['rod_avg_band_width'] = rod_total_size / rfcnt * tmc_fps
    output_bandwidth_meta['rod_max_band_width'] = rod_max_size * tmc_fps

    print(f"Cone总大小: {output_bandwidth_meta['cone_total_size']/1024/1024} MB")
    print(f"Rod 总大小: {output_bandwidth_meta['rod_total_size']/1024/1024} MB")
    print(f"Cone平均带宽: {output_bandwidth_meta['cone_avg_band_width']/1024/1024} MB/s")
    print(f"Cone理论带宽: {output_bandwidth_meta['cone_ideal_band_width']/1024/1024} MB/s")
    print(f"Rod 平均带宽: {output_bandwidth_meta['rod_avg_band_width']/1024/1024} MB/s")
    print(f"Rod 最大带宽: {output_bandwidth_meta['rod_max_band_width']/1024/1024} MB/s")
    return output_bandwidth_meta