import sys,os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ctypes import c_uint64
from tianmoucv.isp import default_rgb_isp

try:
    from tianmoucv.rdp_usb import rod_decoder_py as rdc
except:
    print("WARNING: no decoder found, try to compile it under ./rod_decoder_py")
    current_file_path = os.path.abspath(__file__)
    parent_folder_path = os.path.dirname(os.path.dirname(current_file_path))
    aim_path = os.path.join(parent_folder_path,'rdp_usb')
    os.chdir(aim_path)
    current_path = os.getcwd()
    print("Current Path:", current_path)
    subprocess.run(['sh', './compile_pybind.sh'])
    from tianmoucv.rdp_usb import rod_decoder_py as rdc
    print('compile decoder successfully')
    print('If you still get this message,please try:\n 1. run it in a python script (only once) \n 2. use source code install to see what happened')
  


def cone_tmdat_to_npy(condDataPath, idx=0, mode='RGB'):
    '''
    @mode: RGB or RAW
    '''
    cone_width = 320
    cone_height = 320
    cone_raw = np.zeros((1, cone_width * cone_height), dtype=np.int16)
    c_img_timestamp_np = np.zeros([1], dtype=np.uint64)
    c_fcnt_np = np.zeros([1], dtype=np.int32)
    c_adcprec_np = np.zeros([1], dtype=np.int32)
    
    coneTimeList = []
    conecntList = []
    coneAddrlist =rdc.construct_frm_list(condDataPath, coneTimeList, conecntList)
    
    coneAddrlist = [c_uint64(e).value for e in coneAddrlist]  
    coneTimeList = [c_uint64(e).value for e in coneTimeList]
    
    print('legal index range:',len(coneAddrlist))
    assert idx < len(coneAddrlist)
    
    ret_code = rdc.get_one_cone_fullinfo(condDataPath, coneAddrlist[idx],
                                                          cone_raw,
                                                          c_img_timestamp_np, 
                                                          c_fcnt_np, 
                                                          c_adcprec_np,
                                                          cone_height, 
                                                          cone_width)
    
    cone_raw = np.reshape(cone_raw.astype(np.float32),(cone_height,cone_width))
    cone_RGB,_ = default_rgb_isp(cone_raw,blc=0)
    cone_RGB = (cone_RGB * 255.0).astype(np.uint8)

    if mode == 'RAW':
        return cone_raw,coneTimeList[idx]
    else:
        return cone_RGB,coneTimeList[idx]


def rod_tmdat_to_npy(rodDataPath, idx=0):
    rod_width = 160
    rod_height = 160
    temp_diff_np = np.zeros((1, rod_width * rod_height), dtype=np.int8)
    spat_diff_left_np = np.zeros((1, rod_width * rod_height), dtype=np.int8)
    spat_diff_right_np = np.zeros((1, rod_width * rod_height), dtype=np.int8)
    pkt_size_np = np.zeros([1], dtype=np.int32)
    pkt_size_td = np.zeros([1], dtype=np.int32)
    pkt_size_sd = np.zeros([1], dtype=np.int32) 
    rod_img_timestamp_np = np.zeros([1], dtype=np.uint64)
    rod_fcnt_np = np.zeros([1], dtype=np.int32)
    rod_adcprec_np = np.zeros([1], dtype=np.int32)        

    rodTimeList = []
    rodcntList = []
                
    rodAddrlist =rdc.construct_frm_list(rodDataPath,rodTimeList,rodcntList)
                
            
    rodAddrlist = [c_uint64(e).value for e in rodAddrlist]
    rodTimeList = [c_uint64(e).value for e in rodTimeList]

    print('legal index range:',len(rodAddrlist))
    assert idx < len(rodAddrlist)
    
    ret_code = rdc.get_one_rod_fullinfo(rodDataPath, rodAddrlist[idx],
                                                temp_diff_np, 
                                                spat_diff_left_np, 
                                                spat_diff_right_np,
                                                pkt_size_np, 
                                                pkt_size_td, 
                                                pkt_size_sd,
                                                rod_img_timestamp_np, 
                                                rod_fcnt_np, 
                                                rod_adcprec_np,
                                                rod_height, 
                                                rod_width)

    tsd = np.zeros([3,rod_height,rod_width])
    tsd[0,:,:] = np.reshape(temp_diff_np.astype(np.float32),(rod_height,rod_width))
    tsd[1,:,:] = np.reshape(spat_diff_left_np.astype(np.float32),(rod_height,rod_width))
    tsd[2,:,:] = np.reshape(spat_diff_right_np.astype(np.float32),(rod_height,rod_width))
    
    return tsd,rodTimeList[idx]
