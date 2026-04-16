import numpy as np
import threading
import time
import argparse
import cv2
import torch
import torch.nn.functional as F
import copy

#用于色彩调整
from tianmoucv.isp import SD2XY,ACESToneMapping
#算法
from tianmoucv.proc.reconstruct import laplacian_blending

#start camera and read
from tianmoucv.camera.sdk_utils import TMstreamingDataReader
from tianmoucv.camera.sdk_utils import global_aop_queue,global_cop_queue,result_queue
from tianmoucv.proc.deblur.STDNet import TianmoucDeblurNet
from tianmoucv.proc.deblur.debluirDataBuilder import TianmouDeblurDataset
from tianmoucv.data import TianmoucDataReader

# Create the parser
parser = argparse.ArgumentParser(description='choose camera id')
# Add arguments to the parser
parser.add_argument('--id', type=int, default=0,help='input file')
# Parse the arguments
args = parser.parse_args()

MAX_ROD_BUFFER = 10000

#update exposure time, in us
cfg_path = None #"/home/lyh/projects/lyncam_proj/lyncam_driver/cfg_csv/para_8bit/Golden_HCG_seq.csv"
lib_path = None #"/home/lyh/projects/tianmoucv/cpplib/libtianmouc.so"
tianmoucReader = TMstreamingDataReader(cfg_path=cfg_path,lib_path=lib_path,device_id=args.id)
exposure_time = 6600
tianmoucReader.setExposure(aoptime=1240,coptime=exposure_time,aopG=1,copG=1)

data_reader = TianmoucDataReader('./')
deblur_preprocess = TianmouDeblurDataset('./', cop_exposure_time=exposure_time, matchkey=None)
dbl_network = TianmoucDeblurNet(ckpt_path = None, _optim = False)
local_rank = 0
device = torch.device('cuda:'+str(local_rank))
dbl_network.set_device(device)

# deal with [COP] data and put the result back to result queue
def COP_thread():
    tsdiff_list = []
    with torch.no_grad():
        while True:
            time.sleep(0.01)
            if global_cop_queue.qsize() > 1:
                _ = global_cop_queue.get()
            if global_cop_queue.qsize() > 0:
                sample = {}  
                sample_cat = global_cop_queue.get()
                print('got!')
                print('cop ts',sample_cat['cop'][1])
                print('aop ts',[e[1] for e in sample_cat['aop']])

                rawdata, timestamp = sample_cat['cop']
                rgb = rawdata.astype(np.float32)
                sample['F0'] = torch.FloatTensor(rgb) / 255.0
                tsdiff_list = []
                for rawdata,timestamp in sample_cat['aop']:
                    td,sdl,sdr = [torch.FloatTensor(e) for e in rawdata]
                    tsdiff_raw = torch.stack([td,sdl,sdr],dim=0)
                    tsdiff_list.append(tsdiff_raw)

                tsd = torch.stack(tsdiff_list,dim=1)
                tsdiff_inter,_ = data_reader.tsd_preprocess(tsd)
                tsdiff_resized = F.interpolate(tsdiff_inter,(320,640),mode='bilinear')

                sample['tsdiff'] = tsdiff_resized
                td_voxel = deblur_preprocess._generate_voxel(-1, sample, _type='td_voxel')
                sd_voxel = deblur_preprocess._generate_voxel(-1, sample, _type='sd_voxel')

                    # 转换 frame 和 gt_frame 格式为 (3, 320, 640)
                item = {}
                item['frame'] = deblur_preprocess.transform_frame(sample['F0'],-1).to(device) 
                item['td_voxel'] = deblur_preprocess.transform_voxel(td_voxel, -1).to(device)
                item['sd_voxel'] = deblur_preprocess.transform_voxel(sd_voxel, -1).to(device)
                item['seq'] = 0

                pred = dbl_network(item)
                pred = pred[0,...].cpu().permute(1,2,0).numpy() 
                original_img = torch.flip(sample['F0'],dims=[1]).numpy()

                gap = np.ones([50,640,3])
                imshow = np.concatenate([original_img,gap,pred],axis=0)[:,:,[2,1,0]]

                result_queue.put(('original', imshow))

######################################################
#   main loop
######################################################
cfg_path = None #"/home/lyh/projects/lyncam_proj/lyncam_driver/cfg_csv/para_8bit/Golden_HCG_seq.csv"
lib_path = None #"/home/lyh/projects/tianmoucv/cpplib/libtianmouc.so"
tianmoucReader = TMstreamingDataReader(cfg_path=cfg_path,lib_path=lib_path,device_id=args.id)


with tianmoucReader as reader:
    count = 0
    threadcop = threading.Thread(target=COP_thread)
    threadcop.daemon = True  # Set the thread as a daemon thread
    threadcop.start()

    rod_buffer = dict([])
    cop_time = -1
    samples = {}
    samples['cop'] = None
    samples['aop'] = []
    flag_ready = False

    while True:
        isrod, timestamp, rawdata = reader()
        if isrod:
            timestamp = timestamp
            rod_buffer[timestamp] = [e.copy() for e in rawdata]
        else:
            count += 1
            if count % 10  == 0:
                cop_time = timestamp
                samples = {}
                samples['cop'] = (rawdata.copy(),timestamp)
                samples['aop'] = []

        if samples['cop'] is not None:
            matched_ts = []
            bias = -5
            for ts in rod_buffer:
                if ts > cop_time + bias and ts < cop_time + exposure_time//10 + bias:
                    samples['aop'].append((rod_buffer[ts],ts))

        if len(samples['aop']) > exposure_time//1320:
            global_cop_queue.put(samples)
            samples = {}
            samples['cop'] = None
            samples['aop'] = []

        if result_queue.qsize() > 0:
            result_name,data = result_queue.get()
            cv2.imshow(result_name, data)
            cv2.waitKey(1)
        

