import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))
sys.path.append(BASE_DIR)

import argparse
import time
import cv2
import torch
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.general import (LOGGER, check_dataset, check_img_size, check_yaml,scale_boxes,
                           coco80_to_coco91_class, colorstr, emojis, increment_path, non_max_suppression, print_args, xywh2xyxy, xyxy2xywh)
from tianmoucv.tools import check_url_or_local_path,download_file

def get_opt():
    parser = argparse.ArgumentParser(description='Test Multitask network')
    opt = parser.parse_args(args=[])
    opt.iou_thres= 0.2
    opt.conf_thres = 0.2
    opt.imgsz= 320
    opt.task= 'val'
    opt.device= torch.device("cuda:0")
    opt.half= False
    opt.data = None # check YAML
    opt.cfg = None
    print(vars(opt))
    return opt

class TM_seg():
    
    def __init__(self,ckpt_path=None,device=None,_optim=True):

        if ckpt_path is None:
            ckpt_path = 'http://www.tianmouc.cn:38328/index.php/s/xcRk6CnCC78kKw3/download/yoloViT_SEG_v4_mix_0245.pt'
        status = check_url_or_local_path(ckpt_path)
        print('loading..:',ckpt_path)
        if status == 1:
            default_file_name = 'yoloRepviTSeg.pt'
            if not os.path.exists(default_file_name):
                ckpt_path = download_file(url=ckpt_path,file_name=default_file_name)
            else:
                ckpt_path = default_file_name
       
        self.opt = get_opt()
        iou_thres= self.opt.iou_thres
        dnn = False
        
        self.opt.data = None
        self.opt.cfg = None
        half = self.opt.half 

        self.device  = device
        self.model = DetectMultiBackend(ckpt_path, device=self.device, dnn=dnn, data=None, fp16=half)
        print('load finished')
        stride, pt, jit, engine = self.model.stride, self.model.pt, self.model.jit, self.model.engine
        half = self.model.fp16  # FP16 supported on limited backends with CUDA
        
        self.model.eval()
        self.model.warmup(imgsz=(1,9,320,640))  # warmup
        
        main_version = int(torch.__version__[0])
        if main_version==2 and _optim:
            print('compiling model for pytorch version>= 2.0.0')
            self.model = torch.compile(self.model)
            print('compiled!')
        
    def __call__(self,input_data,img_raw,seg=False):
        seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
 
        t1 = time.time()
        im0 = img_raw

        t2 = time.time()
        dt[0] += t2 - t1

        masks = None
        if seg:
            pred, proto = self.model(input_data, augment=False, visualize=False)[:2]
        else:
            pred = self.model(input_data, augment=False, visualize=False)

        t3 = time.time()
        dt[1] += t3 - t2

        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=None, max_det=100, nm=32)

        dt[2] += time.time() - t3

        mask_list = []

        if seg:
            for i, det in enumerate(pred): 
                if len(det):
                    det[:, :4] = scale_boxes(input_data.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                    mask_list.append(masks)

            return pred,dt,mask_list
        else:
            return pred,dt
               


