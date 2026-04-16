#coding=utf-8
import sys
import numpy as np
import threading
import time
import argparse
import cv2
import torch
import torch.nn.functional as F

#用于色彩调整
from tianmoucv.isp import ACESToneMapping
#start camera and read
from tianmoucv.camera.sdk_utils import TMstreamingDataReader
from tianmoucv.camera.sdk_utils import global_aop_queue,global_cop_queue,result_queue,inference_queue

from tianmoucv.proc.segmentation import TM_seg 
from tianmoucv.proc.nn.utils import tdiff_split,_COLORS,segmentation_classes,coco_segmentation_classes_80,draw_result

import torch._dynamo
torch._dynamo.config.suppress_errors = True
cuda_device = torch.device('cuda:0')
yolo_seg_v5 =TM_seg(ckpt_path = None,device = cuda_device,_optim=False)

#warmup model
yolo_seg_v5(torch.randn([1,9,320,640]).to(cuda_device),np.zeros([320,640,3]),seg=True)
# Create the parser
parser = argparse.ArgumentParser(description='choose camera id')
# Add arguments to the parser
parser.add_argument('--id', type=int, default=0,help='input file')
# Parse the arguments
args = parser.parse_args()

# frame rate settings
INFERENCE_GAP = 30
AOP_SHOW_GAP = 15
MAX_ROD_BUFFER = 100#protect memory

def draw_result(pred,imshow,masks,text=''):
    si = 0  
    npr = pred.shape[0]  # number of labels, predictions
    if len(masks):
        masks = masks[0]
    else:
        return
    for i in range(npr):
        one_pred = pred[i,...].cpu()
        mask = masks[i,...].cpu()
        bbox = one_pred[:4]
        conf = one_pred[4]
        classid = int(one_pred[5])
        bbox = bbox.int().numpy()
        pt1 = bbox[:2]
        pt2 = bbox[2:]
        color = [int(c*255) for c in _COLORS[classid,:]]
        cv2.rectangle(imshow, pt1, pt2, color, 2)
        text_img = coco_segmentation_classes_80[int(classid)+1]
        cv2.putText(imshow, text_img, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        for c in range(3):
            imshow[...,c][mask>0] = (imshow[...,c]*0.5+color[c]*0.5)[mask>0]
            

# deal with [AOP] data and put the result back to result queue
def AOP_thread():
    count = 0
    while True:
        time.sleep(0.001)
        if global_aop_queue.qsize() > 1:
            _ = global_aop_queue.get()
        if global_aop_queue.qsize() > 0:
            count += 1
            #tsd
            rawdata = global_aop_queue.get()
            td,sdl,sdr = rawdata
            rawtsdiff = np.stack([td,sdl,sdr],axis=-1)*8
            #result_queue.put(('aop',showimg/32.0))
            #gray recon
            SD = torch.Tensor(np.stack([sdl,sdr],axis=-1))
            if count%AOP_SHOW_GAP == 0:
                result_queue.put(('AOP',rawtsdiff/255))

            SDt = SD.permute(2,0,1).unsqueeze(0).to(cuda_device)
            TDt = torch.Tensor(td).unsqueeze(0).unsqueeze(0).to(cuda_device)
            TDt = tdiff_split(TDt,cdim=1)
            TDt = F.interpolate(torch.Tensor(TDt), size=(320,640), mode='bilinear')
            SDt = F.interpolate(torch.Tensor(SDt), size=(320,640), mode='bilinear')
            #print('daatarange SDt TDt: ',torch.max(SDt),torch.max(TDt))
            inference_queue.put(('new_aop_channel',(SDt/255.0,TDt/255.0)))

# deal with [COP] data and put the result back to result queue
def COP_thread():
    while True:
        time.sleep(0.001)
        if global_cop_queue.qsize() > 1:
            _,_ = global_cop_queue.get()
        if global_cop_queue.qsize() > 0:
            #RGB show
            rawdata,rawdata_rod = global_cop_queue.get()
            #fuse HDR
            if not rawdata_rod is None:
                rgb = rawdata.astype(np.float32)
                td,sdl,sdr = rawdata_rod
                SD = torch.Tensor(np.stack([sdl,sdr],axis=-1))
                #Ix = F.interpolate(torch.Tensor(Ix).unsqueeze(0).unsqueeze(0), size=(320,640), mode='bilinear').squeeze(0).squeeze(0)
                #Iy = F.interpolate(torch.Tensor(Iy).unsqueeze(0).unsqueeze(0), size=(320,640), mode='bilinear').squeeze(0).squeeze(0)
                result_queue.put(('COP',rgb[...,[2,1,0]]/255.0))

                SD0 = SD.permute(2,0,1).unsqueeze(0).to(cuda_device)
                SD0 = F.interpolate(torch.Tensor(SD0), size=(320,640), mode='bilinear')
                rgb = torch.Tensor(rgb).permute(2,0,1).unsqueeze(0).to(cuda_device)

                inference_queue.put(('new_cop_channel',(rgb/255.0,SD0/255.0)))


# deal with [NN yoloRePViT] data and put the result back to result queue
def inference_thread():
    RGB_cuda = None
    SD_cuda = None
    SDt_cuda = None
    Tdt_cuda = None
    newest_imshow = None
    inference_count =0 
    while True:
        time.sleep(0.001)
        if inference_queue.qsize() > 0:
            result_name,data = inference_queue.get()

            if result_name == 'new_cop_channel':
                RGB_cuda,SD0_cuda = data
                Tdt_cuda = SD0_cuda*0
                SDt_cuda = SD0_cuda*0
            if result_name == 'new_aop_channel':
                SDt_cuda,TDt = data
                if Tdt_cuda is None:
                    Tdt_cuda = TDt
                else:
                    Tdt_cuda += TDt
            if result_name == 'new_cop_channel':
                newest_imshow = RGB_cuda[0,...].permute(1,2,0).cpu().numpy()

            inference_count += 1
            if inference_count%INFERENCE_GAP == 0:
                inference_count = 0
                if (not RGB_cuda is None) and (not SDt_cuda is None) and (not newest_imshow is None):
                    newest_imshow = newest_imshow.copy()*255
                    cat_input_full = torch.cat([RGB_cuda,Tdt_cuda,SDt_cuda,SD0_cuda],dim=1)
                    cat_input_full = cat_input_full.to(yolo_seg_v5.model.device) 
                    preds_full,time_cost,mask_list = yolo_seg_v5(cat_input_full,newest_imshow,seg=True)
                    pred_full = preds_full[0]
                    draw_result(pred_full,newest_imshow,mask_list,text='full')
                    result_queue.put(('inference_result',newest_imshow[...,[2,1,0]]/255.0))
                    print('forward,nms:',time_cost,'s')
                    
                while (inference_queue.qsize() > 30):
                    inference_queue.get()
            

######################################################
#   main loop
######################################################
cfg_path = None #"/home/lyh/projects/lyncam_proj/lyncam_driver/cfg_csv/para_8bit/Golden_HCG_seq.csv"
lib_path = None #"/home/lyh/projects/tianmoucv/cpplib/libtianmouc.so"
tianmoucReader = TMstreamingDataReader(cfg_path=cfg_path,lib_path=lib_path,device_id=args.id)
#update exposure time, in us
tianmoucReader.setExposure(aoptime=1200,coptime=15000,aopG=1,copG=1)


with tianmoucReader as reader:
    count = 0
    threadaop = threading.Thread(target=AOP_thread)
    threadcop = threading.Thread(target=COP_thread)
    threadnn  = threading.Thread(target=inference_thread)
    threadaop.daemon = True  # Set the thread as a daemon thread
    threadcop.daemon = True  # Set the thread as a daemon thread
    threadnn.daemon = True  # Set the thread as a daemon thread
    threadcop.start()
    threadaop.start()
    threadnn.start()
    rod_buffer = dict([])

    while True:
        isrod, timestamp, rawdata = reader()
        #push to rod thread
        if isrod:
            rod_buffer[timestamp] = [e.copy() for e in rawdata]
            count += 1
            global_aop_queue.put(rawdata)
        #push to cone thread
        else:
            min_gap = 1e99
            best_ts = -1
            nearestRod = None
            if len(rod_buffer)>0:
                for ts in rod_buffer:
                    if timestamp>ts:
                        gap = (timestamp-ts)
                    else:
                        gap = (ts-timestamp)
                    if gap < min_gap:
                        best_ts = ts
                        min_gap = gap
                nearestRod = rod_buffer[best_ts]
            global_cop_queue.put((rawdata.copy(),nearestRod))
            if len(rod_buffer)>MAX_ROD_BUFFER:
                new_rod_buffer = dict([])
                for ts in rod_buffer:
                    if best_ts<ts:
                        new_rod_buffer[ts] = rod_buffer[ts]
                rod_buffer = new_rod_buffer
                    
        # has results
        if result_queue.qsize() > 0:
            result_name,data = result_queue.get()
            cv2.imshow(result_name, data)
            cv2.waitKey(1)
        

