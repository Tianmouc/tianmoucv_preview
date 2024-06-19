import numpy as np
import cv2,sys
import torch
import math,time
import torch.nn.functional as F
import os
flag = True

class TianmoucDataSampleParser():
    def __init__(self,sample):
        self.sample = sample
        self.dataRate = sample['dataRatio']
        for i in range(100000):
            if 'F'+str(i) in sample:
                self.N = i
            else:
                break

    def get_data_rate(self):
        return self.dataRate

    def get_sd(self,idx=0,end=-1,ifUpSampled=False):
        if idx > self.N * self.dataRate + 1:
            print('idx exceed the maximum number:',self.N * self.dataRate + 1)
            return None
        if ifUpSampled:
            tsd = self.sample['tsdiff']
        else:
            tsd = self.sample['rawDiff']
        return tsd[1:,idx:end,:,:]

    def get_td(self,idx=0,end=-1,ifUpSampled=False):
        if idx > self.N * self.dataRate + 1:
            print('idx exceed the maximum number:',self.N * self.dataRate + 1)
            return None
        if ifUpSampled:
            tsd = self.sample['tsdiff']
        else:
            tsd = self.sample['rawDiff']
        return tsd[0:1,idx:end,:,:]

    def get_tsd(self,idx=0,end=-1,ifUpSampled=False):
        if idx > self.N * self.dataRate + 1:
            print('idx exceed the maximum number:',self.N * self.dataRate + 1)
            return None
        if ifUpSampled:
            tsd = self.sample['tsdiff']
        else:
            tsd = self.sample['rawDiff']
        return tsd[:,idx:end,:,:]
        
    def get_rgb(self,idx=0,needISP=True):
        if idx > self.N:
            print('idx exceed the maximum number:',self.N)
            return None
        if needISP:
            return self.sample['F'+str(idx)]
        else:
            return self.sample['F'+str(idx)+'_without_isp']
            
    def get_hdr_fusion(self,idx=0,end=-1):
        if idx > self.N:
            print('idx exceed the maximum number:',self.N)
            return None
        return self.sample['F'+str(idx)+'_HDR']
        
    def help(self):
        class_methods = [method for method in dir(self) if callable(getattr(self, method))]
        # 打印类的所有方法
        for method in class_methods:
            if not '__' in method:
                print(method)

    def get_raw_sample(self):
        return self.sample

    def get_aop_in_camera_time_stamp(self,idx=0):
        print('The ',idx,' th AOP sample\'s time stamp is:',self.sample['meta']['R_timestamp'][idx],'us')
        return self.sample['meta']['R_timestamp'][idx]

    def get_cop_in_camera_time_stamp(self,idx=0):
        print('The ',idx,' th COP sample\'s time stamp is:',self.sample['meta']['C_timestamp'][idx],'us')
        return self.sample['meta']['C_timestamp'][idx]

    def get_sample_system_timestamp(self,idx=0):
        print('The entire sample starts from:',self.sample['sysTimeStamp'],'us from 1970/1/1 0:00')
        return self.sample['sysTimeStamp']
