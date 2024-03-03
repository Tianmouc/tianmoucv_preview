import os
train='/home/lyh/data/tianmouc_20240124/'
dirlist = os.listdir(train)
traindata = [train]
key_list = []
print('---------------------------------------------------')
for sampleset in traindata:
    print('---->',sampleset,'有：',len(os.listdir(sampleset)),'个样本')
    for e in os.listdir(sampleset):
        print(e)
        key_list.append(e)

all_data = traindata

key_list = ['long_save_2']

import sys,os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import math,time
import matplotlib.pyplot as plt

from tianmoucv.isp import fourdirection2xy
from tianmoucv.data import TianmoucDataReader

for key in key_list:
    dataset = TianmoucDataReader(all_data,MAXLEN=1000,ifSaveFileDict = False,
                              matchkey=key,speedUpRate=1)
    img_list = []
    for index in range(len(dataset)):
        if index in [0, len(dataset)//2,len(dataset)-2]:
            sample = dataset[index]
            F0 = sample['F0']
            F1 = sample['F1']
            tsdiff = sample['tsdiff']
            print('shapes:',F0.shape,tsdiff.shape)
            plt.figure(figsize=(15,5))
            plt.subplot(1,5,1)
            plt.imshow(F0)
            plt.subplot(1,5,2)
            plt.imshow(tsdiff[:,12,...].transpose(0,2).transpose(0,1))
            plt.subplot(1,5,3)
            plt.imshow(F1)
            plt.show()
