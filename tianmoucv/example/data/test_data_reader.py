#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')


# # 引入必要的库

# In[ ]:


get_ipython().run_line_magic('autoreload', '')
import sys,os
import numpy as np
import matplotlib.pyplot as plt
from tianmoucv.isp import fourdirection2xy,poisson_blend
import torch
from tianmoucv.data import TianmoucDataReader
import torch.nn.functional as F
import cv2

train='/data/lyh/tianmoucData/tianmoucReconDataset/train/'
dirlist = os.listdir(train)
traindata = [train + e for e in dirlist]

val='/data/lyh/tianmoucData/tianmoucReconDataset/test/'
vallist = os.listdir(val)
valdata = [val + e for e in vallist]
key_list = []

print('---------------------------------------------------')
for sampleset in traindata:
    print('---->',sampleset,'有：',len(os.listdir(sampleset)),'个样本')
    for e in os.listdir(sampleset):
        print(e)
        key_list.append(e)
print('---------------------------------------------------')
for sampleset in valdata:
    print('---->',sampleset,'有：',len(os.listdir(sampleset)),'个样本')
    for e in os.listdir(sampleset):
        print(e)
        key_list.append(e)
        
all_data = valdata + traindata


# # 读取

# In[ ]:


get_ipython().run_line_magic('autoreload', '')
import sys,os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import math,time
import matplotlib.pyplot as plt

from tianmoucv.isp import fourdirection2xy
from tianmoucv.data import TianmoucDataReader_multiple

aim = 4
N = 6   # read continue N frames

for key in key_list:
    dataset = TianmoucDataReader_multiple(all_data,N=N,matchkey=key)
    img_list = []
    for index in range(aim,min(aim+1,len(dataset))):
        sample = dataset[index]
        tsdiff = torch.Tensor(sample['tsdiff'])
        length = tsdiff.shape[1]
        print('一次性读出N:',N,'个RGB帧以及与其同步的AOP')
        gap = length//(N+1)
        plt.figure(figsize=(12,2*N))  
        for i in range(N):
            F = sample['F'+str(i)]
            F_HDR = sample['F'+str(i)+'_HDR']
            F_HDR[F_HDR>1]=1
            F_HDR[F_HDR<0]=0
            plt.subplot(N,3,1+i*3)
            plt.imshow(F)
            plt.subplot(N,3,2+i*3)
            plt.imshow(tsdiff[:,i*gap,...].permute(1,2,0)*16)
            plt.subplot(N,3,3+i*3)
            plt.imshow(F_HDR)
        plt.show()
    break

