import torch
import torch.nn as nn
import numpy as np
import math
import os
    
class Basic_larger_kernel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.netBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=dim, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=5, stride=1, padding=2)
                )
    def forward(self, tenInput):
        return self.netBasic(tenInput)

            
class Basic(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.netBasic = torch.nn.Sequential(
                    nn.ReplicationPad2d([2]*4),
                    torch.nn.Conv2d(in_channels=dim, out_channels=32, kernel_size=5, stride=1, padding=0),
                    torch.nn.ReLU(inplace=False),
                    nn.ReplicationPad2d([2]*4),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
                    torch.nn.ReLU(inplace=False),
                    nn.ReplicationPad2d([2]*4),
                    torch.nn.Conv2d(in_channels=64, out_channels=16, kernel_size=5, stride=1, padding=0),
                    torch.nn.ReLU(inplace=False),
                    nn.ReplicationPad2d([2]*4),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=5, stride=1, padding=0) 
                )
    def forward(self, tenInput):
        return self.netBasic(tenInput)
            
class SpyNet(torch.nn.Module):
    def __init__(self,dim=2+2+1,stage=0):
        super().__init__()
        self.N_level = 6
        self.input_dim = dim
        self.netBasic = torch.nn.ModuleList([ Basic_larger_kernel(dim+2) for intLevel in range(self.N_level) ])
        self.backwarp_tenGrid = {}

    def backwarp(self, tenInput, tenFlow):
        if str(tenFlow.shape) not in self.backwarp_tenGrid:
            tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - \
                                    (1.0 / tenFlow.shape[3]),
                                    tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
            tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - \
                                    (1.0 / tenFlow.shape[2]), 
                                    tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])
            self.backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).to(tenInput.device)

        tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), \
                             tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

        return torch.nn.functional.grid_sample(input=tenInput, grid=(self.backwarp_tenGrid[str(tenFlow.shape)] \
                                + tenFlow).permute(0, 2, 3, 1), \
                                mode='bilinear', padding_mode='border', align_corners=False)

    def forward(self, TD, SD_0, SD_t):
        
        SD_0_list = [SD_0]
        SD_t_list = [SD_t]
        TD_list = [TD]
        
        for intLevel in range(self.N_level-1):
            SD_0_list.append(torch.nn.functional.max_pool2d(input=SD_0_list[-1], \
                                       kernel_size=2, stride=2))
            SD_t_list.append(torch.nn.functional.max_pool2d(input=SD_t_list[-1], \
                                       kernel_size=2, stride=2))
            TD_list.append(torch.nn.functional.max_pool2d(input=TD_list[-1], \
                                       kernel_size=2, stride=2))

        tenFlow = torch.zeros([ SD_0_list[-1].size(0), 2, \
                    int(math.floor(SD_0_list[-1].size(2) / 2.0)), int(math.floor(SD_0_list[-1].size(3) / 2.0)) ])
        
        tenFlow = tenFlow.to(TD.device)
        
        for intLevel in range(self.N_level):
            
            invert_index = self.N_level-intLevel-1
            Flow_upsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, \
                                                           mode='bilinear', align_corners=True) * 2.0
            if Flow_upsampled.size(2) != SD_0_list[invert_index].size(2): \
                Flow_upsampled = torch.nn.functional.pad(input=Flow_upsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
            if Flow_upsampled.size(3) != SD_0_list[invert_index].size(3): \
                Flow_upsampled = torch.nn.functional.pad(input=Flow_upsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

            tenFlow = self.netBasic[intLevel](torch.cat([ SD_0_list[invert_index], \
                                                          self.backwarp(SD_t_list[invert_index], Flow_upsampled), \
                                                          TD_list[invert_index],\
                                                          Flow_upsampled ], 1)) + Flow_upsampled
        return tenFlow

