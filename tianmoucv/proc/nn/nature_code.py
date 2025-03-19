#before 20230815
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .unet_modules import *

MSE_LossFn = nn.MSELoss()

class SpyNet(torch.nn.Module):
    def __init__(self,dim=2+2+1):
        super().__init__()

        class Basic(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.netBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=dim, out_channels=32, kernel_size=5, stride=1, padding=2),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=5, stride=1, padding=2)
                )
            def forward(self, tenInput):
                return self.netBasic(tenInput)
            
        self.N_level = 6
        self.input_dim = dim
        self.netBasic = torch.nn.ModuleList([ Basic(dim+2) for intLevel in range(self.N_level) ])
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
            SD_0_list.append(torch.nn.functional.avg_pool2d(input=SD_0_list[-1], \
                                       kernel_size=2, stride=2, count_include_pad=False))
            SD_t_list.append(torch.nn.functional.avg_pool2d(input=SD_t_list[-1], \
                                       kernel_size=2, stride=2, count_include_pad=False))
            TD_list.append(torch.nn.functional.avg_pool2d(input=TD_list[-1], \
                                       kernel_size=2, stride=2, count_include_pad=False))

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
        
class UNet_Original(nn.Module):

    def __init__(self, inChannels, outChannels):
        super(UNet_Original, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.up2   = up(512, 256)
        self.up3   = up(256, 128)
        self.up4   = up(128, 64)
        self.up5   = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)
        
    def forward(self, x):
        x  = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        x = self.down4(s4)
        x  = self.up2(x, s4)
        x  = self.up3(x, s3)
        x  = self.up4(x, s2)
        x  = self.up5(x, s1)
        x  = torch.clamp(self.conv3(x),0,1)
        return x
        
class UNetRecon_Original(nn.Module):

    def __init__(self, inChannels, outChannels):
        super(UNetRecon_Original, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.up3   = up(256, 128)
        self.up4   = up(128, 64)
        self.up5   = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)
        
    def forward(self, catdata):
        catdata  = F.leaky_relu(self.conv1(catdata), negative_slope = 0.1)
        s1 = F.leaky_relu(self.conv2(catdata), negative_slope = 0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        x  = self.down3(s3)
        x  = self.up3(x, s3)
        x  = self.up4(x, s2)
        x  = self.up5(x, s1)
        x  = torch.clamp(self.conv3(x),0,1)
        return x
        