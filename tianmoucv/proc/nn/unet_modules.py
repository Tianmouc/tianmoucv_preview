#0917 version
import torch
try:
    import torchvision
    import torchvision.transforms as transforms
except:
    print('torch vision not supported')

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


######################################################################################################
#AttnNet
######################################################################################################
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        #print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        
        Channel_score = self.channel_attention(x)
        out = Channel_score * x
        Att_score = self.spatial_attention(out)
        out = Att_score * out
        return out,Channel_score

    
######################################################################################################
##UNET
######################################################################################################
class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, instance_norm=False,relu=False):

        super(ConvLayer, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.instance_norm = instance_norm
        self.instance = None
        self.relu = None
        if instance_norm:
            self.instance = nn.InstanceNorm2d(out_channels, affine=True)
        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.instance_norm:
            out = self.instance(out)
        if self.relu:
            out = self.relu(out)
        return out

class down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize, stride=1):
        super(down, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels, filterSize, stride=stride, padding=int((filterSize - stride) / 2))
        self.cbam = CBAM(outChannels)
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        
    def forward(self, x):
        # Average pooling with kernel size 2 (2 x 2).
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        x,_ = self.cbam(x)
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        return x
    
class up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 反卷积 layer 用于上采样
        self.conv_transpose = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.cbam = CBAM(2 * out_channels)
        self.conv2 = nn.Conv2d(2 * out_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x1, x2):
        x1 = self.conv_transpose(x1)
        # Crop x2 到 x1 的尺寸以进行跳跃链接 skip connection
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x,_ = self.cbam(x)
        x = self.conv2(x)
        
        return x


##Training Weakly Supervised Video Frame Interpolation with Events(accepted by ICCV2021)
class FuseLayer(nn.Module):
    def __init__(self, c_h, c_ef,debug=False):
        super(FuseLayer, self).__init__()
        self.c_ef = c_ef
        self.c_h = c_h
        self.debug = debug

        self.convE1 = nn.Conv2d(c_ef, c_h, kernel_size=1, stride=1, padding=0, bias=True)
        self.convE2 = nn.Conv2d(c_ef, c_h, kernel_size=1, stride=1, padding=0, bias=True)

        self.convF1 = nn.Conv2d(c_ef, c_h, kernel_size=1, stride=1, padding=0, bias=True)
        self.convF2 = nn.Conv2d(c_ef, c_h, kernel_size=1, stride=1, padding=0, bias=True)

        self.convMask = nn.Sequential(nn.ReplicationPad2d(padding=[1, 1, 1, 1]),
                                      nn.Conv2d(c_h, 1, kernel_size=3, stride=1, bias=False),
                                      nn.Sigmoid())

    def forward(self, h, z_e, z_f,h_prompt=None):
        gammaE = self.convE1(z_e)
        betaE = self.convE2(z_e)
        E = gammaE * h + betaE

        gammaF = self.convF1(z_f)
        betaF = self.convF2(z_f)
        F = gammaF * h + betaF

        if h_prompt is None:
            h_prompt = h 
            
        M = self.convMask(h_prompt)
        out = M * E + (1.0 - M) * F

        if self.debug:
            print('>>>recon rate:',float(torch.mean(M)),' flow rate:',float(torch.mean(1-M)),float(torch.std(M)) )
            # tmp_mask_attn_img = torch.mean(M,dim=1).detach().cpu().numpy()[0,...] * 255
            # cv2.imwrite('./tmp.png',tmp_mask_attn_img.astype(np.uint8))
        return out,M


class Interp(nn.Module):
    def __init__(self, scale=None, size=None):
        super(Interp, self).__init__()
        self.scale = scale
        self.size = size

    def forward(self, x):
        y = F.interpolate(x, self.size, self.scale, mode='bilinear', align_corners=True)
        return y

    
class FuseBlock(nn.Module):
    def __init__(self, cin, cout, c_ef,debug=False):
        super(FuseBlock, self).__init__()
        self.cin = cin
        self.cout = cout
        self.debug = debug

        self.AAD1 = FuseLayer(cin, c_ef,self.debug)
        self.conv1 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ReplicationPad2d(padding=[1, 1, 1, 1]),
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, bias=True)
        )

        self.AAD2 = FuseLayer(cin, c_ef)
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ReplicationPad2d(padding=[1, 1, 1, 1]),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, bias=True)
        )

        if cin != cout:
            self.AAD3 = FuseLayer(cin, c_ef)
            self.conv3 = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.ReplicationPad2d(padding=[1, 1, 1, 1]),
                nn.Conv2d(cin, cout, kernel_size=3, stride=1, bias=True)
            )

    def forward(self, h, z_e,z_f,h_prompt = None):
        x,M = self.AAD1(h,z_e,z_f,h_prompt)
        x = self.conv1(x)
        x,M = self.AAD2(x,z_e,z_f,h_prompt)
        x = self.conv2(x)
        if self.cin != self.cout:
            h,M = self.AAD3(h,z_e,z_f,h_prompt)
            h = self.conv3(h)
        x = x + h
        return x,M
    

#############################
#  @simpleNN
#############################
class UNetRecon(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Sequential(nn.ReplicationPad2d([1, 1, 1, 1]),
                                   nn.Conv2d(in_channels=inChannels, out_channels=32, kernel_size=3),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv2 = nn.Sequential(nn.ReplicationPad2d([1, 1, 1, 1]),
                                   nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.down1 = down(32, 64, 3)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.up3   = up(256, 128)
        self.up4   = up(128, 64)
        self.up5   = up(64, 32)
        self.conv3 = nn.Sequential(nn.ReplicationPad2d([1, 1, 1, 1]),
                                   nn.Conv2d(in_channels=32, out_channels=outChannels, stride=1, kernel_size=3))
        
    def forward(self, x0):
        s0  = self.conv1(x0)
        s1 = self.conv2(s0)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        x  = self.down3(s3)
        x  = self.up3(x, s3)
        x  = self.up4(x, s2)
        x  = self.up5(x, s1)
        x  = self.conv3(x)
        return x