import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda import amp

from tianmoucv.isp import fourdirection2xy
from tianmoucv.proc.reconstruct import laplacian_blending_1c_batch
from tianmoucv.proc.reconstruct import TianmoucRecon_tiny
from tianmoucv.proc.opticalflow import TianmoucOF_RAFT
from tianmoucv.proc.nn.utils import tdiff_split
from tianmoucv.proc.nn import CBAM
import time
    
from .repvit import Conv2d_BN,RepViTBlock,_make_divisible
from .icafusion import TransformerFusionBlock

#share with common
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
        
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class DeformableConv2d_v3(nn.Module):
    default_act = nn.SiLU()  # default activation
    def __init__(self, in_channels_list, out_channels, kernel_size, stride=1, padding=1,group=4, internelchannels=16, act=True):
        super(DeformableConv2d_v3, self).__init__()
        self.defconv = opsm.DCNv3_Mod(cop_ch = in_channels_list[0],
                                        aop_ch = in_channels_list[1],
                                        internelchannels = internelchannels,
                                        outchannels = out_channels,
                                        kernel_size=kernel_size,
                                        dw_kernel_size=None,
                                        stride=stride,
                                        pad=padding,
                                        dilation=1,
                                        group=group,
                                        offset_scale=1.0,
                                        act_layer='GELU',
                                        norm_layer='BN',
                                        center_feature_scale=False,
                                        remove_center=False)

    def forward(self, x1, x2):

        x1 = x1.permute(0,2,3,1)
        x2 = x2.permute(0,2,3,1)
        defomed_feature = self.defconv(COP=x1,AOP=x2)
        defomed_feature = defomed_feature.permute(0,3,1,2)

        return defomed_feature

class DualAlignedConv2d(nn.Module):
    default_act = nn.SiLU()  # default activation
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, act=True):
        super(DualAlignedConv2d, self).__init__()
        #print('debug args:',in_channels, out_channels, kernel_size, stride, padding, act)
        self.sdof = TianmoucOF_RAFT()
        self.sdof.eval()
        self.conv = nn.Conv2d(5, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        
    def forward_aligned(self, rawx):
        img = rawx[:,:3,...]
        td  = rawx[:,3:5,...]
        sd1 = rawx[:,5:7,...]
        sd0 = rawx[:,7:9,...]
        if self.training:
            iters = 20 # change to 5? original is 20
        else:
            iters = 20
        corrOF = self.sdof(td, sd0, sd1,iters=iters,print_fps=False)
        corrOF = corrOF[1]
        wrappedImg = self.sdof.backWarp(img, corrOF, dim=-1)
        
        aligned_Dual_Feature = torch.cat([wrappedImg,sd1],dim=1)
        return aligned_Dual_Feature
        
    def setExpArg(self,args):
         self.turn_off_sd1 = args
                
    def forward(self, x):
        det_f = self.forward_aligned(x)
        return self.act(self.bn(self.conv(det_f)))

    def forward_fuse(self, x):
        det_f = self.forward_aligned(x)
        return self.act(self.conv(det_f))
        
        
class CBAM_(nn.Module):
    def __init__(self, in_chans=3):
        super().__init__()
        self._cbam = CBAM(in_chans)
    
    def forward(self, x):
        x,score = self._cbam(x)
        return x


class MultiRepresentationFuser_v1(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, act=True):
        super(MultiRepresentationFuser_v1, self).__init__()
        #✅ Gray Recon(roburst)
        #✅ RGB Recon(fine)
        #✅ vector space
        #✅ CBAM for feature strengthen
        default_act = nn.SiLU()  # default activation
        self.recon = Reconstrutor_NN()
        self.recon.eval()

        assert out_channels%3 == 0
        hidden = out_channels//3
        
        self.enc1 = nn.Sequential(Conv(1,hidden,kernel_size,stride,p=1),
                                       C3(hidden,hidden))
        self.enc2 = nn.Sequential(Conv(3,hidden,kernel_size,stride,p=1),
                                       C3(hidden,hidden))
        self.enc3 = nn.Sequential(Conv(3,hidden,kernel_size,stride,p=1),
                                       C3(hidden,hidden))
        
        self.fuse_conv  = nn.Sequential(CBAM_(out_channels),
                                        nn.BatchNorm2d(out_channels),
                                        default_act)

    def forward_gray(self,sdxy1):
        Ix = sdxy1[:,0:1,...]
        Iy = sdxy1[:,1:2,...]
        gray = laplacian_blending_1c_batch(-Ix,-Iy,None,iteration=20)
        return gray
    
    def forward_recon(self,img0,td,sd1):
        inputTensor = torch.cat([img0,td,sd1],dim=1)
        recon = self.recon.reconNet(inputTensor)
        return recon

    def forward_vector_rep(self,td,sd1):
        b,_,h,w = sd1.shape
        vector = torch.zeros([b,3,h,w]).to(sd1.device)
        vector[:,0:2,...] = sd1
        vector[:,2,...] = torch.sum(td,dim=1)
        return vector
        
    def forward(self, x, shallowFeature=False):
        img0 = x[:,:3,...] #keep in 0~1
        td  = x[:,3:5,...]
        sd1 = x[:,5:7,...]
        sd0 = x[:,7:9,...]

        SDxy0 = fourdirection2xy(sd0)
        SDxy1 = fourdirection2xy(sd1)
        
        gray = self.forward_gray(SDxy1).detach()
        recon = self.forward_recon(img0,td,sd1).detach()
        vector = self.forward_vector_rep(td,sd1).detach()

        f1 = self.enc1(gray)
        f2 = self.enc2(recon)
        f3 = self.enc3(vector)

        cat_rep = torch.cat([f1,f2,f3],dim=1)
        fuse_feature = self.fuse_conv(cat_rep)

        return fuse_feature


class MultiRepresentationFuser_v2(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, act=True):
        super(MultiRepresentationFuser_v2, self).__init__()
        #👌 HDR delete
        #❎ OF is too slow for motion
        #✅ Gray Recon(roburst)
        #✅ RGB Recon(fine)
        #✅ deformed conv
        default_act = nn.SiLU()  # default activation
        self.recon = Reconstrutor_NN()
        self.recon.eval()
        self.defconv = DeformableConv2d_v3(7,out_channels,kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           group=4,
                                           internelchannels = 16)
    def forward_gray(self,sdxy1):
        Ix = sdxy1[:,0:1,...]
        Iy = sdxy1[:,1:2,...]
        gray = laplacian_blending_1c_batch(-Ix,-Iy,None,iteration=20)
        #gray_feature = self.encoder_gray(gray)
        return gray
    
    def forward_recon(self,img0,td,sd1):
        inputTensor = torch.cat([img0,td,sd1],dim=1)
        recon = self.recon.reconNet(inputTensor)
        #recon_feature = self.encoder_recon(recon)
        return recon
        
    def forward(self, x, shallowFeature=False):
        img0 = x[:,:3,...] #keep in 0~1
        td  = x[:,3:5,...]
        sd1 = x[:,5:7,...]
        sd0 = x[:,7:9,...]

        SDxy0 = fourdirection2xy(sd0)
        SDxy1 = fourdirection2xy(sd1)
        
        gray = self.forward_gray(SDxy1)
        
        recon = self.forward_recon(img0,td,sd1)
        
        core_representation = torch.cat([img0,gray,recon],dim=1).detach()
        
        guidance_representation = torch.cat([sd0,sd1],dim=1).detach()
        
        deformed_feature = self.defconv(core_representation,guidance_representation)

        return deformed_feature


class Input_(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()

    def forward(self, x):
        return x

    def forward_fuse(self, x):
        return x


####dual stream backbone
class ConvCOP(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(5, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        img0 = x[:,:3,...] #keep in 0~1
        sd0 = x[:,7:9,...]
        input_data = torch.cat([img0,sd0],dim=1)
        return self.act(self.bn(self.conv(input_data)))

    def forward_fuse(self, x):
        img0 = x[:,:3,...] #keep in 0~1
        sd0 = x[:,7:9,...]
        
        input_data = torch.cat([img0,sd0],dim=1)
        return self.act(self.bn(self.conv(input_data)))


class ConvAOP(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(4, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # group conv, one for integration, another for feature strengthen
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        td  = x[:,3:5,...]
        sd1 = x[:,5:7,...]
        input_data = torch.cat([td,sd1],dim=1)
        return self.act(self.bn(self.conv(input_data)))

    def forward_fuse(self, x):
        td  = x[:,3:5,...]
        sd1 = x[:,5:7,...]
        input_data = torch.cat([td,sd1],dim=1)
        return self.act(self.conv(input_data))


class ConvFuse(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation
    def __init__(self, c_in_list, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.c_in_list = c_in_list

        assert c_in_list[0] == c_in_list[1]
        self.fuse_add  = nn.Sequential(nn.Conv2d(c_in_list[0], c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False),
                                       nn.BatchNorm2d(c2),
                                       CBAM_(c2))
        
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()


    def forward(self, x):
        #x_ = torch.cat([x[0],x[1]],dim=1)
        x_ = x[0] + x[1]
        #print(x[0].shape,x[1].shape)
        return self.act(self.bn(self.fuse_add(x_)))

    def forward_fuse(self, x):
        #x_ = torch.cat([x[0],x[1]],dim=1)
        x_ = x[0] + x[1]
        return self.act(self.fuse_add(x_))


######################v4

class ConvCOPv4(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, cin, cout, feature_level):
        '''
        build based on RepViT module
        '''
        super(ConvCOPv4, self).__init__()

        layers = []
        self.feature_level = feature_level
        if feature_level == 0:
            self.cfgs = [
                [3,   2,  cout, 1, 0, 1,True],
                ]
            input_channel = cout
            patch_embed = torch.nn.Sequential(Conv2d_BN(5, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                               Conv2d_BN(input_channel // 2, input_channel, 3, 1, 1))
            layers.append(patch_embed)

        elif feature_level == 1:
            input_channel = cin
            self.cfgs = [
                [3,   2,  cin, 0, 0, 1,False],
                [3,   2,  cin, 0, 0, 1,False],
                [3,   2,  cout, 0, 0, 2,True],
                ]
        elif feature_level == 2:
            input_channel = cin    
            self.cfgs = [
                [3,   2,  cin, 1, 0, 1,False],
                [3,   2,  cin, 0, 0, 1,False],
                [3,   2,  cin, 0, 0, 1,False],
                [3,   2,  cout, 0, 1, 2,True],
                ]            
        elif feature_level == 3:
            input_channel = cin    
            self.cfgs = [
                [3,   2, cin, 1, 1, 1,False],
                [3,   2, cin, 0, 1, 1,False],
                [3,   2, cin, 1, 1, 1,False],
                [3,   2, cin, 0, 1, 1,False],
                [3,   2, cin, 1, 1, 1,False],
                [3,   2, cout, 0, 1, 2,True],
                ]
        elif feature_level == 4:
            input_channel = cin    
            self.cfgs = [
                [3,   2, cin, 1, 1, 1,False],
                [3,   2, cin, 0, 1, 1,False],
                [3,   2, cout, 0, 1, 2,True],
                ]            
        # building first layer
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s, ifoutput in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)

    def forward(self, x):
        if x.shape[1]==9:
            img0 = x[:,:3,...] #keep in 0~1
            sd0 = x[:,7:9,...]
            input_data = torch.cat([img0,sd0],dim=1)
        else:
            input_data = x        
            
        for f in self.features:
            input_data = f(input_data)

        return input_data

    def forward_fuse(self, x):
        if x.shape[1]==9:
            img0 = x[:,:3,...] #keep in 0~1
            sd0 = x[:,7:9,...]
            input_data = torch.cat([img0,sd0],dim=1)
        else:
            input_data = x
        
        for f in self.features:
            input_data = f(input_data)
        return input_data


class ConvAOPv4(nn.Module):

    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(4, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # group conv, one for integration, another for feature strengthen
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        td  = x[:,3:5,...]
        sdt = x[:,5:7,...]
        input_data = torch.cat([td,sdt],dim=1)
        return self.act(self.bn(self.conv(input_data)))

    def forward_fuse(self, x):
        td  = x[:,3:5,...]
        sdt = x[:,5:7,...]
        input_data = torch.cat([td,sdt],dim=1)
        return self.act(self.conv(input_data))


class ConvFusev4(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation
    def __init__(self, c_in_list, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.c_in_list = c_in_list

        assert c_in_list[0] == c_in_list[1]
        self.fuse_add  = nn.Sequential(nn.Conv2d(c_in_list[0], c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False),
                                       nn.BatchNorm2d(c2),
                                       CBAM_(c2))
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        #x_ = torch.cat([x[0],x[1]],dim=1)
        x_ = x[0] + x[1]
        #print(x[0].shape,x[1].shape)
        return self.act(self.bn(self.fuse_add(x_)))

    def forward_fuse(self, x):
        #x_ = torch.cat([x[0],x[1]],dim=1)
        x_ = x[0] + x[1]
        return self.act(self.fuse_add(x_))

    
    
class FuseBlockv5(nn.Module):

    def __init__(self, in_channel, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.fusion_block  = TransformerFusionBlock(in_channel, vert_anchors=vert_anchors, horz_anchors=horz_anchors, 
                                                h=h, block_exp=block_exp, 
                                                n_layer=n_layer, embd_pdrop=embd_pdrop, 
                                                attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)

    def forward(self, x):
        return self.fusion_block(x)

    def forward_fuse(self, x):
        return self.fusion_block(x)
