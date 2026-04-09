'''
EFNet
@inproceedings{sun2022event,
      author = {Sun, Lei and Sakaridis, Christos and Liang, Jingyun and Jiang, Qi and Yang, Kailun and Sun, Peng and Ye, Yaozu and Wang, Kaiwei and Van Gool, Luc},
      title = {Event-Based Fusion for Motion Deblurring with Cross-modal Attention},
      booktitle = {European Conference on Computer Vision (ECCV)},
      year = 2022
      }
'''

import torchvision.utils as vutils
import os

import torch
import torch.nn as nn
import random
from .arch_util import EventImage_ChannelAttentionTransformerBlock
from torch.nn import functional as F

def conv3x3(in_chn, out_chn, bias=True):  #  3x3 的卷积层
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):   # 下采样卷积层（4x4，stride=2）
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):  # 任意大小的卷积层
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


class SAM(nn.Module):    # Supervised Attention Module类：监督注意力模块

    '''
        用于提升图像特征与图像输入的注意力机制。通过卷积操作和 sigmoid 激活函数来生成注意力权重，将这些权重应用于输入特征图，从而增强特定区域的特征。

    '''
    def __init__(self, n_feat, kernel_size=3, bias=True, save_attn=False):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)
        self.save_attn = save_attn  # 是否保存注意力图
        self.counter = 0

    def forward(self, x, x_img):   # x:输入Res特征  x_img:模糊图
        x1 = self.conv1(x)  # Res特征
        img = self.conv2(x) + x_img #Res图+模糊图
        x2 = torch.sigmoid(self.conv3(img)) #（Res图+模糊图）注意力图 0-1

        # # ------- 可视化部分 -------
        # if self.save_attn and self.counter == 0:
        #     self.counter += 1  # 确保只保存一次
        #     attn_mean = x2.mean(dim=1, keepdim=True)   # (B,1,H,W)
        #     attn_max = x2.max(dim=1, keepdim=True)[0]  # (B,1,H,W)

        #     # 归一化到 [0,1]
        #     attn_mean = (attn_mean - attn_mean.min()) / (attn_mean.max() - attn_mean.min() + 1e-8)
        #     attn_max = (attn_max - attn_max.min()) / (attn_max.max() - attn_max.min() + 1e-8)

        #     os.makedirs("attn_vis", exist_ok=True)
        #     vutils.save_image(attn_mean, "/data32/yanglin/tianmouc_project/EFNet/FEG_attn_vis/A_mean.png")
        #     vutils.save_image(attn_max, "/data32/yanglin/tianmouc_project/EFNet/FEG_attn_vis/A_max.png")
        #     print("[FEG] Saved attention map visualization to attn_vis/")

        # # -------------------------
            

        x1 = x1*x2 #增强Res特征
        x1 = x1+x  #增强Res特征+输入Res特征
        # return x1, img
        # x1 = x + self.conv1(x) * torch.sigmoid(self.conv3(out))  # out = self.conv2(x) + x_img
        return x1, img


class oneStageRecurrent3(nn.Module):  
    def __init__(self, in_chn=3, td_chn=12, sd_chn=2, wf=48, depth=3, fuse_before_downsample=True, relu_slope=0.2, num_heads=[1,2,4]):
        super().__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        
        ########## Modified: Split event branch into TD/SD ##########
        # Image branch
        self.down_path_1 = nn.ModuleList()
        # self.down_path_2 = nn.ModuleList()
        
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        # self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)

        self.residual_image_init = nn.Parameter(torch.zeros((1, wf, 320, 640)))

        # TD branch 
        self.down_path_td = nn.ModuleList()  # Added TD path
        self.conv_td1 = nn.Conv2d(1, wf, 3, 1, 1)  # (5, 64, 3, 3)
        # self.conv_td1 = nn.Conv2d(td_chn, wf, 3, 1, 1)  # (5, 64, 3, 3)
        ## NEW TD 逻辑：输入 B C(12) H W -> reshape B*C 1 H W -> self.conv_td1 -> self.down_path_td -> 
        
        # SD branch 
        self.down_path_sd = nn.ModuleList()  # Added SD path
        self.conv_sd1 = nn.Conv2d(sd_chn, wf, 3, 1, 1)  # (2, 64, 3, 3)
        # ------------------------------------------------------------------



        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i+1) < depth else False

            if i == 0:
                self.down_path_1.append(UNetConvBlock(prev_channels*2, (2**i) * wf, downsample, relu_slope, num_heads=self.num_heads[i]))
            else:
                self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, num_heads=self.num_heads[i]))
            # self.down_path_2.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_emgc=downsample))

            ########## Modified: Create TD/SD down paths ##########
            if i < self.depth:
                # TD encoder
                self.down_path_td.append(UNetEVConvBlock(prev_channels, (2**i)*wf, downsample, relu_slope))
                # SD encoder
                self.down_path_sd.append(UNetEVConvBlock(prev_channels, (2**i)*wf, downsample, relu_slope))
            # ------------------------------------------------------------------

            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        # self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        # self.skip_conv_2 = nn.ModuleList()
        
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            # self.up_path_2.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            # self.skip_conv_2.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            prev_channels = (2**i)*wf
        
        self.sam12 = SAM(prev_channels,save_attn=False)
        # self.cat12 = nn.Conv2d(prev_channels*2, prev_channels, 1, 1, 0)
        self.last = conv3x3(prev_channels, in_chn, bias=True)


    def forward(self, x, td, sd, mask=None):  # Modified input parameters
        image = x

        x_image = self.conv_01(image)
        # x_prev = x1

        residual_image = self.residual_image_init.repeat(x.shape[0], 1, 1, 1)  # (1, wf, 320, 640) -> (B, wf, 320, 640)

        # 未来考虑改成 SD也一帧一帧输入 先只用TD
        # SD encoding SD先只编码一次
        sdiff = []
        sd_feat = self.conv_sd1(sd)

        for i, down in enumerate(self.down_path_sd):
            if i < self.depth-1:
                sd_feat, sd_up = down(sd_feat, self.fuse_before_downsample)
                sdiff.append(sd_up if self.fuse_before_downsample else sd_feat)
            else:
                sd_feat = down(sd_feat, self.fuse_before_downsample)
                sdiff.append(sd_feat)

        # TD shape is (B, C, H, W)
        num_TD_frames = td.shape[1]
        print("num_TD_frames:", num_TD_frames, td.shape)
        for t in range(num_TD_frames):

            # TD encoding 当前步 TD
            td_t = td[:, [t], :, : ]  # B, 1, H, W
            tdiff = []
            td_feat = self.conv_td1(td_t)
            
            for i, down in enumerate(self.down_path_td):
                if i < self.depth-1:
                    td_feat, td_up = down(td_feat, self.fuse_before_downsample)
                    tdiff.append(td_up if self.fuse_before_downsample else td_feat)
                else:
                    td_feat = down(td_feat, self.fuse_before_downsample)
                    tdiff.append(td_feat)

            # residual_image_prev = residual_image

            #stage 1            
            encs = []
            # decs = []
            # masks = []

            residual_image, _ = self.sam12(residual_image, image)
            residual_image = torch.cat([x_image, residual_image], dim=1)

            for i, down in enumerate(self.down_path_1):
                if (i+1) < self.depth:
                    residual_image, x1_up = down(residual_image, td_filter=tdiff[i], sd_filter=sdiff[i], merge_before_downsample=self.fuse_before_downsample)
                    # x1, x1_up = down(x1, td_filter=tdiff[i], sd_filter=None, merge_before_downsample=self.fuse_before_downsample)
                    encs.append(x1_up)

                    # if mask is not None:
                    #     masks.append(F.interpolate(mask, scale_factor = 0.5**i))
                else:
                    residual_image = down(residual_image, td_filter=tdiff[i], sd_filter=sdiff[i], merge_before_downsample=self.fuse_before_downsample)
                    # x1 = down(x1, td_filter=tdiff[i], sd_filter=None, merge_before_downsample=self.fuse_before_downsample)



            ########## 后续部分保持原样 ##########
                    
            # 图像分支的上采样
            for i, up in enumerate(self.up_path_1):
                # 原定义：self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
                # 原定义：self.up_path_2.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
                
                # 对 x1 进行上采样
                residual_image = up(residual_image, self.skip_conv_1[i](encs[-i-1])) # 使用跳跃连接（skip_conv_1[i]）将编码器阶段的特征 encs 与上采样的特征融合。
                # decs.append(x1) # 将每个上采样后的特征图 x1 存入 decs，用于后续解码器阶段的计算
            
            # residual_image = residual_image + residual_image_prev


        out1 = self.last(residual_image) + image

        return [out1]

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

class EventImage_ConcatFusionBlock(nn.Module):
    def __init__(self, dim, relu_slope):
        super(EventImage_ConcatFusionBlock, self).__init__()

        self.identity = nn.Conv2d(dim, dim, 1, 1, 0)

        # 基础卷积层定义
        self.conv_1 = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        # 通道拼接后进行1x1卷积融合，输出仍是dim通道
        # self.merge_conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

    def forward(self, image, event):
        # image: b, c, h, w
        # event: b, c, h, w
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c, h, w = image.shape

        # LayerNorm + concat + 1x1 conv 融合
        # image_norm = self.norm_image(image)
        # event_norm = self.norm_event(event)
        concat = torch.cat([image, event], dim=1)  # b, 2c, h, w
        # fused = self.merge_conv(concat)  # b, c, h, w
        fused = self.relu_1(self.conv_1(concat))
        fused = self.relu_2(self.conv_2(fused))
        fused = fused + self.identity(image)

        # MLP
        # fused = to_3d(fused)             # b, h*w, c
        # fused = fused + self.ffn(self.norm2(fused))
        # fused = to_4d(fused, h, w)       # b, c, h, w

        return fused


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False, num_heads=None, use_tsd=True):
        super(UNetConvBlock, self).__init__()
        self.use_tsd = use_tsd
        
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc
        self.num_heads = num_heads

        # 基础卷积层定义
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)        

        # 下采样层定义
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        # Transformer和特征融合层定义
        if self.num_heads is not None:
            # 修改点1：
            # 初始化通道注意力Transformer模块
            self.image_event_transformer_td = EventImage_ChannelAttentionTransformerBlock(out_size, num_heads=self.num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
            self.image_event_transformer_sd = EventImage_ChannelAttentionTransformerBlock(out_size, num_heads=self.num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
            # self.image_event_transformer_td = EventImage_ConcatFusionBlock(out_size, relu_slope)
            # self.image_event_transformer_sd = EventImage_ConcatFusionBlock(out_size, relu_slope)
            # self.image_event_transformer_sd = EventImage_ConcatFusionBlock(out_size, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')

    def forward(self, x, td_filter=None, sd_filter=None, merge_before_downsample=True):
        # 基础卷积处理流程
        out = self.conv_1(x)
        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))
        out = out_conv2 + self.identity(x)  # 残差连接

        # 修改点2：
        if merge_before_downsample:
            if td_filter is not None:
                out = self.image_event_transformer_td(out, td_filter) # 先 TD + 图像
            if sd_filter is not None:
                out = self.image_event_transformer_sd(out, sd_filter) # 再 SD + 上一步结果

        # 下采样分支处理
        if self.downsample:
            out_down = self.downsample(out)
            
            # 修改点3：
            if not merge_before_downsample:
                if td_filter is not None:
                    out = self.image_event_transformer_td(out, td_filter) # 先 TD + 图像
                if sd_filter is not None:
                    out = self.image_event_transformer_sd(out, sd_filter) # 再 SD + 上一步结果

            return out_down, out
        # 非下采样分支处理
        else:
            if merge_before_downsample:
                return out
            else:
                # 修改点4：
                if td_filter is not None:
                    out = self.image_event_transformer_td(out, td_filter) # 先 TD + 图像
                if sd_filter is not None:
                    out = self.image_event_transformer_sd(out, sd_filter) # 再 SD + 上一步结果
                return out





########## 剩余U-net都没变 ##########
class UNetEVConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False):
        super(UNetEVConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_before_merge = nn.Conv2d(out_size, out_size , 1, 1, 0) 
        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, merge_before_downsample=True):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)
             
        if self.downsample:

            out_down = self.downsample(out)
            
            if not merge_before_downsample: 
                # 如果 merge after downsample, 就要对out_down进行卷积
                out_down = self.conv_before_merge(out_down)
            else : 
                # 如果 merge before downsample, 就要对out进行卷积
                out = self.conv_before_merge(out)
            return out_down, out

        else:

            out = self.conv_before_merge(out)
            return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


if __name__ == "__main__":
    pass