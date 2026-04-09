"""Self-contained network implementations copied/adapted from CBRDM modules.

This file provides the minimal classes required by the demo: UpsampleTSDConv,
UNet encoder blocks, event fusion blocks, ThickConvGRU, TmcRGBTSDFusionBranch_Dev
and BiIterativeTSDInterpolatorFull. The implementations are adapted to remove
diffusers-specific mixins so this file can be used standalone for deployment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch, out_ch, bias=True):
    return nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias)


def conv_down(in_chn, out_chn, bias=False):
    return nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)


class UpsampleTSDConv(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
        k[..., 1, 0] = 0.25
        k[..., 1, 2] = 0.25
        k[..., 0, 1] = 0.25
        k[..., 2, 1] = 0.25
        self.register_buffer("kernel", k, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, out_hw=(320, 640), flip_lr=True):
        H, W0 = x.shape[-2], x.shape[-1]
        W = W0 * 2
        x_expand = x.new_zeros(*x.shape[:-2], H, W)
        x_expand[..., ::2, ::2] = x[..., ::2, :]
        x_expand[..., 1::2, 1::2] = x[..., 1::2, :]
        lead_shape = x_expand.shape[:-2]
        N = 1
        for s in lead_shape:
            N *= int(s)
        inp = x_expand.reshape(N, H, W).unsqueeze(1)
        k = self.kernel.to(device=inp.device, dtype=inp.dtype)
        out = F.conv2d(F.pad(inp, (1, 1, 1, 1), mode="reflect"), k, padding=0)
        res = inp.clone()
        res[..., 0:-1:2, 1:-1:2] = out[..., 0:-1:2, 1:-1:2]
        res[..., 1:-1:2, 0:-1:2] = out[..., 1:-1:2, 0:-1:2]
        if out_hw is not None:
            res = F.interpolate(res, size=(int(out_hw[0]), int(out_hw[1])), mode="bilinear", align_corners=False)
        if flip_lr:
            res = torch.flip(res, dims=[-1])
        H2, W2 = res.shape[-2], res.shape[-1]
        return res.squeeze(1).reshape(*lead_shape, H2, W2)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, num_heads=None):
        super().__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.num_heads = num_heads
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=True)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=True)
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        # For simplicity in this standalone demo, we omit the attention transformer
        self.image_event_transformer_td = None
        self.image_event_transformer_sd = None

    def forward(self, img, td_filter=None, sd_filter=None, merge_before_downsample=True):
        out_conv1 = self.relu_1(self.conv_1(img))
        out_conv2 = self.relu_2(self.conv_2(out_conv1))
        out = out_conv2 + self.identity(img)

        if merge_before_downsample and self.image_event_transformer_td is not None and td_filter is not None:
            out = self.image_event_transformer_td(out, td_filter)
        if merge_before_downsample and self.image_event_transformer_sd is not None and sd_filter is not None:
            out = self.image_event_transformer_sd(out, sd_filter)

        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample and self.image_event_transformer_td is not None and td_filter is not None:
                out_down = self.image_event_transformer_td(out_down, td_filter)
            if not merge_before_downsample and self.image_event_transformer_sd is not None and sd_filter is not None:
                out_down = self.image_event_transformer_sd(out_down, sd_filter)
            return out_down, out
        else:
            if merge_before_downsample:
                return out
            else:
                if self.image_event_transformer_td is not None and td_filter is not None:
                    out = self.image_event_transformer_td(out, td_filter)
                if self.image_event_transformer_sd is not None and sd_filter is not None:
                    out = self.image_event_transformer_sd(out, sd_filter)
                return out


class UNetEVConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope):
        super(UNetEVConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=True)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=True)
        self.conv_before_merge = nn.Conv2d(out_size, out_size, 1, 1, 0)
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, merge_before_downsample=True):
        out_conv1 = self.relu_1(self.conv_1(x))
        out_conv2 = self.relu_2(self.conv_2(out_conv1))
        out = out_conv2 + self.identity(x)
        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample:
                out_down = self.conv_before_merge(out_down)
            else:
                out = self.conv_before_merge(out)
            return out_down, out
        else:
            out = self.conv_before_merge(out)
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class SimpleUpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, relu_slope=0.2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=True)
        self.conv1 = nn.Conv2d(out_ch + skip_ch, out_ch, 3, 1, 1)
        self.act1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.act2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.id = nn.Conv2d(out_ch + skip_ch, out_ch, 1, 1, 0)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        y = self.act1(self.conv1(x))
        y = self.act2(self.conv2(y))
        return y + self.id(x)


class ThickConvGRU(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, gate_depth=2, norm="gn", relu_slope=0.2):
        super().__init__()
        in_ch = input_size + hidden_size
        def conv_block(in_ch_, out_ch_, k=3, norm="gn", act=True, relu_slope=0.2):
            pad = k // 2
            layers = [nn.Conv2d(in_ch_, out_ch_, k, padding=pad)]
            if norm == "gn":
                layers.append(nn.GroupNorm(num_groups=8, num_channels=out_ch_))
            elif norm == "bn":
                layers.append(nn.BatchNorm2d(out_ch_))
            if act:
                layers.append(nn.LeakyReLU(relu_slope, inplace=True))
            return nn.Sequential(*layers)

        self.fuse = nn.Sequential(
            conv_block(in_ch, hidden_size, kernel_size, norm=norm, act=True, relu_slope=relu_slope),
            conv_block(hidden_size, hidden_size, kernel_size, norm=norm, act=True, relu_slope=relu_slope),
        )
        self.cand_fuse = nn.Sequential(
            conv_block(in_ch, hidden_size, kernel_size, norm=norm, act=True, relu_slope=relu_slope),
            conv_block(hidden_size, hidden_size, kernel_size, norm=norm, act=True, relu_slope=relu_slope),
        )
        self.update_head = nn.Conv2d(hidden_size, hidden_size, 1)
        self.reset_head  = nn.Conv2d(hidden_size, hidden_size, 1)
        self.out_head    = nn.Conv2d(hidden_size, hidden_size, 1)

    def forward(self, input_, prev_state):
        if prev_state is None:
            prev_state = input_.new_zeros((input_.shape[0], self.reset_head.out_channels, *input_.shape[2:]))
        x = torch.cat([input_, prev_state], dim=1)
        feat = self.fuse(x)
        z = torch.sigmoid(self.update_head(feat))
        r = torch.sigmoid(self.reset_head(feat))
        cand_in = torch.cat([input_, prev_state * r], dim=1)
        cand_feat = self.cand_fuse(cand_in)
        h_tilde = torch.tanh(self.out_head(cand_feat))
        new_state = prev_state * (1 - z) + h_tilde * z
        return new_state


class TmcRGBTSDFusionBranch_Dev(nn.Module):
    def __init__(self, in_chn=3, td_chn=10, sd_chn=2, wf=32, depth=3, fuse_before_downsample=True, relu_slope=0.2, num_heads=[1,2,4]):
        super().__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.down_path_img = nn.ModuleList()
        self.conv_img_in = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.down_path_td = nn.ModuleList()
        self.conv_td_in = nn.Conv2d(td_chn, wf, 3, 1, 1)
        self.down_path_sd = nn.ModuleList()
        self.conv_sd_in = nn.Conv2d(sd_chn, wf, 3, 1, 1)
        prev_channels = wf
        for i in range(depth):
            out_channels = (2**(i+1)) * wf
            self.down_path_img.append(UNetConvBlock(prev_channels, out_channels, True, relu_slope, num_heads=self.num_heads[i] if i < len(self.num_heads) else None))
            self.down_path_td.append(UNetEVConvBlock(prev_channels, out_channels, True, relu_slope))
            self.down_path_sd.append(UNetEVConvBlock(prev_channels, out_channels, True, relu_slope))
            prev_channels = out_channels

    def forward(self, img, td, sd, add_first_output=True):
        sdiff = []
        sd_feat = self.conv_sd_in(sd)
        for i, down in enumerate(self.down_path_sd):
            sd_feat, sd_up = down(sd_feat, self.fuse_before_downsample)
            sdiff.append(sd_up if self.fuse_before_downsample else sd_feat)
        tdiff = []
        td_feat = self.conv_td_in(td)
        for i, down in enumerate(self.down_path_td):
            td_feat, td_up = down(td_feat, self.fuse_before_downsample)
            tdiff.append(td_up if self.fuse_before_downsample else td_feat)
        encs = []
        F_image = self.conv_img_in(img)
        if add_first_output:
            encs.append(F_image)
        for i, down in enumerate(self.down_path_img):
            F_image, x1_up = down(F_image, td_filter=tdiff[i], sd_filter=sdiff[i], merge_before_downsample=self.fuse_before_downsample)
            encs.append(F_image)
        return encs


class MultiScaleEventEncoder(nn.Module):
    def __init__(self, td_chn=1, sd_chn=2, wf=64, depth=3, relu_slope=0.2, fuse_before_downsample=True):
        super().__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.conv_td0 = nn.Conv2d(td_chn, wf, 3, 1, 1)
        self.conv_sd0 = nn.Conv2d(sd_chn, wf, 3, 1, 1)
        self.td_path = nn.ModuleList()
        self.sd_path = nn.ModuleList()
        prev = wf
        for i in range(depth):
            out = (2**(i+1)) * wf
            self.td_path.append(UNetEVConvBlock(prev, out, True, relu_slope))
            self.sd_path.append(UNetEVConvBlock(prev, out, True, relu_slope))
            prev = out

    def forward(self, td, sd):
        td_feat = self.conv_td0(td)
        sd_feat = self.conv_sd0(sd)
        td_feats = [td_feat]
        sd_feats = [sd_feat]
        for i in range(self.depth):
            td_feat, td_up = self.td_path[i](td_feat, self.fuse_before_downsample)
            sd_feat, sd_up = self.sd_path[i](sd_feat, self.fuse_before_downsample)
            td_feats.append(td_up if self.fuse_before_downsample else td_feat)
            sd_feats.append(sd_up if self.fuse_before_downsample else sd_feat)
        return td_feats, sd_feats


class BiIterativeTSDInterpolatorFull(nn.Module):
    def __init__(self, wf=64, depth=3, relu_slope=0.2, fuse_before_downsample=True, num_heads=[1,2,4], rgb_chn=3, td_expo_chn=10, sd_expo_chn=2, td_seq_chn=1, sd_seq_chn=2, has_up_sample=False):
        super().__init__()
        self.tsd_preprocess_conv = UpsampleTSDConv()
        self.depth = depth
        self.wf = wf
        self.fuse_before_downsample = fuse_before_downsample
        self.has_up_sample = has_up_sample
        self.rgb_tsd_encoder = TmcRGBTSDFusionBranch_Dev(in_chn=rgb_chn, td_chn=td_expo_chn, sd_chn=sd_expo_chn, wf=wf, depth=depth, fuse_before_downsample=fuse_before_downsample, relu_slope=relu_slope, num_heads=num_heads)
        self.seq_event_encoder = MultiScaleEventEncoder(td_chn=td_seq_chn, sd_chn=sd_seq_chn, wf=wf, depth=depth, relu_slope=relu_slope, fuse_before_downsample=False)
        self.f_grus = nn.ModuleList()
        self.b_grus = nn.ModuleList()
        for i in range(depth + 1):
            hidden_ch = (2**(i)) * wf
            self.f_grus.append(ThickConvGRU(input_size=hidden_ch*2, hidden_size=hidden_ch, kernel_size=3, relu_slope=relu_slope))
            self.b_grus.append(ThickConvGRU(input_size=hidden_ch*2, hidden_size=hidden_ch, kernel_size=3, relu_slope=relu_slope))
        if self.has_up_sample:
            self.bi_fuse = nn.ModuleList()
            for i in range(depth + 1):
                hidden_ch = (2**(i)) * wf
                self.bi_fuse.append(nn.Conv2d(hidden_ch * 2, hidden_ch, 1, 1, 0))
            self.up_blocks = nn.ModuleList()
            for i in reversed(range(0, depth)):
                in_ch  = (2**(i+1)) * wf
                out_ch = (2**i) * wf
                skip_ch = out_ch
                self.up_blocks.append(SimpleUpBlock(in_ch, skip_ch, out_ch, relu_slope=relu_slope))
            self.to_rgb = conv3x3(wf, 3, bias=True)

    def _encode_endpoints(self, F0, td0, sd0, F1, td1, sd1):
        F = torch.cat([F0, F1], dim=0)
        td = torch.cat([td0, td1], dim=0)
        sd = torch.cat([sd0, sd1], dim=0)
        encs = self.rgb_tsd_encoder(F, td, sd)
        B = F0.shape[0]
        encs0 = [e[:B].contiguous() for e in encs]
        encs1 = [e[B:].contiguous() for e in encs]
        return encs0, encs1

    def _encode_sequence_diffs(self, td_seq, sd_seq):
        B, C, T, H, W = sd_seq.shape
        td = td_seq.permute(0, 2, 1, 3, 4).contiguous().view(B * (T + 1), 1, H, W)
        sd = sd_seq.permute(0, 2, 1, 3, 4).contiguous().view(B * T, 2, H, W)
        td_feats_bt, sd_feats_bt = self.seq_event_encoder(td, sd)
        td_ms, sd_ms = [], []
        for i in range(len(td_feats_bt)):
            f_td = td_feats_bt[i]
            f_sd = sd_feats_bt[i]
            td_ms.append(f_td.view(B, T + 1, *f_td.shape[1:]))
            sd_ms.append(f_sd.view(B, T, *f_sd.shape[1:]))
        return td_ms, sd_ms

    def forward(self, F0, F1, td0, sd0, td1, sd1, td_seq, sd_seq):
        td0 = self.tsd_preprocess_conv(td0)
        sd0 = self.tsd_preprocess_conv(sd0)
        td1 = self.tsd_preprocess_conv(td1)
        sd1 = self.tsd_preprocess_conv(sd1)
        td_seq = self.tsd_preprocess_conv(td_seq)
        sd_seq = self.tsd_preprocess_conv(sd_seq)
        encs0, encs1 = self._encode_endpoints(F0, td0, sd0, F1, td1, sd1)
        td_ms, sd_ms = self._encode_sequence_diffs(td_seq, sd_seq)
        TT = sd_ms[0].shape[1]
        f_states = [[None for _ in range(self.depth + 1)] for _ in range(TT)]
        f_hidden = [None for _ in range(self.depth + 1)]
        for d in range(self.depth + 1):
            f_hidden[d] = encs0[d]
        for t in range(0, TT):
            for d in range(self.depth + 1):
                ev = torch.cat([td_ms[d][:, t], sd_ms[d][:, t]], dim=1)
                f_hidden[d] = self.f_grus[d](ev, f_hidden[d])
                f_states[t][d] = f_hidden[d]
        b_states = [[None for _ in range(self.depth + 1)] for _ in range(TT)]
        b_hidden = [None for _ in range(self.depth + 1)]
        for d in range(self.depth + 1):
            b_hidden[d] = encs1[d]
        for t in reversed(range(TT)):
            for d in range(self.depth + 1):
                ev = torch.cat([td_ms[d][:, t + 1], sd_ms[d][:, t]], dim=1)
                b_hidden[d] = self.b_grus[d](ev, b_hidden[d])
                b_states[t][d] = b_hidden[d]
        fused_states = []
        for d in range(self.depth + 1):
            fused = []
            for t in range(TT):
                bi = torch.cat([f_states[t][d], b_states[t][d]], dim=1)
                fused.append(bi)
            fused = torch.stack(fused, dim=2)
            fused_states.append(fused)
        if self.has_up_sample:
            bi_fused = []
            for d in range(self.depth + 1):
                f_i = torch.stack([f_states[t][d] for t in range(TT)], dim=1)
                b_i = torch.stack([b_states[t][d] for t in range(TT)], dim=1)
                bi_i = torch.cat([f_i, b_i], dim=2)
                B, TT_, CC, H, W = bi_i.shape
                bi_i = bi_i.reshape(B * TT_, CC, H, W)
                fu_i = self.bi_fuse[d](bi_i)
                C2 = fu_i.shape[1]
                fu_i = fu_i.reshape(B, TT_, C2, H, W)
                bi_fused.append(fu_i)
            x = bi_fused[self.depth]
            B, TT_, Cx, Hx, Wx = x.shape
            x = x.reshape(B * TT_, Cx, Hx, Wx)
            for k, up in enumerate(self.up_blocks):
                skip_idx = self.depth - (k + 1)
                skip = bi_fused[skip_idx]
                Bs, TT_s, Cs, Hs, Ws = skip.shape
                skip = skip.reshape(B * TT_, Cs, Hs, Ws)
                x = up(x, skip)
            rgb = self.to_rgb(x)
            rgb = rgb.reshape(B, TT_, 3, rgb.shape[-2], rgb.shape[-1])
            frames = rgb.permute(0, 2, 1, 3, 4).contiguous()
            return frames, fused_states
        return fused_states
