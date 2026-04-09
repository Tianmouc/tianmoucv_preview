import os
import numpy as np
import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

# from basicsr.utils import get_root_logger

from einops import rearrange
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class UpsampleTSDConv(nn.Module):
    """
    No-parameter module. Vectorized GPU-friendly upsample for Tianmou TSD-like data.

    Input:  (..., H, W0)
    Output: (..., H, 2*W0)  then optionally interpolated to out_hw.

    Options:
      - out_hw: (out_h, out_w) e.g. (320, 640)
      - flip_lr: horizontal flip (left-right) on the final result
    """
    def __init__(self,sd_2_xy=False):
        super().__init__()
        k = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
        k[..., 1, 0] = 0.25
        k[..., 1, 2] = 0.25
        k[..., 0, 1] = 0.25
        k[..., 2, 1] = 0.25
        # no learnable params; buffer follows .to(device)
        self.sd_2_xy = sd_2_xy
        self.register_buffer("kernel", k, persistent=False)

    @torch.no_grad()  # 如果你希望它参与反传，删除这一行即可
    def forward(
        self,
        x: torch.Tensor,
        out_hw: Optional[Tuple[int, int]] = (320, 640),
        flip_lr: bool = True,
        interp_mode: str = "bilinear",
        align_corners: Optional[bool] = False,
    ) -> torch.Tensor:
        if x.dim() < 3:
            raise ValueError(f"Expect x with dim>=3 (...,H,W), got shape={tuple(x.shape)}")

        H, W0 = x.shape[-2], x.shape[-1]
        W = W0 * 2

        # 1) Expand/interleave to (..., H, 2W)
        x_expand = x.new_zeros(*x.shape[:-2], H, W)
        x_expand[..., ::2, ::2] = x[..., ::2, :]
        x_expand[..., 1::2, 1::2] = x[..., 1::2, :]

        # 2) Flatten leading dims -> N for one-shot conv2d
        lead_shape = x_expand.shape[:-2]
        # robust product without moving tensor to CPU explicitly
        N = 1
        for s in lead_shape:
            N *= int(s)

        inp = x_expand.reshape(N, H, W).unsqueeze(1)  # (N,1,H,2W)

        # 3) Reflect pad + conv once
        k = self.kernel.to(device=inp.device, dtype=inp.dtype)
        out = F.conv2d(F.pad(inp, (1, 1, 1, 1), mode="reflect"), k, padding=0)  # (N,1,H,2W)

        # 4) Fill only the missing checkerboard positions (exactly like your loop)
        res = inp.clone()
        res[..., 0:-1:2, 1:-1:2] = out[..., 0:-1:2, 1:-1:2]
        res[..., 1:-1:2, 0:-1:2] = out[..., 1:-1:2, 0:-1:2]

        # 5) Optional interpolate to target size (e.g., 320x640)
        if out_hw is not None:
            out_h, out_w = int(out_hw[0]), int(out_hw[1])
            # still (N,1,H,W) so we can directly interpolate
            res = F.interpolate(res, size=(out_h, out_w), mode=interp_mode, align_corners=align_corners)

        # 6) Optional horizontal flip on final output
        if flip_lr:
            res = torch.flip(res, dims=[-1])

        # 7) Restore shape: (..., H, 2W) or (..., out_h, out_w)
        H2, W2 = res.shape[-2], res.shape[-1]
        return res.squeeze(1).reshape(*lead_shape, H2, W2)
    

