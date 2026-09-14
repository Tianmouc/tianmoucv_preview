"""Standalone Tianmouc DDDNet inference wrapper.

This file intentionally contains the complete inference-time network.  It only
depends on packages already required by TianMouCV (torch, numpy, requests and
opencv-python) and can later be moved into ``tianmoucv.proc.denoise`` without
bringing this training repository with it.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_DENOISE_URL = (
    "http://www.tianmouc.cn:38328/index.php/s/E3JySZFb4XBHSWw/download/"
    "mix_sigma124_supervised,consistent_dual_recurrent_distill_best.ckpt"
)
DEFAULT_DENOISE_NAME = (
    "mix_sigma124_supervised,consistent_dual_recurrent_distill_best.ckpt"
)


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "tianmoucv" / "checkpoints"


def resolve_checkpoint(
    checkpoint: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve a local checkpoint or download a URL into TianMouCV's cache."""
    source = str(checkpoint or DEFAULT_DENOISE_URL)
    parsed = urlparse(source)
    if parsed.scheme not in ("http", "https"):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Denoise checkpoint not found: {path}")
        return path

    root = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    filename = Path(parsed.path).name or DEFAULT_DENOISE_NAME
    target = root / filename
    if target.is_file() and target.stat().st_size > 0:
        return target

    partial = target.with_suffix(target.suffix + ".part")
    print(f"Downloading DDDNet weights to {target}")
    try:
        with requests.get(source, stream=True, timeout=60) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return target


class ConvGRUCellAsym(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        if input_dim == 3:
            self.cross_attn_cbam = nn.Sequential(
                nn.Conv2d(2, 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, 1),
                nn.Sigmoid(),
            )
            self.td_conv = nn.Conv2d(1, hidden_dim // 4, kernel_size, padding=padding)
            self.sd_conv = nn.Conv2d(2, hidden_dim - hidden_dim // 4, kernel_size, padding=padding)
            self.input_feat_dim = hidden_dim
        else:
            self.input_feat_dim = input_dim
        self.conv_gates = nn.Conv2d(
            self.input_feat_dim + hidden_dim,
            2 * hidden_dim,
            kernel_size,
            padding=padding,
            bias=bias,
        )
        self.conv_candidate = nn.Conv2d(
            self.input_feat_dim + hidden_dim,
            hidden_dim,
            kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, input_t: Tensor, h_prev: Tensor) -> Tensor:
        if self.input_dim == 3:
            td, sd = input_t[:, :1], input_t[:, 1:3]
            td_feat = self.td_conv(td * self.cross_attn_cbam(sd))
            sd_feat = self.sd_conv(sd)
            input_feat = torch.cat((td_feat, sd_feat), dim=1)
        else:
            input_feat = input_t
        combined = torch.cat((input_feat, h_prev), dim=1)
        gamma_u, gamma_r = self.conv_gates(combined).chunk(2, dim=1)
        gamma_u, gamma_r = torch.sigmoid(gamma_u), torch.sigmoid(gamma_r)
        candidate = torch.tanh(
            self.conv_candidate(torch.cat((input_feat, gamma_r * h_prev), dim=1))
        )
        return gamma_u * h_prev + (1.0 - gamma_u) * candidate


HiddenState = List[Tensor]


class ConvGRUTD(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        num_layers: int = 2,
        bias: bool = False,
        use_fpn: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_fpn = use_fpn
        self.cell_list = nn.ModuleList(
            ConvGRUCellAsym(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                kernel_size,
                bias,
            )
            for i in range(num_layers)
        )
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, 1, bias=bias)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.channel_fpn_refiners = None
        if use_fpn:
            fpn_hidden = max(8, hidden_channels // 4)
            self.channel_fpn_refiners = nn.ModuleList(
                nn.Sequential(
                    nn.Conv2d(2, fpn_hidden, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(fpn_hidden, fpn_hidden, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(fpn_hidden, 1, 3, padding=1),
                )
                for _ in range(in_channels)
            )

    def forward(
        self, x: Tensor, hidden_states: Optional[HiddenState] = None
    ) -> Tuple[Tensor, HiddenState]:
        batch, channels, time, height, width = x.shape
        if hidden_states is None:
            hidden = [
                x.new_zeros(batch, self.hidden_channels, height, width)
                for _ in range(self.num_layers)
            ]
            mean_running = x.mean(dim=2)
            count = torch.full((batch,), time, device=x.device, dtype=torch.float32)
            current_segment_avg = mean_running
        else:
            hidden = list(hidden_states[: self.num_layers])
            mean_running = hidden_states[self.num_layers]
            count = hidden_states[self.num_layers + 1]
            current_segment_avg = x.mean(dim=2)
            old_count = count.view(batch, 1, 1, 1)
            mean_running = (
                mean_running * old_count + current_segment_avg * time
            ) / (old_count + time)
            count = count + time

        if self.use_fpn:
            estimates = []
            assert self.channel_fpn_refiners is not None
            for channel, refiner in enumerate(self.channel_fpn_refiners):
                prior = mean_running[:, channel : channel + 1]
                current = current_segment_avg[:, channel : channel + 1]
                estimates.append(prior + refiner(torch.cat((prior, current), dim=1)))
            x = x - torch.cat(estimates, dim=1).unsqueeze(2)

        for t in range(time):
            layer_input = x[:, :, t]
            for layer, cell in enumerate(self.cell_list):
                hidden[layer] = cell(layer_input, hidden[layer])
                layer_input = hidden[layer]
        output = self.relu(self.conv_out(hidden[-1]))
        return output, hidden + [mean_running, count]


class SmallUNetLayers(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_dim: int):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_dim, base_dim, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_dim * 2, base_dim * 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.down2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_dim * 4, base_dim * 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim * 2, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_dim * 6, base_dim * 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_dim * 2, base_dim, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(base_dim, base_dim, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_dim * 3, base_dim, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_dim, out_channels, 3, 1, 1),
        )


class DualUNetNet(nn.Module):
    def __init__(self, embed_dim_cross: int = 48):
        super().__init__()
        self.td_layers = SmallUNetLayers(1, 1, embed_dim_cross)
        self.sd_layers = SmallUNetLayers(2, 2, embed_dim_cross)

    def forward(self, x: Tensor) -> Tensor:
        td_in, sd_in = x[:, :1], x[:, 1:3]
        td_s1, sd_s1 = self.td_layers.enc1(td_in), self.sd_layers.enc1(sd_in)
        td_s2 = self.td_layers.enc2(self.td_layers.down1(td_s1))
        sd_s2 = self.sd_layers.enc2(self.sd_layers.down1(sd_s1))
        td_b = self.td_layers.bottleneck(self.td_layers.down2(td_s2))
        sd_b = self.sd_layers.bottleneck(self.sd_layers.down2(sd_s2))
        td_u1, sd_u1 = self.td_layers.up1(td_b), self.sd_layers.up1(sd_b)
        if td_u1.shape[-2:] != td_s2.shape[-2:]:
            td_u1 = F.interpolate(td_u1, td_s2.shape[-2:], mode="bilinear", align_corners=False)
        if sd_u1.shape[-2:] != sd_s2.shape[-2:]:
            sd_u1 = F.interpolate(sd_u1, sd_s2.shape[-2:], mode="bilinear", align_corners=False)
        td_d1 = self.td_layers.dec1(torch.cat((td_u1, td_s2, sd_u1), dim=1))
        sd_d1 = self.sd_layers.dec1(torch.cat((sd_u1, sd_s2, td_u1), dim=1))
        td_u2, sd_u2 = self.td_layers.up2(td_d1), self.sd_layers.up2(sd_d1)
        if td_u2.shape[-2:] != td_s1.shape[-2:]:
            td_u2 = F.interpolate(td_u2, td_s1.shape[-2:], mode="bilinear", align_corners=False)
        if sd_u2.shape[-2:] != sd_s1.shape[-2:]:
            sd_u2 = F.interpolate(sd_u2, sd_s1.shape[-2:], mode="bilinear", align_corners=False)
        noise_td = self.td_layers.dec2(torch.cat((td_u2, td_s1, sd_u2), dim=1))
        noise_sd = self.sd_layers.dec2(torch.cat((sd_u2, sd_s1, td_u2), dim=1))
        return torch.cat((noise_td, noise_sd), dim=1)


class DDDNetStage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser_temporal = ConvGRUTD()

    def forward(
        self, raw_tsd: Tensor, hidden_state: Optional[HiddenState] = None
    ) -> Tuple[Tensor, HiddenState]:
        if raw_tsd.ndim == 4:
            raw_tsd = raw_tsd.unsqueeze(1)
        frames = []
        for t in range(raw_tsd.shape[1]):
            noise, hidden_state = self.denoiser_temporal(
                raw_tsd[:, t : t + 1].permute(0, 2, 1, 3, 4), hidden_state
            )
            frames.append(raw_tsd[:, t] - noise)
        return torch.stack(frames, dim=1), hidden_state


class DDDNetStage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dual_unet_net = DualUNetNet()

    def forward(self, x: Tensor) -> Tensor:
        batch, time, channels, height, width = x.shape
        flat = x.reshape(batch * time, channels, height, width)
        return (flat - self.dual_unet_net(flat)).view(batch, time, channels, height, width)


class DDDNetInference(nn.Module):
    """Inference-only student network; training-time distillation teachers omitted."""

    def __init__(self):
        super().__init__()
        # Attribute names intentionally match the training checkpoint.
        self.DDDNet_stage1 = DDDNetStage1()
        self.DDDNet_stage2 = DDDNetStage2()

    def forward(
        self, x: Tensor, hidden_state: Optional[HiddenState] = None
    ) -> Tuple[Tensor, HiddenState]:
        x, hidden_state = self.DDDNet_stage1(x, hidden_state)
        return self.DDDNet_stage2(x), hidden_state


def tmc_to_physical_grid(raw_tsd: Tensor) -> Tuple[Tensor, Tensor]:
    """Map [TD, SDX, SDY] from TMC layout to the H x 2W physical grid."""
    batch, _, time, height, width = raw_tsd.shape
    td, sdl, sdr = raw_tsd[:, 0], raw_tsd[:, 1], raw_tsd[:, 2]
    grid = raw_tsd.new_zeros(batch, 3, time, height, 2 * width)
    masks = raw_tsd.new_zeros(batch, 3, 1, height, 2 * width)
    grid[:, 0, :, 0::2, 0::2] = td[:, :, 0::2]
    grid[:, 0, :, 1::2, 1::2] = td[:, :, 1::2]
    grid[:, 1, :, 0::2, 0::2] = sdl[:, :, 0::2]
    grid[:, 1, :, 1::2, 1::2] = -sdr[:, :, 1::2]
    grid[:, 2, :, 0::2, 1::2] = sdr[:, :, 0::2]
    grid[:, 2, :, 1::2, 0::2] = -sdl[:, :, 1::2]
    masks[:, 0, :, 0::2, 0::2] = 1
    masks[:, 0, :, 1::2, 1::2] = 1
    masks[:, 1, :, 0::2, 0::2] = 1
    masks[:, 1, :, 1::2, 1::2] = 1
    masks[:, 2, :, 0::2, 1::2] = 1
    masks[:, 2, :, 1::2, 0::2] = 1
    kernel = raw_tsd.new_tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    flat = grid.reshape(-1, 1, height, 2 * width)
    mask = masks.expand(-1, -1, time, -1, -1).reshape_as(flat)
    interp = F.conv2d(flat, kernel, padding=1) / (F.conv2d(mask, kernel, padding=1) + 1e-8)
    grid = (flat * mask + interp * (1 - mask)).view_as(grid)
    return grid, masks.squeeze(2)


def physical_grid_to_tmc(physical: Tensor, height: int, width: int) -> Tensor:
    """Map [B,T,3,H,2W] physical-grid output back to [B,3,T,H,W]."""
    p = physical.permute(0, 2, 1, 3, 4)
    td, sdx, sdy = p[:, 0], p[:, 1], p[:, 2]
    output = physical.new_zeros(physical.shape[0], 3, physical.shape[1], height, width)
    output[:, 0, :, 0::2] = td[:, :, 0::2, 0::2]
    output[:, 0, :, 1::2] = td[:, :, 1::2, 1::2]
    output[:, 1, :, 0::2] = sdx[:, :, 0::2, 0::2]
    output[:, 1, :, 1::2] = -sdy[:, :, 1::2, 0::2]
    output[:, 2, :, 0::2] = sdy[:, :, 0::2, 1::2]
    output[:, 2, :, 1::2] = -sdx[:, :, 1::2, 1::2]
    return output


def sample_to_tsd(sample: dict, scale: float = 128.0) -> Tensor:
    """Read normalized [3,T,H,W] TSD from a TianmoucDataReader sample."""
    if sample.get("rawDiff") is not None:
        return torch.as_tensor(sample["rawDiff"], dtype=torch.float32) / scale
    if sample.get("tsdiff") is not None:
        tsd = torch.as_tensor(sample["tsdiff"], dtype=torch.float32)
        if tsd.ndim == 4 and tsd.shape[0] != 3 and tsd.shape[-1] == 3:
            tsd = tsd.permute(3, 0, 1, 2)
        return tsd
    raise KeyError("Sample has neither 'rawDiff' nor 'tsdiff'.")


class TianmoucDenoiser:
    """High-level stateful DDDNet wrapper for normalized Tianmouc TSD."""

    def __init__(
        self,
        checkpoint: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        path = resolve_checkpoint(checkpoint, cache_dir)
        checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint_data.get("state_dict_denoiser", checkpoint_data)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        state = {
            k: v
            for k, v in state.items()
            if k.startswith("DDDNet_stage1.") or k.startswith("DDDNet_stage2.")
        }
        self.model = DDDNetInference()
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.state: Optional[HiddenState] = None
        self.checkpoint_path = path

    def reset_state(self) -> None:
        self.state = None

    @torch.inference_mode()
    def __call__(
        self,
        raw_tsd: Union[Tensor, np.ndarray],
        state: Optional[HiddenState] = None,
        keep_state: bool = True,
    ) -> dict:
        raw = torch.as_tensor(raw_tsd, dtype=torch.float32, device=self.device)
        if raw.ndim == 4:
            raw = raw.unsqueeze(0)
        if raw.ndim != 5 or raw.shape[1] != 3:
            raise ValueError(f"Expected [3,T,H,W] or [B,3,T,H,W], got {tuple(raw.shape)}")
        batch, _, _, height, width = raw.shape
        active_state = self.state if state is None else state
        if active_state is not None and active_state[0].shape[0] != batch:
            active_state = None
        physical, masks = tmc_to_physical_grid(raw)
        denoised_physical, next_state = self.model(
            physical.permute(0, 2, 1, 3, 4).contiguous(), active_state
        )
        denoised_tsd = physical_grid_to_tmc(denoised_physical, height, width)
        if keep_state:
            self.state = next_state
        return {
            "denoised_tsd": denoised_tsd,
            # Kept for code written against the retired TorchScript wrapper.
            "denoised_tmc": denoised_tsd,
            "denoised_physical": denoised_physical,
            "state": next_state,
            "masks": masks,
        }

    def forward(
        self,
        raw_tsd: Union[Tensor, np.ndarray],
        state: Optional[HiddenState] = None,
        keep_state: bool = True,
    ) -> dict:
        """Compatibility spelling for callers that invoked ``forward`` directly."""
        return self(raw_tsd, state=state, keep_state=keep_state)


class DualRecurrentDenoiserPT(TianmoucDenoiser):
    """Backward-compatible name for the former TorchScript wrapper.

    The implementation now loads the current DDDNet training checkpoint and
    strips its training-only teacher branches instead of loading the retired
    traced model.
    """

    DEFAULT_URL = DEFAULT_DENOISE_URL

    def __init__(
        self,
        ckpt_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = "cuda:0",
        cache_dir: Optional[Union[str, Path]] = None,
        *,
        checkpoint: Optional[Union[str, Path]] = None,
    ):
        if ckpt_path is not None and checkpoint is not None:
            raise ValueError("Pass only one of ckpt_path or checkpoint.")
        if device is not None and str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        super().__init__(
            checkpoint=checkpoint if checkpoint is not None else ckpt_path,
            device=device,
            cache_dir=cache_dir,
        )

    tmc_to_physical_grid = staticmethod(tmc_to_physical_grid)

    @staticmethod
    def physical_grid_to_tmc(
        sd_lu_dr: Tensor,
        sd_dl_ur: Tensor,
        td_grid: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        physical = torch.stack((td_grid, sd_lu_dr, sd_dl_ur), dim=2)
        return physical_grid_to_tmc(physical, height, width)


def signed_signal_to_rgb(signal: Union[Tensor, np.ndarray], gain: float = 8.0) -> np.ndarray:
    """Render a normalized signed signal with positive=green, negative=blue."""
    data = torch.as_tensor(signal).detach().float().cpu().numpy()
    positive = np.clip(data * gain, 0.0, 1.0)
    negative = np.clip(-data * gain, 0.0, 1.0)
    rgb = np.stack((np.zeros_like(data), positive, negative), axis=-1)
    return np.round(rgb * 255.0).astype(np.uint8)


def make_tsd_comparison(
    raw_tsd: Union[Tensor, np.ndarray],
    denoised_tsd: Union[Tensor, np.ndarray],
    frame_index: int,
    gain: float = 8.0,
    display_frame_index: Optional[int] = None,
) -> np.ndarray:
    """Create the requested 3-row x 2-column raw/denoised TSD RGB matrix.

    ``frame_index`` selects a frame from the current reader sample. Pass a
    global ``display_frame_index`` to keep labels continuous across samples.
    """
    raw = torch.as_tensor(raw_tsd).detach().cpu()
    denoised = torch.as_tensor(denoised_tsd).detach().cpu()
    if raw.ndim == 5:
        raw = raw[0]
    if denoised.ndim == 5:
        denoised = denoised[0]
    t = frame_index % raw.shape[1]
    label_t = t if display_frame_index is None else display_frame_index
    names = ("TD", "SDX", "SDY")
    rows = []
    for channel, name in enumerate(names):
        left = signed_signal_to_rgb(raw[channel, t], gain)
        right = signed_signal_to_rgb(denoised[channel, t], gain)
        cv2.putText(left, f"Raw {name}  t={label_t}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(right, f"Denoised {name}  t={label_t}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.concatenate((left, right), axis=1))
    return np.concatenate(rows, axis=0)


def save_video(frames: Sequence[np.ndarray], path: Union[str, Path], fps: float = 30.0) -> Path:
    """Write RGB uint8 frames to an mp4 using TianMouCV's existing OpenCV dependency."""
    if not frames:
        raise ValueError("No frames were provided.")
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")
    for frame in frames:
        if frame.shape[:2] != (height, width):
            raise ValueError("All video frames must have the same size.")
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return output.resolve()


__all__ = [
    "DEFAULT_DENOISE_URL",
    "DualRecurrentDenoiserPT",
    "TianmoucDenoiser",
    "make_tsd_comparison",
    "resolve_checkpoint",
    "sample_to_tsd",
    "save_video",
    "signed_signal_to_rgb",
]
