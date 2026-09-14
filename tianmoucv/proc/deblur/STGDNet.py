"""Standalone current STGDNet inference wrapper for Tianmouc captures."""

from __future__ import annotations

import glob
import math
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import torch
from torch import Tensor, nn


DEFAULT_DEBLUR_URL = (
    "http://www.tianmouc.cn:38328/index.php/s/NeEd8X3Z72oDTXN/download/"
    "Generalized_Model.pth"
)
DEFAULT_DEBLUR_NAME = "Generalized_Model.pth"


def resolve_checkpoint(
    checkpoint: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    source = str(checkpoint or DEFAULT_DEBLUR_URL)
    parsed = urlparse(source)
    if parsed.scheme not in ("http", "https"):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Deblur checkpoint not found: {path}")
        return path
    root = (
        Path(cache_dir).expanduser()
        if cache_dir
        else Path.home() / ".cache" / "tianmoucv" / "checkpoints"
    )
    root.mkdir(parents=True, exist_ok=True)
    target = root / (Path(parsed.path).name or DEFAULT_DEBLUR_NAME)
    if target.is_file() and target.stat().st_size > 0:
        return target
    partial = target.with_suffix(target.suffix + ".part")
    print(f"Downloading STGDNet weights to {target}")
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


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def _to_3d(x: Tensor) -> Tensor:
    return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


def _to_4d(x: Tensor, height: int, width: int) -> Tensor:
    return x.reshape(x.shape[0], height, width, x.shape[2]).permute(0, 3, 1, 2)


class WithBiasLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.body = WithBiasLayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), height, width)


class MutualAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, 1, bias=bias)
        self.k = nn.Conv2d(dim, dim, 1, bias=bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        head_channels = channels // self.num_heads
        q = self.q(x).reshape(batch, self.num_heads, head_channels, -1)
        k = self.k(y).reshape(batch, self.num_heads, head_channels, -1)
        v = self.v(y).reshape(batch, self.num_heads, head_channels, -1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1) * self.temperature).softmax(dim=-1)
        output = (attention @ v).reshape(batch, channels, height, width)
        return self.project_out(output)


class TSDImageChannelAttentionTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1_image = LayerNorm(dim)
        self.norm1_tsd = LayerNorm(dim)
        self.attn = MutualAttention(dim, num_heads, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = Mlp(dim, dim * 2)

    def forward(self, image: Tensor, tsd: Tensor) -> Tensor:
        fused = image + self.attn(self.norm1_image(image), self.norm1_tsd(tsd))
        batch, _, height, width = fused.shape
        fused_3d = _to_3d(fused)
        return _to_4d(fused_3d + self.ffn(self.norm2(fused_3d)), height, width)


def _conv3x3(in_channels: int, out_channels: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)


class SAM(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = _conv3x3(features, features)
        self.conv2 = _conv3x3(features, 3)
        self.conv3 = _conv3x3(3, features)
        self.counter = 0

    def forward(self, x: Tensor, image: Tensor) -> Tuple[Tensor, Tensor]:
        x1 = self.conv1(x)
        restored = self.conv2(x) + image
        x1 = x1 * torch.sigmoid(self.conv3(restored)) + x
        return x1, restored


class UNetConvBlock(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        downsample: bool,
        slope: float,
        num_heads: Optional[int] = None,
    ):
        super().__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, 3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, 3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(slope, inplace=False)
        self.downsample = (
            nn.Conv2d(out_size, out_size, 4, 2, 1, bias=False) if downsample else False
        )
        self.num_heads = num_heads
        self.use_tsd = True
        if num_heads is not None:
            self.image_td_transformer = TSDImageChannelAttentionTransformerBlock(out_size, num_heads)
            self.image_sd_transformer = TSDImageChannelAttentionTransformerBlock(out_size, num_heads)

    def forward(
        self,
        x: Tensor,
        td_filter: Optional[Tensor] = None,
        sd_filter: Optional[Tensor] = None,
        merge_before_downsample: bool = True,
    ):
        output = self.relu_2(self.conv_2(self.relu_1(self.conv_1(x)))) + self.identity(x)
        if merge_before_downsample:
            if td_filter is not None:
                output = self.image_td_transformer(output, td_filter)
            if sd_filter is not None:
                output = self.image_sd_transformer(output, sd_filter)
        if self.downsample:
            down = self.downsample(output)
            if not merge_before_downsample:
                if td_filter is not None:
                    output = self.image_td_transformer(output, td_filter)
                if sd_filter is not None:
                    output = self.image_sd_transformer(output, sd_filter)
            return down, output
        if not merge_before_downsample:
            if td_filter is not None:
                output = self.image_td_transformer(output, td_filter)
            if sd_filter is not None:
                output = self.image_sd_transformer(output, sd_filter)
        return output


class UNetEVConvBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, downsample: bool, slope: float):
        super().__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, 3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, 3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(slope, inplace=False)
        self.conv_before_merge = nn.Conv2d(out_size, out_size, 1, 1, 0)
        self.downsample = (
            nn.Conv2d(out_size, out_size, 4, 2, 1, bias=False) if downsample else False
        )

    def forward(self, x: Tensor, merge_before_downsample: bool = True):
        output = self.relu_2(self.conv_2(self.relu_1(self.conv_1(x)))) + self.identity(x)
        if self.downsample:
            down = self.downsample(output)
            if merge_before_downsample:
                output = self.conv_before_merge(output)
            else:
                down = self.conv_before_merge(down)
            return down, output
        return self.conv_before_merge(output)


class UNetUpBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, slope: float):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, 2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, slope)

    def forward(self, x: Tensor, bridge: Tensor) -> Tensor:
        return self.conv_block(torch.cat((self.up(x), bridge), dim=1))


class STGDNet(nn.Module):
    def __init__(
        self,
        in_chn: int = 3,
        td_chn: int = 12,
        sd_chn: int = 2,
        wf: int = 48,
        depth: int = 3,
        fuse_before_downsample: bool = True,
        relu_slope: float = 0.2,
        num_heads: Sequence[int] = (1, 2, 4),
    ):
        super().__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = list(num_heads)
        self.down_path_1 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.residual_image_init = nn.Parameter(torch.zeros(1, wf, 320, 640))
        self.down_path_td = nn.ModuleList()
        self.conv_td1 = nn.Conv2d(1, wf, 3, 1, 1)
        self.down_path_sd = nn.ModuleList()
        self.conv_sd1 = nn.Conv2d(sd_chn, wf, 3, 1, 1)
        previous = wf
        for i in range(depth):
            downsample = i + 1 < depth
            image_in = previous * 2 if i == 0 else previous
            out_channels = (2**i) * wf
            self.down_path_1.append(
                UNetConvBlock(image_in, out_channels, downsample, relu_slope, self.num_heads[i])
            )
            self.down_path_td.append(UNetEVConvBlock(previous, out_channels, downsample, relu_slope))
            self.down_path_sd.append(UNetEVConvBlock(previous, out_channels, downsample, relu_slope))
            previous = out_channels
        self.up_path_1 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(previous, (2**i) * wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2**i) * wf, (2**i) * wf, 3, 1, 1))
            previous = (2**i) * wf
        self.sam12 = SAM(previous)
        self.last = _conv3x3(previous, in_chn)

    def forward(self, x: Tensor, td: Tensor, sd: Tensor):
        if x.shape[-2:] != self.residual_image_init.shape[-2:]:
            raise ValueError(f"STGDNet expects 320x640 input, got {tuple(x.shape[-2:])}")
        image_features = self.conv_01(x)
        residual = self.residual_image_init.repeat(x.shape[0], 1, 1, 1)
        sd_features = []
        sd_current = self.conv_sd1(sd)
        for i, down in enumerate(self.down_path_sd):
            result = down(sd_current, self.fuse_before_downsample)
            if i < self.depth - 1:
                sd_current, up = result
                sd_features.append(up if self.fuse_before_downsample else sd_current)
            else:
                sd_current = result
                sd_features.append(sd_current)
        for t in range(td.shape[1]):
            td_features = []
            td_current = self.conv_td1(td[:, t : t + 1])
            for i, down in enumerate(self.down_path_td):
                result = down(td_current, self.fuse_before_downsample)
                if i < self.depth - 1:
                    td_current, up = result
                    td_features.append(up if self.fuse_before_downsample else td_current)
                else:
                    td_current = result
                    td_features.append(td_current)
            residual, _ = self.sam12(residual, x)
            residual = torch.cat((image_features, residual), dim=1)
            skips = []
            for i, down in enumerate(self.down_path_1):
                result = down(
                    residual,
                    td_filter=td_features[i],
                    sd_filter=sd_features[i],
                    merge_before_downsample=self.fuse_before_downsample,
                )
                if i + 1 < self.depth:
                    residual, up = result
                    skips.append(up)
                else:
                    residual = result
            for i, up in enumerate(self.up_path_1):
                residual = up(residual, self.skip_conv_1[i](skips[-i - 1]))
        return [self.last(residual) + x]


class TianmoucDeblurrer:
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
        data = torch.load(path, map_location="cpu", weights_only=False)
        state = data.get("params", data.get("params_ema", data))
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.model = STGDNet(wf=48, fuse_before_downsample=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.net_g = self.model
        self.checkpoint_path = path

    def set_device(self, device: Union[str, torch.device]) -> "TianmoucDeblurrer":
        """Move the network after construction (legacy TianMouCV API)."""
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def to(self, device: Union[str, torch.device]) -> "TianmoucDeblurrer":
        """Mirror the ``nn.Module.to`` spelling used by downstream callers."""
        return self.set_device(device)

    def eval(self) -> "TianmoucDeblurrer":
        self.model.eval()
        return self

    @torch.inference_mode()
    def __call__(
        self,
        frame: Union[Tensor, np.ndarray, dict],
        td: Optional[Union[Tensor, np.ndarray]] = None,
        sd: Optional[Union[Tensor, np.ndarray]] = None,
    ) -> Tensor:
        if isinstance(frame, dict):
            data = frame
            frame = data["frame"]
            td = data.get("td_signal", data.get("td_voxel"))
            sd = data.get("sd_signal", data.get("sd_voxel"))
        if td is None or sd is None:
            raise ValueError("TD and SD inputs are required.")
        image = torch.as_tensor(frame, dtype=torch.float32, device=self.device)
        td_tensor = torch.as_tensor(td, dtype=torch.float32, device=self.device)
        sd_tensor = torch.as_tensor(sd, dtype=torch.float32, device=self.device)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if td_tensor.ndim == 3:
            td_tensor = td_tensor.unsqueeze(0)
        if sd_tensor.ndim == 3:
            sd_tensor = sd_tensor.unsqueeze(0)
        return self.model(image, td_tensor, sd_tensor)[0].clamp(0.0, 1.0)

    def forward(
        self,
        frame: Union[Tensor, np.ndarray, dict],
        td: Optional[Union[Tensor, np.ndarray]] = None,
        sd: Optional[Union[Tensor, np.ndarray]] = None,
    ) -> Tensor:
        return self(frame, td=td, sd=sd)


class TianmoucDeblurNet(TianmoucDeblurrer):
    """Backward-compatible constructor for the retired deblur wrapper."""

    def __init__(
        self,
        ckpt_path: Optional[Union[str, Path]] = None,
        _optim: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        # ``_optim`` is accepted for API compatibility.  Inference mode already
        # avoids autograd overhead and does not force torch.compile on users.
        del _optim
        super().__init__(checkpoint=ckpt_path, device=device, cache_dir=cache_dir)


_EXPOSURE_PATTERN = re.compile(
    r"\bexp(?:osure)?\s*time\s*[:=]\s*"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>us|µs|μs|ms)\b",
    flags=re.IGNORECASE,
)


def read_exposure_times(sequence_path: Union[str, Path]) -> list[int]:
    """Read every per-COP exposure from the standard ``cone/info_*.txt`` file.

    Standard records look like ``Exp Time:414 us,Gain:1``.  The expression is
    deliberately tolerant of whitespace, ``Exposure Time``, ``=`` and ms/µs
    spellings so camera-firmware formatting changes do not silently break the
    deblur input window.
    """
    files = [
        Path(name)
        for name in sorted(glob.glob(str(Path(sequence_path) / "cone" / "*.txt")))
        if ".ipynb_checkpoints" not in name
    ]
    if not files:
        raise FileNotFoundError(f"No exposure metadata txt found under {Path(sequence_path) / 'cone'}")
    exposures = []
    for filename in files:
        with filename.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = _EXPOSURE_PATTERN.search(line)
                if match:
                    value = float(match.group("value"))
                    if match.group("unit").lower() == "ms":
                        value *= 1000.0
                    exposures.append(int(round(value)))
    if not exposures:
        raise ValueError(f"Could not parse an exposure record from {files[0]}")
    return exposures


def read_exposure_time(sequence_path: Union[str, Path], cone_index: int = 0) -> int:
    """Read the exposure corresponding to a Tianmouc cone-frame index."""
    exposures = read_exposure_times(sequence_path)
    if not 0 <= cone_index < len(exposures):
        raise IndexError(
            f"Cone index {cone_index} is outside the {len(exposures)} exposure records."
        )
    return exposures[cone_index]


def detect_aop_interval(sample: dict) -> int:
    timestamps = sample.get("meta", {}).get("R_timestamp", [])
    if len(timestamps) < 21:
        raise ValueError("At least 21 R_timestamp values are required.")
    measured_us = (timestamps[20] - timestamps[0]) * 10.0 / 20.0
    if 600 <= measured_us <= 700:
        return 660
    if 1200 <= measured_us <= 1400:
        return 1320
    raise ValueError(f"Unsupported AOP interval: {measured_us:.1f} us")


def infer_blur_length(
    sequence_path: Union[str, Path],
    sample: dict,
    exposure_times: Optional[Sequence[int]] = None,
) -> Tuple[int, int, int]:
    cone_indices = sample.get("meta", {}).get("C_idx", [0])
    cone_index = int(torch.as_tensor(cone_indices).flatten()[0].item())
    records = list(exposure_times) if exposure_times is not None else read_exposure_times(sequence_path)
    if not 0 <= cone_index < len(records):
        raise IndexError(
            f"Sample cone index {cone_index} is outside the {len(records)} exposure records."
        )
    exposure_us = int(records[cone_index])
    interval_us = detect_aop_interval(sample)
    blur_length = int(np.clip(math.ceil(exposure_us / interval_us), 5, 12))
    return blur_length, exposure_us, interval_us


def prepare_deblur_input(sample: dict, blur_length: int) -> dict:
    """Reproduce the current TMC-Deblur real-capture preprocessing exactly."""
    frame = torch.as_tensor(sample["F0"], dtype=torch.float32)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected F0 [H,W,3], got {tuple(frame.shape)}")
    frame = frame.flip(1).permute(2, 0, 1).contiguous()
    source = sample.get("tsdiff")
    if source is None:
        source = sample.get("txydiff")
    if source is None:
        raise KeyError("Deblur preprocessing requires upsampled 'tsdiff' (or 'txydiff').")
    tsd = torch.as_tensor(source, dtype=torch.float32)
    last_td_index = min(blur_length, tsd.shape[1] - 1)
    indices = torch.arange(2, last_td_index + 1)
    if indices.numel() == 0:
        raise ValueError(f"Not enough TSD frames for blur_length={blur_length}")
    aop_index = min((blur_length + 1) // 2, tsd.shape[1] - 1)
    td = tsd[0, indices].flip(-1).contiguous()
    sd = tsd[1:3, aop_index].flip(-1).contiguous()
    return {"frame": frame, "td_signal": td, "sd_signal": sd}


def _rgb8(image: Union[Tensor, np.ndarray]) -> np.ndarray:
    value = torch.as_tensor(image).detach().float().cpu()
    if value.ndim == 4:
        value = value[0]
    if value.ndim == 3 and value.shape[0] == 3:
        value = value.permute(1, 2, 0)
    return np.ascontiguousarray(
        np.round(value.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    )


def make_deblur_comparison(
    blurred: Union[Tensor, np.ndarray], deblurred: Union[Tensor, np.ndarray]
) -> np.ndarray:
    """Render the project's standard blurred/deblurred side-by-side comparison."""
    left, right = _rgb8(blurred), _rgb8(deblurred)
    cv2.putText(left, "Blurred F0", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right, "STGDNet deblurred", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate((left, right), axis=1)


def save_video(frames: Sequence[np.ndarray], path: Union[str, Path], fps: float = 30.0) -> Path:
    if not frames:
        raise ValueError("No frames were provided.")
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return output.resolve()


__all__ = [
    "DEFAULT_DEBLUR_URL",
    "TianmoucDeblurNet",
    "TianmoucDeblurrer",
    "detect_aop_interval",
    "infer_blur_length",
    "make_deblur_comparison",
    "prepare_deblur_input",
    "read_exposure_time",
    "read_exposure_times",
    "resolve_checkpoint",
    "save_video",
]
