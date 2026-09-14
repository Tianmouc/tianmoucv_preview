from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
from typing import Sequence

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import requests
import torch


DEFAULT_TUTORIAL_SHARE_URL = (
    "http://www.tianmouc.cn:38328/index.php/s/2ptYY27g3eRMydG/download/fishe8.7z"
)
DEFAULT_TUTORIAL_ARCHIVE_NAME = "fishe8.7z"
DEFAULT_TUTORIAL_SAMPLE_NAME = "fishe8"


@dataclass(frozen=True)
class TutorialDatasetInfo:
    data_home: Path
    archive_path: Path
    extract_root: Path
    sample_dir: Path
    sample_name: str
    share_url: str


def _expand_path(path: str | Path | None) -> Path:
    if path is None:
        return Path.home() / ".cache" / "tianmoucv" / "tutorial_data"
    return Path(path).expanduser().resolve()


def _find_extractor() -> tuple[str, str | None]:
    for binary in ("bsdtar", "7z", "7za", "7zr"):
        resolved = shutil.which(binary)
        if resolved:
            return binary, resolved
    try:
        import py7zr  # noqa: F401

        return "py7zr", None
    except ImportError:
        return "", None


def _download_file(url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(output_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)


def _extract_archive(archive_path: Path, extract_root: Path) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    extractor, binary = _find_extractor()
    if extractor == "bsdtar":
        subprocess.run(
            [binary, "-xf", str(archive_path), "-C", str(extract_root)],
            check=True,
        )
        return
    if extractor in {"7z", "7za", "7zr"}:
        subprocess.run(
            [binary, "x", str(archive_path), f"-o{extract_root}", "-y"],
            check=True,
        )
        return
    if extractor == "py7zr":
        import py7zr

        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            archive.extractall(path=extract_root)
        return
    raise RuntimeError(
        "未找到可用的 7z 解压工具。请安装 `bsdtar` / `7z`，"
        "或在 Python 环境中安装 `py7zr`。"
    )


def ensure_tutorial_dataset(
    root_dir: str | Path | None = None,
    *,
    share_url: str = DEFAULT_TUTORIAL_SHARE_URL,
    archive_name: str = DEFAULT_TUTORIAL_ARCHIVE_NAME,
    sample_name: str = DEFAULT_TUTORIAL_SAMPLE_NAME,
    force_download: bool = False,
    force_extract: bool = False,
) -> TutorialDatasetInfo:
    data_home = _expand_path(root_dir)
    archive_path = data_home / archive_name
    extract_root = data_home / "extracted"
    sample_dir = extract_root / sample_name

    if force_download or not archive_path.exists():
        _download_file(share_url, archive_path)

    if force_extract and extract_root.exists():
        shutil.rmtree(extract_root)

    if force_extract or not sample_dir.exists():
        _extract_archive(archive_path, extract_root)

    if not sample_dir.exists():
        raise FileNotFoundError(
            f"测试数据已下载并解压，但没有找到样本目录: {sample_dir}"
        )

    return TutorialDatasetInfo(
        data_home=data_home,
        archive_path=archive_path,
        extract_root=extract_root,
        sample_dir=sample_dir,
        sample_name=sample_name,
        share_url=share_url,
    )


def list_available_samples(data_root: str | Path) -> list[str]:
    root = Path(data_root).expanduser().resolve()
    if (root / "cone").is_dir() and (root / "rod").is_dir():
        return [root.name]

    return [
        entry.name
        for entry in sorted(root.iterdir())
        if entry.is_dir() and (entry / "cone").is_dir() and (entry / "rod").is_dir()
    ]


def normalize_to_uint8(array, *, eps: float = 1e-8) -> np.ndarray:
    if torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    array = np.asarray(array, dtype=np.float32)
    array_min = float(array.min())
    array_max = float(array.max())
    if array_max - array_min < eps:
        return np.zeros_like(array, dtype=np.uint8)
    normalized = (array - array_min) / (array_max - array_min)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)


def ensure_uint8_image(image) -> np.ndarray:
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if image.ndim == 3 and image.max() <= 1.0:
        return (image * 255.0).clip(0, 255).astype(np.uint8)
    return normalize_to_uint8(image)


def save_video(
    frames: Sequence[np.ndarray],
    output_path: str | Path,
    *,
    fps: int = 20,
    normalize: bool = False,
    rgb_input: bool = True,
) -> Path:
    if not frames:
        raise ValueError("frames 不能为空。")

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first = ensure_uint8_image(frames[0])
    if first.ndim == 2:
        first = np.repeat(first[..., None], 3, axis=2)
    height, width = first.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {output_path}")

    for frame in frames:
        frame_np = ensure_uint8_image(frame) if normalize else ensure_uint8_image(frame)
        if frame_np.ndim == 2:
            frame_np = np.repeat(frame_np[..., None], 3, axis=2)
        if frame_np.shape[:2] != (height, width):
            frame_np = cv2.resize(frame_np, (width, height))
        if rgb_input:
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        writer.write(frame_np)
    writer.release()
    return output_path
