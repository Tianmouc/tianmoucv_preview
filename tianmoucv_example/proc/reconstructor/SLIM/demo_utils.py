"""演示用的工具函数集（可视化、TSD 处理、视频写入）。

该文件包含：
- 可视化单通道差分为 RGB 的 `vizDiff_BBG`
- 从 cone 文本中解析曝光时间 `extract_exp_time_from_txt`
- 从 tsd 张量中选取指定时间点并构造 TD/SD 输入 `extract_tsd_with_select_T`
- 生成组合可视化画布的 `make_output_img`
- 将每个时间步合成为 2x2 四宫格帧的 `get_quad_frames`
- 将多批次结果拼接并写为 MP4 的 `save_concatenated_quad_mp4`

所有注释和行内注释均为中文，帮助部署/阅读人员快速理解数据格式和处理流程。
"""

from collections.abc import Sequence
import os
import re
import cv2
import numpy as np
import torch
from .networks import UpsampleTSDConv

def vizDiff_BBG(diff, thresh=0, gain=4):
    """将单通道差分映射为 3 通道视觉表示（BBG 方案）。

    功能：把一个二维差分图（范围近似 [-1,1]）转换为 RGB 显示，
    负值使用绿色/蓝色通道，正值使用红色通道，便于观察正/负变化区域。

    输入：
        diff (np.ndarray): 形状为 (H, W) 的差分图，数值应在 [-1,1] 或近似范围。
        thresh (float): 小于该绝对值的差分视为零以减少噪声显示。
        gain (float): 乘法增益用于扩展对比度。

    输出：
        np.ndarray: 形状为 (H, W, 3) 的 RGB 图像，dtype 与输入相同，但通道顺序为 HWC。
    """
    # 限制数值范围，避免极值影响显示
    diff = np.clip(diff, -1, 1)
    h, w = diff.shape
    rgb_diff = np.zeros((3, h, w), dtype=diff.dtype)
    # 将接近零的值抑制以减少噪声显示
    diff[np.abs(diff) < thresh] = 0
    # 放大对比并再次裁剪到 [-1,1]
    diff = np.clip(diff * gain, -1, 1)
    # 负值显示在 G 和 B 通道
    neg_mask = diff < 0
    rgb_diff[1, neg_mask] = -diff[neg_mask]
    rgb_diff[2, neg_mask] = -diff[neg_mask]
    # 正值显示在 R 通道
    pos_mask = diff > 0
    rgb_diff[0, pos_mask] = diff[pos_mask]
    # 返回 HWC 排列
    return np.transpose(rgb_diff, (1, 2, 0))

def extract_exp_time_from_txt(clip_path):
    """从样本目录的 `cone` 子目录中的 TXT 文件解析曝光时间（Exp Time）。

    说明：原项目在 `cone/*.txt` 中记录了各种传感器元信息，本函数寻找第一个匹配
    并读取其中包含 `Exp Time:<整数>` 的行。

    输入：
      clip_path (str): 样本目录路径，函数将在其下的 `cone` 子目录中查找 TXT 文件。

    返回：
      int 或 None -- 解析得到的曝光时间（整数），若未找到则返回 None。
    """
    cone_dir = os.path.join(clip_path, 'cone')
    files = [f for f in os.listdir(cone_dir) if f.endswith('.txt')]
    if len(files) == 0:
        return None
    file_path = os.path.join(cone_dir, files[0])
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            match = re.search(r'Exp Time:(\d+)\s', line)
            if match:
                return int(match.group(1))
    return None

def extract_tsd_with_select_T(tsd_between_exposure_tensor: torch.Tensor, select_T: Sequence[int]):
    """从 [C,T,H,W] 格式的 tsd_between_exposure 张量中选取指定时间点并构造 TD/SD 输入。

    输入约定：
      - C==3：第一通道为 TD（时间差累积），第 1/2 通道为 SD（两路的短时差分）。
      - select_T: 严格递增的时间索引列表，用于在时间维度上抽取样本。

    输出：
      - td_out: 形状为 [1, len(select_T), H, W] 的 TD 张量（按片段累积差分计算）。
      - sd_out: 形状为 [2, len(select_T), H, W] 的 SD 张量（直接从输入按索引抽取）。

    说明：该函数会在 TD 的时间轴上做前缀和并计算选定区间的差值，得到每个选定时间窗口的 TD 区间累积量。
    """
    if not isinstance(tsd_between_exposure_tensor, torch.Tensor):
        raise TypeError("tsd_between_exposure_tensor must be a torch.Tensor")
    if tsd_between_exposure_tensor.ndim != 4:
        raise ValueError(f"Expect [C,T,H,W], got {tsd_between_exposure_tensor.shape}")
    C, T, H, W = tsd_between_exposure_tensor.shape
    if C != 3:
        raise ValueError(f"Expect C=3 (TD + 2*SD). Got C={C}")
    select_T = list(select_T)
    if len(select_T) == 0:
        raise ValueError("select_T must be non-empty")
    # 参数校验：确保 select_T 为严格递增且在范围内
    for i, x in enumerate(select_T):
        if not isinstance(x, int):
            raise TypeError(f"select_T[{i}] is not int: {type(x)}")
        if x < 0 or x >= T:
            raise ValueError(f"select_T[{i}] out of range [0,{T-1}]: {x}")
        if i > 0 and x <= select_T[i - 1]:
            raise ValueError(f"select_T must be strictly increasing, got {select_T}")
    device = tsd_between_exposure_tensor.device
    idx = torch.tensor(select_T, device=device, dtype=torch.long)
    # SD 部分直接按时间索引抽取，保持两个通道
    sd_out = tsd_between_exposure_tensor[1:3].index_select(dim=1, index=idx)
    # TD 需要构建区间累积：先补一个零位用于边界计算
    td = tsd_between_exposure_tensor[0]
    td = torch.cat([td, td.new_zeros((1, H, W))], dim=0)
    L = td.shape[0]
    td[0].zero_()
    # 前缀和用于快速计算区间和
    ps = torch.cat([td.new_zeros((1, H, W)), td.cumsum(dim=0)], dim=0)
    # starts/ends 定位每个选取窗口的起止索引，方便用前缀和求差
    starts = torch.cat([idx.new_zeros((1,)), idx + 1], dim=0)
    ends = torch.cat([idx + 1, idx.new_tensor([L])], dim=0)
    td_out = ps[ends] - ps[starts]
    td_out.unsqueeze_(0)
    return td_out, sd_out


def make_output_img(F0_blur: torch.Tensor,
                    F1_blur: torch.Tensor,
                    tsd_between_exposure: torch.Tensor,
                    gt_pred_np: torch.Tensor,
                    ) -> np.ndarray:
    """构建可视化画布：左/右原始帧 + 中间每个时间步的 TD/SD/预测。

    输入：
        - F0_blur, F1_blur: torch.Tensor，形如 [1,3,H,W]，像素范围假定在 [-1,1]
        - tsd_between_exposure: torch.Tensor，形如 [1, C, T, H, W] 或可由 UpsampleTSDConv 处理
        - gt_pred_np: np.ndarray，模型预测序列，形如 (3, T, H, W) 或 (C,T,H,W) 视具体实现而定

    输出：
        - np.ndarray: H_out x W_out x 3 的可视化图像（uint8），便于保存为 JPG。
    """
    def rgb_uint8_from_m11(x_chw: np.ndarray) -> np.ndarray:
        x = (np.clip(x_chw, -1.0, 1.0) + 1.0) * 0.5
        return (x * 255.0).astype(np.uint8).transpose(1, 2, 0)

    def rgb_uint8_from_01(x_chw: np.ndarray) -> np.ndarray:
        x = np.clip(x_chw, 0.0, 1.0)
        return (x * 255.0).astype(np.uint8).transpose(1, 2, 0)

    def vizDiff_BBG_uint8(diff_hw: np.ndarray, thresh=0.01, gain=16.0) -> np.ndarray:
        diff = diff_hw.astype(np.float32)
        diff = np.clip(diff, -1.0, 1.0)
        diff[np.abs(diff) < thresh] = 0.0
        diff = np.clip(diff * gain, -1.0, 1.0)
        h, w = diff.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        neg = diff < 0
        pos = diff > 0
        rgb[neg, 0] = -diff[neg]
        rgb[pos, 2] = diff[pos]
        rgb = np.clip(rgb, 0.0, 1.0)
        return (rgb * 255.0).astype(np.uint8)

    # 将左右图像从 [-1,1] 映射回 uint8
    F0_u8 = rgb_uint8_from_m11(F0_blur.squeeze().detach().cpu().numpy())
    F1_u8 = rgb_uint8_from_m11(F1_blur.squeeze().detach().cpu().numpy())
    # 先通过预处理层将 tsd 上采样或规范化为 (C,T,H,W)
    preprocess_layer = UpsampleTSDConv()
    tsd_between_exposure = preprocess_layer(tsd_between_exposure)
    tsd_np = tsd_between_exposure.squeeze().detach().cpu().numpy()
    T = tsd_np.shape[1]
    H, W = tsd_np.shape[2], tsd_np.shape[3]
    pred_np = gt_pred_np
    rows = 3
    cols = T + 2
    canvas = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)

    # 辅助函数：将图像粘贴到画布指定格子
    def paste(img_u8: np.ndarray, r: int, c: int):
        y0, y1 = r * H, (r + 1) * H
        x0, x1 = c * W, (c + 1) * W
        canvas[y0:y1, x0:x1, :] = img_u8

    # 左上角放 F0，右上角放 F1
    paste(F0_u8, r=0, c=0)
    paste(F1_u8, r=0, c=cols - 1)

    # 每列展示一个时间步：上行 TD（可视化），中行 SD，底行 预测
    for t in range(T):
        c = t + 1
        ch0_u8 = vizDiff_BBG_uint8(tsd_np[0, t], thresh=0.01, gain=16.0)
        paste(ch0_u8, r=0, c=c)
        ch1_u8 = vizDiff_BBG_uint8(tsd_np[1, t], thresh=0.01, gain=16.0)
        paste(ch1_u8, r=1, c=c)
        pred_u8 = rgb_uint8_from_01(pred_np[:, t])
        paste(pred_u8, r=2, c=c)
    return canvas


def get_quad_frames(F0_blur: torch.Tensor,
                    F1_blur: torch.Tensor,
                    tsd_between_exposure: torch.Tensor,
                    gt_pred_np: np.ndarray,
                    ) -> list:
    """为每个时间步生成 2x2 四宫格帧列表，用于视频拼接。

    生成规则：
        - 左上 (TL): 来自 F0 或 F1 的原始图像（根据时间步选择前或后半用 F0/F1），
        - 右上 (TR): 预测的 RGB 图像，
        - 左下 (BL): TD 可视化，
        - 右下 (BR): SD 可视化。

    返回：按时间步排列的帧列表，每帧为 H x W x 3 的 uint8 图像。
    """
    def rgb_uint8_from_m11(x_chw: np.ndarray) -> np.ndarray:
        x = (np.clip(x_chw, -1.0, 1.0) + 1.0) * 0.5
        return (x * 255.0).astype(np.uint8).transpose(1, 2, 0)

    def rgb_uint8_from_01(x_chw: np.ndarray) -> np.ndarray:
        x = np.clip(x_chw, 0.0, 1.0)
        return (x * 255.0).astype(np.uint8).transpose(1, 2, 0)

    def vizDiff_BBG_uint8(diff_hw: np.ndarray, thresh=0.01, gain=16.0) -> np.ndarray:
        diff = diff_hw.astype(np.float32)
        diff = np.clip(diff, -1.0, 1.0)
        diff[np.abs(diff) < thresh] = 0.0
        diff = np.clip(diff * gain, -1.0, 1.0)
        h, w = diff.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        neg = diff < 0
        pos = diff > 0
        rgb[neg, 0] = -diff[neg]
        rgb[pos, 2] = diff[pos]
        rgb = np.clip(rgb, 0.0, 1.0)
        return (rgb * 255.0).astype(np.uint8)

    F0_u8 = rgb_uint8_from_m11(F0_blur.squeeze().detach().cpu().numpy())
    F1_u8 = rgb_uint8_from_m11(F1_blur.squeeze().detach().cpu().numpy())
    preprocess_layer = UpsampleTSDConv()
    tsd_between_exposure = preprocess_layer(tsd_between_exposure)
    tsd_np = tsd_between_exposure.squeeze().detach().cpu().numpy()
    T = tsd_np.shape[1]
    H, W = tsd_np.shape[2], tsd_np.shape[3]
    pred_np = gt_pred_np
    frames = []
    half = T // 2
    for t in range(T):
        # 根据时间步决定左上角使用 F0 还是 F1（前半步用 F0，后半步用 F1）
        tl = F0_u8 if t < half else F1_u8
        tr = rgb_uint8_from_01(pred_np[:, t])
        bl = vizDiff_BBG_uint8(tsd_np[0, t], thresh=0.01, gain=16.0)
        br = vizDiff_BBG_uint8(tsd_np[1, t], thresh=0.01, gain=16.0)
        # 若尺寸不匹配则调整到目标 H,W
        if tl.shape[:2] != (H, W):
            tl = cv2.resize(tl, (W, H), interpolation=cv2.INTER_LINEAR)
        if tr.shape[:2] != (H, W):
            tr = cv2.resize(tr, (W, H), interpolation=cv2.INTER_LINEAR)
        if bl.shape[:2] != (H, W):
            bl = cv2.resize(bl, (W, H), interpolation=cv2.INTER_NEAREST)
        if br.shape[:2] != (H, W):
            br = cv2.resize(br, (W, H), interpolation=cv2.INTER_NEAREST)
        top = np.concatenate([tl, tr], axis=1)
        bot = np.concatenate([bl, br], axis=1)
        frame = np.concatenate([top, bot], axis=0)
        frames.append(frame)
    return frames


def save_concatenated_quad_mp4(mp4_path: str, list_of_batches: list, fps: int = 5):
    """将若干批次（每批次若干时间步的帧）拼接成一个长视频并写入 MP4 文件。

    行为细节：
      - 对于每个批次（对应一个 pidx），先生成其所有时间步的四宫格帧；
      - 为避免相邻批次边界重复帧（重叠帧），对除最后一批次外的每个批次，丢弃该批次的最后一帧；
      - 将所有帧串联写入 MP4（BGR -> RGB 转换注意 OpenCV 的通道顺序）。

    参数：
      mp4_path (str): 输出 MP4 文件路径。
      list_of_batches (list): 每项为四元组 (F0_blur, F1_blur, tsd_between_exposure, gt_pred_np)。
      fps (int): 帧率。

    返回：None（文件写入磁盘，异常时抛出 RuntimeError）。
    """
    os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
    all_frames = []
    for i, (F0_blur, F1_blur, tsd_between_exposure, gt_pred_np) in enumerate(list_of_batches):
        frames = get_quad_frames(F0_blur, F1_blur, tsd_between_exposure, gt_pred_np)
        # 如果不是最后一个批次，则丢弃最后一帧以避免与下一个批次重复
        if i < len(list_of_batches) - 1 and len(frames) > 0:
            frames = frames[:-1]
        all_frames.extend(frames)
    if len(all_frames) == 0:
        raise RuntimeError('No frames to write')
    H, W = all_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter open failed: {mp4_path}")
    # OpenCV 的 VideoWriter 期望 BGR 顺序，因此写入时需要反转 RGB->BGR
    for frame in all_frames:
        writer.write(frame[..., ::-1])
    writer.release()
