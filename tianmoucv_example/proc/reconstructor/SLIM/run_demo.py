"""天眸插帧（同时具备去模糊和一定的HDR功能）-独立运行的演示脚本。

功能概述：
- 从tmdat中 进入数据预处理流程
- 使用本地复制的 `BiIterativeTSDInterpolatorFull` 模型进行前向推理，
- 保存每个 pidx 的可视化 JPG，并将多个结果拼接为一个长 MP4。

用法示例：
    python demo_slim/run_demo.py --data_root /data6/xiangru/Real_TMC_data/cxr_suuperres \
                                 --sample indoor_1 --pidx 50 54 --device cuda:0 --dtype bf16
"""
import os
import argparse
import torch
import numpy as np
from PIL import Image

# local networks (standalone copies)
from .networks import BiIterativeTSDInterpolatorFull

# reuse demo utility functions for visualization and TSD extraction
from .demo_utils import extract_exp_time_from_txt, extract_tsd_with_select_T, make_output_img, get_quad_frames, save_concatenated_quad_mp4
from tianmoucv.data import TianmoucDataReader


def parse_args():
    """解析命令行参数。

    返回：
        argparse.Namespace -- 包含解析后的参数：
            - data_root: 数据根目录路径
            - sample: 样本子目录名
            - pidx: 单个索引或起止区间（1 或 2 个整数）
            - device: 运行设备，例如 'cuda:0'
            - dtype: 计算精度选项，'fp32'/'fp16'/'bf16'
            - weights: 可选的权重路径，用于加载预训练参数
            - out_dir: 输出目录
    """
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--sample', required=True)
    p.add_argument('--pidx', type=int, nargs='+', required=True, help='single pidx or two ints start end')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=['fp32','fp16','bf16'], default='fp16', help='Computation dtype for inference')
    p.add_argument('--weights', default="/data2/myp/published_projects/GenRec/output/HS_HDR_unet_dev_final/checkpoint/iter_040000/tmcBiDirectionInterpolationBranch/diffusion_pytorch_model.bin", help='Optional state_dict path')
    p.add_argument('--out_dir', default='./demo_slim_output')
    return p.parse_args()


def load_weights_if_given(model, path, device):
    """尝试将指定路径的权重加载到模型中。
    加载失败将会报错，请检查
    """
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    print('Loaded weights into model from', path)


def run_one(args):
    device = torch.device(args.device)
    sample_path = os.path.join(args.data_root, args.sample)
    dataset = TianmoucDataReader(sample_path, N=1, camera_idx=0)
    cone_exp_time = extract_exp_time_from_txt(sample_path)
    cone_exp_period = cone_exp_time // 1320

    # 支持 pidx 为单个索引或 [start end] 区间
    if len(args.pidx) == 1:
        pidx_list = [args.pidx[0]]
    elif len(args.pidx) == 2:
        start, end = args.pidx
        pidx_list = list(range(start, end + 1))
    else:
        raise ValueError('--pidx must be one int or two ints (start end)')

    # 只构建一次模型（复用权重），提高效率
    model = BiIterativeTSDInterpolatorFull(wf=24, depth=3, has_up_sample=True).to(device)
    load_weights_if_given(model, args.weights, device)

    # 用于后续生成长视频的中间结果收集（在 CPU 上保持张量，便于写视频）
    batches_for_video = []

    # 根据用户选择决定是否使用 AMP autocast
    dtype_opt = getattr(args, 'dtype', 'fp32')
    use_amp = False
    amp_dtype = None
    if device.type == 'cuda' and dtype_opt in ('fp16', 'bf16'):
        use_amp = True
        amp_dtype = torch.float16 if dtype_opt == 'fp16' else torch.bfloat16

    for p_idx in pidx_list:
        # load two consecutive frames
        sample0 = dataset[p_idx]
        sample1 = dataset[p_idx + 1]

        F0 = sample0['F0'].numpy()
        F1 = sample1['F0'].numpy()
        F0_norm_np = F0 * 2.0 - 1.0
        F1_norm_np = F1 * 2.0 - 1.0

        tsdiff0 = sample0['rawDiff'].numpy() / 128.0
        tsdiff1 = sample1['rawDiff'].numpy() / 128.0

        sd_index = cone_exp_period // 2
        sd_0_np = tsdiff0[1:, sd_index, ...]
        sd_1_np = tsdiff1[1:, sd_index, ...]

        RGBOFFSET = 0
        TD_FIX_LEN = 10
        td_in_exposure_0_np = tsdiff0[0, RGBOFFSET+1:RGBOFFSET+cone_exp_period, ...]
        td_in_exposure_1_np = tsdiff1[0, RGBOFFSET+1:RGBOFFSET+cone_exp_period, ...]

        cur_len = td_in_exposure_0_np.shape[0]
        if cur_len < TD_FIX_LEN:
            pad_len = TD_FIX_LEN - cur_len
            pad = np.zeros((pad_len, td_in_exposure_0_np.shape[1], td_in_exposure_0_np.shape[2]), dtype=td_in_exposure_0_np.dtype)
            td_in_exposure_0_np = np.concatenate([td_in_exposure_0_np, pad], axis=0)
            td_in_exposure_1_np = np.concatenate([td_in_exposure_1_np, pad], axis=0)

        F0_norm = torch.from_numpy(F0_norm_np).permute(2,0,1).float().unsqueeze(0)
        td_in_exposure_0_norm = torch.from_numpy(td_in_exposure_0_np).float().unsqueeze(0)
        sd_0_norm = torch.from_numpy(sd_0_np).float().unsqueeze(0)
        F1_norm = torch.from_numpy(F1_norm_np).permute(2,0,1).float().unsqueeze(0)
        td_in_exposure_1_norm = torch.from_numpy(td_in_exposure_1_np).float().unsqueeze(0)
        sd_1_norm = torch.from_numpy(sd_1_np).float().unsqueeze(0)
        tsd_between_exposure_np = np.concatenate([tsdiff0[:, sd_index:-1, ...], tsdiff1[:, 0:sd_index+1, ...]], axis=1)
        tsd_between_exposure_norm = torch.from_numpy(tsd_between_exposure_np).float()

        # flip to match original demo
        F0_norm = torch.flip(F0_norm, dims=[-1])
        F1_norm = torch.flip(F1_norm, dims=[-1])

        td_between_exposure_norm, sd_between_exposure_norm = extract_tsd_with_select_T(tsd_between_exposure_norm, list(range(26)))
        td_between_exposure_norm = td_between_exposure_norm.unsqueeze(0)
        sd_between_exposure_norm = sd_between_exposure_norm.unsqueeze(0)
        tsd_between_exposure_norm = torch.cat([td_between_exposure_norm[:,:,:26,...], sd_between_exposure_norm], dim=1)

        # move inputs to device
        F0_norm = F0_norm.to(device)
        F1_norm = F1_norm.to(device)
        sd_0_norm = sd_0_norm.to(device)
        sd_1_norm = sd_1_norm.to(device)
        td_in_exposure_0_norm = td_in_exposure_0_norm.to(device)
        td_in_exposure_1_norm = td_in_exposure_1_norm.to(device)
        td_between_exposure_norm = td_between_exposure_norm.to(device)
        sd_between_exposure_norm = sd_between_exposure_norm.to(device)

        # forward
        with torch.no_grad():
            if use_amp:
                try:
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        out_frames, _ = model(F0_norm, F1_norm, td_in_exposure_0_norm, sd_0_norm, td_in_exposure_1_norm, sd_1_norm, td_between_exposure_norm, sd_between_exposure_norm)
                except Exception:
                    # fallback to no autocast if device doesn't support requested dtype
                    out_frames, _ = model(F0_norm, F1_norm, td_in_exposure_0_norm, sd_0_norm, td_in_exposure_1_norm, sd_1_norm, td_between_exposure_norm, sd_between_exposure_norm)
            else:
                out_frames, _ = model(F0_norm, F1_norm, td_in_exposure_0_norm, sd_0_norm, td_in_exposure_1_norm, sd_1_norm, td_between_exposure_norm, sd_between_exposure_norm)

        # out_frames: (B, 3, T, H, W)
        out = out_frames.squeeze(0).cpu().numpy()  # (3, T, H, W)

        os.makedirs(args.out_dir, exist_ok=True)
        jpg_path = os.path.join(args.out_dir, f"{args.sample}_{p_idx}.jpg")
        canvas = make_output_img(F0_blur=F0_norm.cpu(), F1_blur=F1_norm.cpu(), tsd_between_exposure=tsd_between_exposure_norm.cpu(), gt_pred_np=out)
        Image.fromarray(canvas).save(jpg_path, format='JPEG')
        print('Saved visualization to', jpg_path)

        # prepare frames for concatenated video; keep tensors on CPU for utils
        batches_for_video.append((F0_norm.cpu(), F1_norm.cpu(), tsd_between_exposure_norm.cpu(), out))

    # after all pidx processed, write concatenated long video
    long_mp4 = os.path.join(args.out_dir, f"{args.sample}_{pidx_list[0]}-{pidx_list[-1]}.mp4")
    try:
        save_concatenated_quad_mp4(long_mp4, batches_for_video, fps=5)
        print('Saved concatenated mp4 to', long_mp4)
    except Exception as e:
        print('Could not save concatenated mp4:', e)


if __name__ == '__main__':
    args = parse_args()
    run_one(args)
