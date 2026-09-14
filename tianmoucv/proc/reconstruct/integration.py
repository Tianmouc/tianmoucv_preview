import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

from tianmoucv.isp import SD2XY,upsampleTSD

def TD_integration(tsdiff,F0,F1,t, TD_BG_NOISE = 0, threshGate=4/255, dig_scaling= 1.5):
    '''
    AOP+COP合成灰度
    
    1. 校正TD的正向和负向差分的不一致性
    
    2. 计算AOP到COP的线性缩放系数
    
    3. 双向TD积累+SD灰度合成最终结果
    
    parameter:
        :param F0: [h,w,3],torch.Tensor
        :param F0: [h,w,3],torch.Tensor
        :param tsdiff: [3,T,h,w],torch.Tensor, 默认decoded结果的堆积
        :param TDnoise: 噪声矩阵 [h,w], torch.Tensor
        :param threshGate=4/255: 积累时的噪声阈值
        :param t: int

    '''
    gray0 = torch.mean(F0,dim=-1)
    gray1 = torch.mean(F1,dim=-1)
    TD_COP = gray1-gray0
    
    #adjust TD bias for tianmouc
    TD = tsdiff[0,1:,...]     

    TD -= TD_BG_NOISE
    TD[abs(TD)<threshGate]=0
    
    possum  = torch.sum(TD[TD>0])
    negsum  = torch.sum(abs(TD[TD<0]))
    bias = (negsum-possum)/TD[TD>0].view(1,-1).shape[1]
    TD[TD>0] += bias

    AOPDiff = torch.sum(TD[1:,...],dim=0)
    AOPDiff = F.interpolate(AOPDiff.unsqueeze(0).unsqueeze(0), size=TD_COP.shape, mode='bilinear').squeeze(0).squeeze(0)
    AOP_COP_scale_neg = torch.sum(TD_COP[TD_COP<0])/torch.sum(AOPDiff[AOPDiff<0]) 
    AOP_COP_scale_pos = torch.sum(TD_COP[TD_COP>0])/torch.sum(AOPDiff[AOPDiff>0]) 

    TD[TD<0] *= AOP_COP_scale_neg * dig_scaling
    TD[TD>0] *= AOP_COP_scale_pos  * dig_scaling

    forward_TD =  torch.sum(TD[0:t,...],dim=0)
    backward_TD =  torch.sum(TD[t:,...],dim=0)


    forward_TD  = F.interpolate(forward_TD.unsqueeze(0).unsqueeze(0), size=(320,640), mode='bilinear').squeeze(0).squeeze(0)
    backward_TD = F.interpolate(backward_TD.unsqueeze(0).unsqueeze(0), size=(320,640), mode='bilinear').squeeze(0).squeeze(0)
    
    hdr =  (gray0+forward_TD + gray1 - backward_TD)/2

    #hdr[hdr<0]=0
    #hdr[hdr>1]=1
    
    return hdr


def SD_integration_legacy(SDx:np.array, SDy:np.array)  -> np.array:
    '''
    SD直接积分累加重建，简单可视化用
    use mapped SDx and SDy to conduct direct integration
    '''
    canvas = np.zeros([SDx.shape[0],SDx.shape[1]+1])
    grayy = np.cumsum(SDy,axis=0)
    goody_first = grayy[:,0]
    canvas[:,0] = goody_first
    canvas[:,1:] = SDx
    gray = np.cumsum(canvas[:,:-1],axis=1)
    return gray


def _to_numpy_float64(data):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    return np.asarray(data, dtype=np.float64)


def _stable_cumsum(data, axis):
    """Kahan-style cumulative sum along one axis."""
    data = np.asarray(data, dtype=np.float64)
    moved = np.moveaxis(data, axis, 0)
    out = np.empty_like(moved, dtype=np.float64)
    total = np.zeros(moved.shape[1:], dtype=np.float64)
    compensation = np.zeros_like(total)

    for i in range(moved.shape[0]):
        y = moved[i] - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
        out[i] = total

    return np.moveaxis(out, 0, axis)


def _forward_integral(grad, axis):
    out = np.zeros_like(grad, dtype=np.float64)
    if grad.shape[axis] <= 1:
        return out

    src = [slice(None)] * grad.ndim
    dst = [slice(None)] * grad.ndim
    src[axis] = slice(0, -1)
    dst[axis] = slice(1, None)
    out[tuple(dst)] = _stable_cumsum(grad[tuple(src)], axis=axis)
    return out


def _backward_integral(grad, axis):
    out = np.zeros_like(grad, dtype=np.float64)
    if grad.shape[axis] <= 1:
        return out

    src = [slice(None)] * grad.ndim
    dst = [slice(None)] * grad.ndim
    src[axis] = slice(0, -1)
    dst[axis] = slice(0, -1)
    reversed_grad = np.flip(grad[tuple(src)], axis=axis)
    out[tuple(dst)] = -np.flip(_stable_cumsum(reversed_grad, axis=axis), axis=axis)
    return out


def _align_path(path, align):
    if align is None:
        return path
    if align == 'median':
        return path - np.nanmedian(path)
    if align == 'mean':
        return path - np.nanmean(path)
    raise ValueError("align must be 'median', 'mean', or None")


def _reduce_paths(paths, reduce):
    stack = np.stack(paths, axis=0)
    if reduce == 'mean':
        return np.nanmean(stack, axis=0)
    if reduce == 'median':
        return np.nanmedian(stack, axis=0)
    if reduce == 'trimmed_mean':
        if stack.shape[0] <= 2:
            return np.nanmean(stack, axis=0)
        sorted_stack = np.sort(stack, axis=0)
        return np.nanmean(sorted_stack[1:-1], axis=0)
    raise ValueError("reduce must be 'trimmed_mean', 'mean', or 'median'")


def _moving_average_1d(data, window):
    data = np.asarray(data, dtype=np.float64)
    if window is None or window <= 1 or data.size < 3:
        return np.zeros_like(data) + np.nanmedian(data)

    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, data.size if data.size % 2 == 1 else data.size - 1)
    if window <= 1:
        return np.zeros_like(data) + np.nanmedian(data)

    pad = window // 2
    padded = np.pad(data, pad, mode='reflect')
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode='valid')


def _robust_clip_1d(data, sigma):
    if sigma is None or sigma <= 0:
        return data

    center = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - center))
    if not np.isfinite(mad) or mad <= 0:
        return data

    limit = float(sigma) * 1.4826 * mad
    return np.clip(data, center - limit, center + limit)


def _low_gradient_mask(grad_support, percentile):
    finite = np.isfinite(grad_support)
    if not np.any(finite):
        return finite

    threshold = np.nanpercentile(grad_support[finite], percentile)
    return finite & (grad_support <= threshold)


def _masked_median_profile(values, mask, axis, min_fraction):
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if axis not in (0, 1):
        raise ValueError('axis must be 0 or 1')

    values_moved = np.moveaxis(values, axis, 0)
    mask_moved = np.moveaxis(mask, axis, 0)
    profile = np.zeros(values_moved.shape[1:], dtype=np.float64)
    min_count = max(3, int(values_moved.shape[0] * min_fraction))

    for index in np.ndindex(profile.shape):
        series = values_moved[(slice(None),) + index]
        series_mask = mask_moved[(slice(None),) + index] & np.isfinite(series)
        if np.count_nonzero(series_mask) >= min_count:
            profile[index] = np.nanmedian(series[series_mask])

    return profile


def _integrate_profile_gradient(profile, length):
    bias = np.zeros(length, dtype=np.float64)
    if length <= 1:
        return bias

    bias[1:] = _stable_cumsum(profile[:length - 1], axis=0)
    return bias - np.nanmedian(bias)


def _stripe_component_from_bias(bias, smooth_window, clip_sigma):
    smooth = _moving_average_1d(bias, smooth_window)
    stripe = bias - smooth
    stripe -= np.nanmedian(stripe)
    return _robust_clip_1d(stripe, clip_sigma)


def _suppress_stripe_noise(gray, sdx, sdy, axes='both', strength=1.0,
                           smooth_window=9, edge_percentile=70,
                           min_fraction=0.1, clip_sigma=6.0):
    '''
    Suppress row/column stripe bias that is not supported by input gradients.

    The estimator only removes a separable row/column offset whose induced
    image gradient appears in ``gray`` but not in ``sdx``/``sdy``. Strong SD
    edges are excluded from the robust profile estimate, so real stripe-shaped
    image content is preserved when it is supported by the sensor gradients.
    '''
    corrected = np.asarray(gray, dtype=np.float64).copy()
    sdx = np.asarray(sdx, dtype=np.float64)
    sdy = np.asarray(sdy, dtype=np.float64)
    axes = axes.lower()
    strength = float(strength)

    if axes not in ('both', 'horizontal', 'vertical', 'h', 'v', 'row', 'col'):
        raise ValueError("axes must be 'both', 'horizontal', or 'vertical'")

    if strength <= 0:
        return corrected

    if axes in ('both', 'horizontal', 'h', 'row') and corrected.shape[0] > 1:
        residual_y = np.diff(corrected, axis=0) - sdy[:-1, :]
        support_y = np.abs(sdy[:-1, :])
        if sdx.shape[0] > 1:
            support_y = support_y + 0.5 * (np.abs(sdx[:-1, :]) + np.abs(sdx[1:, :]))
        mask_y = _low_gradient_mask(support_y, edge_percentile)
        row_delta = _masked_median_profile(residual_y, mask_y, axis=1, min_fraction=min_fraction)
        row_bias = _integrate_profile_gradient(row_delta, corrected.shape[0])
        row_stripe = _stripe_component_from_bias(row_bias, smooth_window, clip_sigma)
        corrected -= strength * row_stripe[:, None]

    if axes in ('both', 'vertical', 'v', 'col') and corrected.shape[1] > 1:
        residual_x = np.diff(corrected, axis=1) - sdx[:, :-1]
        support_x = np.abs(sdx[:, :-1])
        if sdy.shape[1] > 1:
            support_x = support_x + 0.5 * (np.abs(sdy[:, :-1]) + np.abs(sdy[:, 1:]))
        mask_x = _low_gradient_mask(support_x, edge_percentile)
        col_delta = _masked_median_profile(residual_x, mask_x, axis=0, min_fraction=min_fraction)
        col_bias = _integrate_profile_gradient(col_delta, corrected.shape[1])
        col_stripe = _stripe_component_from_bias(col_bias, smooth_window, clip_sigma)
        corrected -= strength * col_stripe[None, :]

    return corrected


def SD_integration(SDx:np.array, SDy:np.array, *, method='multi_path',
                   reduce='trimmed_mean', align='median', anchor=None,
                   suppress_stripes=False, stripe_axes='both',
                   stripe_strength=1.0, stripe_window=9,
                   stripe_edge_percentile=70, stripe_min_fraction=0.1,
                   stripe_clip_sigma=6.0, return_tensor=False) -> np.array:
    '''
    Directly integrate mapped SDx and SDy to reconstruct a gray image.

    The default ``multi_path`` method keeps the direct-cumulative character of
    the original algorithm, but reduces stripe artifacts by:

    1. accumulating in float64 with compensated cumulative sums;
    2. integrating from four corners with horizontal-first and vertical-first
       paths;
    3. aligning the unknown DC offset of each path and combining them with a
       trimmed mean.

    Set ``suppress_stripes=True`` to remove one-pixel row/column stripe bias
    whose gradient is not supported by the input SDx/SDy. This is intentionally
    not a blind Fourier notch: strong SD edges are excluded while estimating the
    bias, so real stripe-shaped content is preserved when it is present in the
    gradients.

    Use ``method='legacy'`` to reproduce the original one-path integration.
    '''
    if method in ('legacy', 'single_path'):
        result = SD_integration_legacy(SDx, SDy)
    elif method == 'multi_path':
        sdx = _to_numpy_float64(SDx)
        sdy = _to_numpy_float64(SDy)
        if sdx.shape != sdy.shape:
            raise ValueError('SDx and SDy must have the same shape')
        if sdx.ndim != 2:
            raise ValueError('SDx and SDy must be 2-D arrays')

        x_from_left = _forward_integral(sdx, axis=1)
        x_from_right = _backward_integral(sdx, axis=1)
        y_from_top = _forward_integral(sdy, axis=0)
        y_from_bottom = _backward_integral(sdy, axis=0)

        paths = [
            y_from_top[:, 0:1] + x_from_left,
            x_from_left[0:1, :] + y_from_top,
            y_from_top[:, -1:] + x_from_right,
            x_from_right[0:1, :] + y_from_top,
            y_from_bottom[:, 0:1] + x_from_left,
            x_from_left[-1:, :] + y_from_bottom,
            y_from_bottom[:, -1:] + x_from_right,
            x_from_right[-1:, :] + y_from_bottom,
        ]
        result = _reduce_paths([_align_path(path, align) for path in paths], reduce)
    else:
        raise ValueError("method must be 'multi_path', 'legacy', or 'single_path'")

    if suppress_stripes:
        stripe_options = suppress_stripes if isinstance(suppress_stripes, dict) else {}
        result = _suppress_stripe_noise(
            result,
            _to_numpy_float64(SDx),
            _to_numpy_float64(SDy),
            axes=stripe_options.get('axes', stripe_axes),
            strength=stripe_options.get('strength', stripe_strength),
            smooth_window=stripe_options.get('window', stripe_window),
            edge_percentile=stripe_options.get('edge_percentile', stripe_edge_percentile),
            min_fraction=stripe_options.get('min_fraction', stripe_min_fraction),
            clip_sigma=stripe_options.get('clip_sigma', stripe_clip_sigma),
        )

    if anchor is not None:
        anchor_np = _to_numpy_float64(anchor)
        if anchor_np.ndim == 3:
            anchor_np = np.mean(anchor_np, axis=-1)
        result = result - np.nanmedian(result) + np.nanmedian(anchor_np)

    if return_tensor:
        device = SDx.device if torch.is_tensor(SDx) else None
        dtype = SDx.dtype if torch.is_tensor(SDx) else torch.float32
        return torch.as_tensor(result, dtype=dtype, device=device)
    return result
