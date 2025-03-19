import numpy as np
import cv2
import math
from scipy import signal

'''
https://github.com/DerrickXuNu/Illuminant-Aware-Gamut-Based-Color-Transfer/blob/master/weightedGE.py
'''

def fill_border(in_, bw):
    hh = in_.shape[0]
    ww = in_.shape[1]
    dd = 1 if len(in_.shape) == 2 else in_.shape[2]
    bw = int(bw)
    if dd == 1:
        out = np.zeros((hh + bw * 2, ww + bw * 2))
        out[:bw, :bw] = np.ones((bw, bw)) * in_[0, 0]
        out[bw + hh:2 * bw + hh, :bw] = np.ones((bw, bw)) * in_[hh - 1, 0]
        out[: bw, bw + ww: 2 * bw + ww] = np.ones((bw, bw)) * in_[0, ww - 1]
        out[bw + hh:2 * bw + hh, bw + ww: 2 * bw + ww] = np.ones((bw, bw)) * in_[hh - 1, ww - 1]
        out[bw: bw + hh, bw: bw + ww] = in_
        out[: bw, bw: bw + ww] = np.ones((bw, 1)).dot(in_[0, :].reshape(1, -1))
        out[bw + hh: 2 * bw + hh, bw: bw + ww] = np.ones((bw, 1)).dot(in_[hh - 1, :].reshape(1, -1))
        out[bw: bw + hh, : bw] = in_[:, 0].reshape(-1, 1).dot(np.ones((1, bw)))
        out[bw: bw + hh, bw + ww: 2 * bw + ww] = in_[:, ww - 1].reshape(-1, 1).dot(np.ones((1, bw)))
    else:
        out = np.zeros((hh + bw * 2, ww + bw * 2, dd))
        for ii in range(dd):
            out[:bw, :bw, ii] = np.ones((bw, bw)) * in_[0, 0, ii]
            out[bw + hh: 2 * bw + hh, : bw, ii] = np.ones((bw, bw)) * in_[hh - 1, 0, ii]
            out[: bw, bw + ww: 2 * bw + ww, ii] = np.ones((bw, bw)) * in_[0, ww - 1, ii]
            out[bw + hh: 2 * bw + hh, bw + ww: 2 * bw + ww, ii] = np.ones((bw, bw)) * in_[hh - 1, ww - 1, ii]
            out[bw: bw + hh, bw: bw + ww, ii] = in_[:, :, ii]
            out[: bw, bw: bw + ww, ii] = np.ones((bw, 1)).dot(in_[0, :, ii].reshape(1, -1))
            out[bw + hh: 2 * bw + hh, bw: bw + ww, ii] = np.ones((bw, 1)).dot(in_[hh - 1, :, ii].reshape(1, -1))
            out[bw: bw + hh, : bw, ii] = in_[:, 0, ii].reshape(-1, 1).dot(np.ones((1, bw)))
            out[bw: bw + hh, bw + ww: 2 * bw + ww, ii] = in_[:, ww - 1, ii].reshape(-1, 1).dot(np.ones((1, bw)))
    return out

def gDer(f, sigma, iorder, jorder):
    break_off_sigma = 3.0
    filtersize = math.floor(break_off_sigma * sigma + 0.5)
    f = fill_border(f, filtersize)
    x = np.arange(-filtersize, filtersize + 1)
    Gauss = 1 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp((x ** 2) / (-2.0 * sigma * sigma))
    if iorder == 0:
        Gx = Gauss / (1.0 * np.sum(Gauss))
    elif iorder == 1:
        Gx = -(x / (1.0 * sigma ** 2)) * Gauss
        Gx = Gx / (1.0 * np.sum(np.sum(x * Gx)))
    elif iorder == 2:
        Gx = (x ** 2 / sigma ** 4.0 - 1 / sigma ** 2.0) * Gauss
        Gx = Gx - np.sum(Gx) / (1.0 * x.shape[0])
        Gx = Gx / (1.0 * np.sum(0.5 * x * x * Gx))
    Gx = Gx.reshape(1, -1)
    H = -signal.convolve2d(f, Gx, mode='same')

    if jorder == 0:
        Gy = Gauss / (1.0 * np.sum(Gauss))
    elif jorder == 1:
        Gy = -(x / (1.0 * sigma ** 2)) * Gauss
        Gy = Gy / (1.0 * (np.sum(np.sum(x * Gy))))
    elif jorder == 2:
        Gy = (x ** 2 / sigma ** 4.0 - 1 / sigma ** 2.0) * Gauss
        Gy = Gy - np.sum(Gy) / (1.0 * x.shape[0])
        Gy = Gy / (1.0 * np.sum(0.5 * x * x * Gy))
    Gy = Gy.reshape(1, -1)
    H = signal.convolve2d(H, Gy.conj().T, mode='same')
    filtersize = int(filtersize)
    H = H[filtersize:H.shape[0] - filtersize, filtersize: H.shape[1] - filtersize]
    return H

def set_board(in_, width, method=1):
    temp = np.ones(in_.shape)
    y, x = np.mgrid[1:(in_.shape[0] + 1), 1:(in_.shape[1] + 1)]
    temp = temp * ((x < temp.shape[1] - width + 1) & (x > width))
    temp = temp * ((y < temp.shape[0] - width + 1) & (y > width))
    out = temp * in_
    if method == 1:
        out = out + (np.sum(out[:]) / (1.0 * np.sum(temp[:]))) * (np.ones(np.shape(in_)) - temp)
    return out

def dilation33(in_, it=1):
    in_ = np.asarray(in_)
    hh = in_.shape[0]
    ll = in_.shape[1]
    out = np.zeros((hh, ll, 3))

    while it > 0:
        it = it - 1
        out[:hh - 1, :, 0] = in_[1:hh, :]
        out[hh - 1, :, 0] = in_[hh - 1, :]
        out[:, :, 1] = in_
        out[0, :, 2] = in_[0, :]
        out[1:, :, 2] = in_[0:hh - 1, :]
        out2 = out.max(2)
        out[:, :ll - 1, 0] = out2[:, 1:ll]
        out[:, ll - 1, 0] = out2[:, ll - 1]
        out[:, :, 1] = out2
        out[:, 0, 2] = out2[:, 0]
        out[:, 1:, 2] = out2[:, :ll - 1]
        out = out.max(2)
        in_ = out
    return out

def compute_spvar(im, sigma):
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    Rx = gDer(R, sigma, 1, 0)
    Ry = gDer(R, sigma, 0, 1)
    Rw = np.sqrt(Rx ** 2 + Ry ** 2)

    Gx = gDer(G, sigma, 1, 0)
    Gy = gDer(G, sigma, 0, 1)
    Gw = np.sqrt(Gx ** 2 + Gy ** 2)

    Bx = gDer(B, sigma, 1, 0)
    By = gDer(B, sigma, 0, 1)
    Bw = np.sqrt(Bx ** 2 + By ** 2)

    # Opponent_der
    O3_x = (Rx + Gx + Bx) / np.sqrt(3)
    O3_y = (Ry + Gy + By) / np.sqrt(3)
    sp_var = np.sqrt(O3_x ** 2 + O3_y ** 2)

    return sp_var, Rw, Gw, Bw

def im2double(im):
    info = np.iinfo(im.dtype)  # Get the data type of the input image
    return im.astype(np.float) / (1.0 * info.max)

def weightedGE_apply(input_im, kappa=1, mink_norm=6, sigma=2):
    iter = 10
    mask_cal = np.zeros((input_im.shape[0], input_im.shape[1]))
    tmp_ill = np.array([1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)])
    final_ill = tmp_ill.copy()
    tmp_image = input_im.copy()
    flag = 1
    while iter > 0 and flag == 1:
        iter -= 1
        tmp_image[:, :, 0] = tmp_image[:, :, 0] / (np.sqrt(3) * (tmp_ill[0]))
        tmp_image[:, :, 1] = tmp_image[:, :, 1] / (np.sqrt(3) * (tmp_ill[1]))
        tmp_image[:, :, 2] = tmp_image[:, :, 2] / (np.sqrt(3) * (tmp_ill[2]))
        sp_var, Rw, Gw, Bw = compute_spvar(tmp_image, sigma)
        mask_zeros = np.maximum(Rw, np.maximum(Gw, Bw)) < np.finfo(float).eps
        mask_pixels = dilation33(((tmp_image.max(2)) == 255))
        mask = set_board((np.logical_or(np.logical_or(mask_pixels, mask_zeros), mask_cal) == 0).astype(float),
                         sigma + 1, 0)
        mask[mask != 0] = 1
        grad_im = np.sqrt(Rw ** 2 + Gw ** 2 + Bw ** 2)
        weight_map = (sp_var / (1.0 * grad_im)) ** kappa
        weight_map[weight_map > 1] = 1
        data_Rx = np.power(Rw * weight_map, mink_norm)
        data_Gx = np.power(Gw * weight_map, mink_norm)
        data_Bx = np.power(Bw * weight_map, mink_norm)

        tmp_ill[0] = np.power(np.sum(data_Rx * mask), 1 / (1.0 * mink_norm))
        tmp_ill[1] = np.power(np.sum(data_Gx * mask), 1 / (1.0 * mink_norm))
        tmp_ill[2] = np.power(np.sum(data_Bx * mask), 1 / (1.0 * mink_norm))

        tmp_ill = tmp_ill / (1.0 * np.linalg.norm(tmp_ill))
        final_ill = final_ill * tmp_ill
        final_ill = final_ill / (1.0 * np.linalg.norm(final_ill))
        if np.arccos(tmp_ill.dot(1 / math.sqrt(3) * np.array([1, 1, 1]).T)) / np.pi * 180 < 0.05:
            flag = 0
    white_R = final_ill[0]
    white_G = final_ill[1]
    white_B = final_ill[2]
    
    Kb = 1/ (np.sqrt(3) * (final_ill[0]))
    Kg = 1/ (np.sqrt(3) * (final_ill[1]))
    Kr = 1/ (np.sqrt(3) * (final_ill[2]))

    return Kb,Kg,Kr
