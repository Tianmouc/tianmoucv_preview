import cv2
import numpy as np

"""
AUTO WHITE BALANCE (AWB)
AWB is applied to raw mosaiced images.
"""
from enum import Enum
import numpy as np

# ===============================================================
# 白平衡调整——灰度世界假设
#:param img: cv2.imread读取的图片数据
#:return: 返回的白平衡结果图片数据
# ===============================================================
def gray_world_awb(img,HSB=256):
    '''
    白平衡调整——灰度世界假设
    
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    
    '''
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
    K = (B_ave + G_ave + R_ave) / 3
    Kb, Kg, Kr = K / (B_ave+1e-8), K / (G_ave+1e-8), K / (R_ave+1e-8)
    Ba = (B * Kb)
    Ga = (G * Kg)
    Ra = (R * Kr)
    Ba[Ba>HSB] = HSB
    Ga[Ga>HSB] = HSB
    Ra[Ra>HSB] = HSB
    img[:, :, 0] = Ba
    img[:, :, 1] = Ga
    img[:, :, 2] = Ra
    #print('wb:',np.max(img))
    return img



# Config dictionary
CONFIG = {
    "raw_image_shape": (320, 640),  # height x width
    "pca_pickup_percentage": 0.05,  # Top and bottom n% will be considered. 3.5% is recommended.
    "color_enhancement_coef": 2.0,
    # Set no_processing to be True (no gamma, CCM, or CE) when you want to get input data for CCM construction.
    "verbose": False
}


class AutoWhiteBalance:
    def __init__(
            self,
            verbose=False
    ):
        self.wb_gain = np.array([1, 1, 1])
        self.verbose = verbose
        self.saturation_value = 1023
        self.color_correction_coef = CONFIG["color_enhancement_coef"]
        self.verbose = CONFIG["verbose"]
        self.pca_pickup_percentage = CONFIG["pca_pickup_percentage"]


    def __call__(self, raw_mosaic_img , method = 'GW' ,blc_avg=90):

        b = raw_mosaic_img[::2, ::2]
        gr = raw_mosaic_img[::2, 1::2]
        gb = raw_mosaic_img[1::2, ::2]
        r = raw_mosaic_img[1::2, 1::2]
        g = 0.5 * (gr + gb)  # This is based on an assumption where AWB does not rely on spatial information.
        saturation_mask = self.get_saturation_mask(raw_mosaic_img=raw_mosaic_img)

        if method == 'PCA':
            self.saturation_value = 1023 - blc_avg
            sat = self.get_saturation_mask(raw_mosaic_img)
            self.wb_gain = self.apply_pca_based_method(r, g, b, saturation_mask)
        if method == 'GW':
            self.wb_gain = self.apply_GW_method(r, g, b)

        awb_mosaic_img = raw_mosaic_img.copy().astype(np.float32)
        awb_mosaic_img[::2, ::2] = b * self.wb_gain[0]
        awb_mosaic_img[::2, 1::2] = gr * self.wb_gain[1]
        awb_mosaic_img[1::2, ::2] = gb * self.wb_gain[1]
        awb_mosaic_img[1::2, 1::2] = r * self.wb_gain[2]
        
        return awb_mosaic_img


    def apply_GW_method(self,R,G,B):
        
        B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
        
        K = (B_ave + G_ave + R_ave) / 3

        Kb, Kg, Kr = K / (B_ave+1e-8), K / (G_ave+1e-8), K / (R_ave+1e-8)
        
        return np.array([Kb,Kg,Kr])


    def apply_pca_based_method(
            self, r: np.ndarray, g: np.ndarray, b: np.ndarray, saturation_mask: np.ndarray
    ) -> np.ndarray:
        """
        # Via https://github.com/ksonod/pca-auto-white-balance/blob/main/algorithm/white_balance.py
        This is a function to use the algorithm developed in the following paper:
        - D. Cheng, D. K. Prasad, and M. S. Brown, Illuminant estimation for color consistency: why spatial domain
         methods work and the role of the color distribution, J. Opt. Soc. Am. A 31. 1049-1058, 2014

        :param r: Numpy array of the red component
        :param g: Numpy array of the green component
        :param b: Numpy array of the blue component
        :param saturation_mask: saturation mask to exclude some pixels
        :return: Numpy array with (3, )-shape of white balance gain
        """
        signal_scale = self.saturation_value

        # Data points in the RGB space.
        ix = np.array([
            r[~saturation_mask],
            g[~saturation_mask],
            b[~saturation_mask]
        ]).T / signal_scale

        # Center in the RGB space.
        i0 = np.mean(ix, axis=0).reshape(-1, 1)

        # Scalar distance
        dx = (np.dot(ix, i0) / np.linalg.norm(ix, axis=1).reshape(-1, 1) / np.linalg.norm(i0)).flatten()
        dx_sorted = np.sort(dx)

        # Top and bottom n% of data will be selected.
        idx = int(np.round(dx.shape[0] * self.pca_pickup_percentage))
        ix_selected = ix[(dx < dx_sorted[idx]) + (dx_sorted[-idx] < dx), :]

        # Conduct principle component analysis (PCA)
        sigma = np.matmul(ix_selected.T, ix_selected) / ix_selected.shape[0]
        eigen_vals, eigen_vecs = np.linalg.eig(sigma)
        principle_vec = np.abs(eigen_vecs[:, np.argmax(eigen_vals)])

        return np.array([
            principle_vec[1] / principle_vec[0],
            1.0,
            principle_vec[1] / principle_vec[2]
        ])


    def get_saturation_mask(self, raw_mosaic_img: np.ndarray) -> np.ndarray:
        """
        From raw mosaic image, numpy array including boolean is constructed to show saturated pixels. True means the
        pixel is saturated.
        :param raw_mosaic_img: h x w x 3 demosaiced image (numpy array)
        :return: numpy array containing boolean to show saturated pixels.
        """
        saturation_value = self.saturation_value

        return (saturation_value <= raw_mosaic_img[::2, ::2]) + \
                (saturation_value <= raw_mosaic_img[::2, 1::2]) + \
                (saturation_value <= raw_mosaic_img[1::2, ::2]) + \
                (saturation_value <= raw_mosaic_img[1::2, 1::2])
