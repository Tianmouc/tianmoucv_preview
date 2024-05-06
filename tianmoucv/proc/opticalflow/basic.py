import cv2
import numpy as np
import torch
import torch.nn as nn

# ===============================================================
# 用算出的光流插帧
# =============================================================== 
def interpolate_image(image:np.array, u: torch.Tensor,v: torch.Tensor):
    '''
    用算出的光流插帧

    parameter:
        :param image: [h,w,3],np.array
        :param u: x向光流,[h,w],np.array
        :param v: y向光流,[h,w],np.array

    '''
    height, width = image.shape[:2]
    interpolated_image = np.zeros_like(image)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    new_x = x + u.squeeze(0).numpy()
    new_y = y + v.squeeze(0).numpy()
    interpolated_image = cv2.remap(image, 
                                   new_x.astype(np.float32), 
                                   new_y.astype(np.float32), 
                                   interpolation=cv2.INTER_LINEAR)
    return interpolated_image


# ===============================================================
# 可视化光流
# ===============================================================
def compute_color(u:np.array, v:np.array):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 0.1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img

# ===============================================================
# 光流用的色轮
# ===============================================================
def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

# ===============================================================
# 光流uv to RGB
# ===============================================================
def flow_to_image(flow:np.array):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8
    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0
    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))
    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))
    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)
    img = compute_color(u, v)
    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0
    return np.uint8(img)

# ===============================================================
# 利用uv输入做变形
# uv : [b,2,w,h]
# img: [b,3,w,h]
# ===============================================================
class backWarp(nn.Module):
    """
    A class for creating a backwarping object.
    This is used for backwarping to an image:
    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).
    ...
    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """
    def __init__(self, W, H, device):
        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)
        
    def forward(self, img:torch.Tensor, flow:torch.Tensor):
        # uv有奇怪的偏移
        MAGIC_NUM =  0.5
        # Extract horizontal and vertical flows.
        self.W = flow.size(3)
        self.H = flow.size(2)
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u + MAGIC_NUM
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v + MAGIC_NUM
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut

# ===============================================================
# nature论文中用的光流filter
# ===============================================================
class opticalDetector_Maxone():
    
    def __init__(self,noiseThresh=8,distanceThresh=0.2):
        self.noiseThresh = noiseThresh
        self.th = distanceThresh
        self.accumU = 0
        self.accumV = 0
        
    def __call__(self,sd:torch.Tensor,td:torch.Tensor,ifInterploted = False):
        
        td[abs(td)<self.noiseThresh] = 0
        sd[abs(sd)<self.noiseThresh] = 0

        rawflow = HS_optical_flow(sd,td,ifInterploted = ifInterploted)
        
        flow = flow_to_image(rawflow.permute(1,2,0).numpy())
        
        flowup = np.zeros([flow.shape[0]*2,flow.shape[1]*2,3])
        flowup[1::2,1::2,:] = flow/255.0
        flowup[0::2,1::2,:] = flow/255.0
        flowup[1::2,0::2,:] = flow/255.0
        flowup[0::2,0::2,:] = flow/255.0

        #计算平均速度
        u = rawflow.permute(1,2,0).numpy()[:, :, 0]
        v = rawflow.permute(1,2,0).numpy()[:, :, 1]
        uv = [u,v]

        # case相关，去掉u是正的的那些背景光流
        distance = ((u)**2 + (v)**2) *(u<0)
        
        #和平均光流方向之差
        distance[distance>self.th] = 1
        distance[distance<self.th] = 0
        distanceup = np.zeros([flow.shape[0]*2,flow.shape[1]*2])

        # 膨胀
        kernel = np.ones((3,3),np.uint8)              
        distance = cv2.dilate(distance,kernel,iterations=3) 

        distanceup[1::2,1::2] = distance * 255.0
        distanceup[0::2,1::2] = distance * 255.0
        distanceup[1::2,0::2] = distance * 255.0
        distanceup[0::2,0::2] = distance * 255.0
        f = (distanceup).copy().astype(np.uint8)
        contours,hierarchy = cv2.findContours(f,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(f, contours, -1, (0, 255, 255), 2)
        #找到最大区域并填充
        area = []
        box = None
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))
        if len(area)>0:
            if np.max(area) < 1200:
                return None,distanceup,flowup
            max_idx = np.argmax(area)
            for i in range(max_idx - 1):
                cv2.fillConvexPoly(f, contours[max_idx - 1], 0)
            cv2.fillConvexPoly(f, contours[max_idx], 255)
            #求最大连通域的中心坐标
            maxcon = contours[max_idx]
            x1 = np.min(maxcon[:,:,0])  
            x2 = np.max(maxcon[:,:,0])  
            y1 = np.min(maxcon[:,:,1])  
            y2 = np.max(maxcon[:,:,1])  
            box = [x1,y1,x2,y2]
            #print(u[y1//2:y2//2,x1//2:x2//2]>0)
            #print(u[y1//2:y2//2,x1//2:x2//2],v[y1//2:y2//2,x1//2:x2//2])

        return box,distanceup,flowup

