import numpy as np
import cv2,time
import torch
from scipy.optimize import linear_sum_assignment

# ===============================================================
# ******描述子匹配****** 
# ===============================================================
#ratio=0.85:knn中前两个匹配的距离的比例
def feature_matching(des1:np.array, des2:np.array, ratio=0.85):
    """
    Match SIFT descriptors between two images.
    
    parameter:
        :param des1: kp list1,[x,y],list
        :param des2: kp list2,[x,y],list
        :param ratio: knn中前两个匹配的距离的比例,筛选匹配得足够好的点, float

    """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    if len(matches) > 0:
        for m,n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append([m])
    return good_matches



def mini_l2_cost_matching(des1, des2, num=20):
    """
    Match SIFT descriptors between two images.
    
    parameter:
        :param des1: kp list1,[x,y],list
        :param des2: kp list2,[x,y],list
        :param num: 筛选匹配得足够好的点, int

    """
    distance_matrix = np.zeros((len(des1), len(des2)))
    for i, ref_feature in enumerate(des1):
        for j, query_feature in enumerate(des2):
            distance_matrix[i, j] = np.linalg.norm(ref_feature - query_feature)
    matched_indices = linear_sum_assignment(distance_matrix)
    matches = []
    for i in range(len(matched_indices[0])):
        matches.append((matched_indices[0][i],matched_indices[1][i]))
    matches = matches[:num]
    return matches

# ===============================================================
# ******刚性对齐****** 
# ===============================================================
def align_images(image, kpList1, kpList2,matches, canvas=None):
    """
    单应性矩阵，刚性对齐
    
    parameter:
        :param image: np图像,np.array
        :param kpList1: kp list [x,y],list
        :param kpList2: kp list [x,y],list
        :param matches: 匹配点列表 [id1,id2],list

    """
    H = None
    src_pts = []
    dst_pts = []
    if(len(matches)>4):
        if isinstance(matches[0],tuple):
            src_pts = np.float32([kpList1[m[0]] for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpList2[m[1]] for m in matches]).reshape(-1, 1, 2)
        else:
            src_pts = np.float32([kpList1[m[0].queryIdx] for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpList2[m[0].trainIdx] for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(H)
    for i in range(len(src_pts)):
        y1 , x1 = int(src_pts[i][0][0]),int(src_pts[i][0][1])
        y2 , x2 = int(dst_pts[i][0][0]),int(dst_pts[i][0][1])
        if canvas is not None:
            cv2.line(canvas,(x1,y1),(x2+640,y2),(255,0,0))
        print(x1,',',y1,'--->',x2,',',y2)
    w,h = image.shape[1],image.shape[0]
    
    imagewp = image
    if H is not None:
        imagewp = cv2.warpPerspective(image,H, (w,h))
    return imagewp,H

# ===============================================================
# ******朴素的tracking 类****** 
#- 输入sd图像，uint8
#- 输出追踪历史轨迹和可视化结果
# ===============================================================
class Feature_tracker_sd:

    def __init__(self,d_factor=0.5, fix_update_frame = 250):
        self.fl_aim = []
        self.kp_aim = []
        self.history = dict([])
        self.tracking_count = 0
        self.d_factor = d_factor
        self.fix_update_frame = 250
        print('fix_update_frame:', 250)

    def reset(self):
        self.fl_aim = []
        self.kp_aim = []
        self.history = dict([])
        self.tracking_count = 0
        
    def update(self,sd_img,viz=True):

        sd_img = (sd_img-torch.min(sd_img))/(torch.max(sd_img)-torch.min(sd_img)) *  255
        sd_img = sd_img.numpy().astype(np.uint8)
        
        #第1步：计算两张图对应Harris角点检测
        startT = time.time()
        sift = cv2.SIFT_create()
        good_kp, sift_feature_List = sift.detectAndCompute(sd_img, None)
        endT = time.time()
        kp = [(p.pt[1],p.pt[0]) for p in good_kp]
        fl = sift_feature_List
        imshow = sd_img.copy()
        imshow = np.stack([imshow]*3,axis=2)
        
        #第3步：更新待追踪特征点列表
        if self.tracking_count % self.fix_update_frame == 0 or len(self.fl_aim)==0:
            print('update tracking target')
            self.kp_aim = kp
            self.fl_aim = fl
            self.history = dict([])
            for i in range(len(self.kp_aim)):
                self.history[i] = [ kp[i] ]
        else:
            if len(fl)>0:
                matches = feature_matching(self.fl_aim,fl,ratio=self.d_factor)
                #只要匹配上，就更新待追踪点坐标和对应的特征描述子，以免场景变化过大影响追踪
                for m in matches:
                    src_pts = self.kp_aim[m[0].queryIdx]
                    dst_pts = kp[m[0].trainIdx]
                    dist = (src_pts[0]-dst_pts[0])**2 + (src_pts[1]-dst_pts[1])**2
                    if dist < 1600:
                        self.history[m[0].queryIdx].append(kp[m[0].trainIdx])
                        self.kp_aim[m[0].queryIdx] = kp[m[0].trainIdx]
                        self.fl_aim[m[0].queryIdx,:] = fl[m[0].trainIdx,:]

                #绘制追踪结果
                if viz:
                    for k in self.history:
                        traj = self.history[k]
                        y2, x2 = (None,None)
                        for kp_i in traj:
                            y1, x1 = (int(kp_i[0]),int(kp_i[1]))
                            if not x2 is None:
                                cv2.line(imshow,(x1,y1),(x2,y2),(0,255,0))
                                cv2.circle(imshow,(x1,y1),2,(0,0,255))
                            y2 = y1
                            x2 = x1
                        cv2.circle(imshow,(x2,y2),2,(255,0,0)) 
            else:
                print('no useable new feature')
        self.tracking_count += 1
        
        return self.history, imshow
