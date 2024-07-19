import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

# ===============================================================
# ******描述子匹配****** 
# ===============================================================
#ratio=0.85:knn中前两个匹配的距离的比例
def feature_matching(des1, des2, ratio=0.85):
    """
    Match SIFT descriptors between two images.
    
    parameter:
        :param des1: kp list1,[x,y],list
        :param des2: kp list2,[x,y],list
        :param ratio: knn中前两个匹配的距离的比例,筛选匹配得足够好的点, float

    """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1.cpu().numpy(), des2.cpu().numpy(), k=2)
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

