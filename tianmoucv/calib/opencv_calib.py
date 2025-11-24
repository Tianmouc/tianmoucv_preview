################################################################
# Author: Yuguo Chen
# opencv_calib基于OpenCV实现棋盘格标定
#     将原始例程进行封装以便处理便捷
#     TODO: 考虑calibration_opencv_singleSB新版函数, 考虑多张图像联合标定
################################################################

import numpy as np
import cv2
import time
import torch
import matplotlib.pyplot as plt

cv2.ocl.setUseOpenCL(False) # 避免opencv统一内存占用显存

################################################################
# calibration_opencv对任意张灰度图案按OpenCV算法进行棋盘格标定
#     参考网址: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
#     参考网址: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
#     输入I_in_list为图像列表, 每张图片要求numpy或torch, H*W或H*W*3(rgb顺序), 整数或浮点
#     输入board_size为棋盘格尺寸, 为方块的数目, 角点数目为(H-1)*(W-1)
#         注意本处的尺寸定义与OpenCV本身不同, OpenCV需要输入角点数目
#     输入square_size为方格尺寸, 建议单位为mm
#     输入en_draw默认为0, 设1则输出可视化图像, en_draw同时是可绘制图像的最大数目
#     输入en_time默认为0, 设1则输出检测棋盘格用时
#     输入factor_resize默认为2, 表示图像分辨率扩展倍数
#         ! 注意, OpenCV工作原理为检测黑色方块, 棋盘格过小可能导致检测失败
#         ! 注意, 根据经验, 等比例扩展至1280*640再检测, 但内参以原分辨率为准
#     输入en_SB默认为1
#         1表示使用findChessboardCornersSB(), 为新版方法, 基于BMVC2018开发, 自带亚像素
#         0表示使用findChessboardCorners + cornerSubPix, 为传统方法
#     输出变量为字典, 包括
#         'found_list'为各个图像是否检测成功的指示
#         'camera_matrix'为内参矩阵
#         'distortion_coeffs'为畸变系数(k1, k2, p1, p2, k3)
#         'rotation_vectors'与'translation_vectors'为外参的旋转与平移向量
#         'mean_reproj_error'为平均重投影误差
#         'corner_sub_list'为图像中棋盘格的亚像素坐标
#         ！注意外参平移与旋转向量，亚像素坐标均为列表, 与found_list中1的位置对应
################################################################
def calibration_opencv(I_in_list, board_size, square_size, en_draw = 0, factor_resize = 2, en_time = 0, en_SB = 1):
    
    #### 设置扩展分辨率
    img_H, img_W = I_in_list[0].shape[0:2] # 默认320*640 
    F_resize = factor_resize               # 默认为2
    if img_W * F_resize < 1280:            # 扩展后图像宽度不小于1280
        F_resize = int(np.ceil(1280 / img_W))
    img_H, img_W = img_H * F_resize, img_W * F_resize  

    #### 输入图像规范化
    img_resize_list = []
    for I_in in I_in_list:    
        img_in = check_image_input_forOpenCV(I_in) # 约束为numpy, 灰度, uint8. 并复制作为缓冲
        img_resize = cv2.resize(img_in, (img_W, img_H), interpolation=cv2.INTER_NEAREST)
        img_resize_list.append(img_resize)

    #### 预先设置OpenCV标定参数
    if board_size[0] > board_size[1]: 
        # OpenCV要求输入角点规模, 本处默认[0]>[1]
        corner_size  = (board_size[0] - 1, board_size[1] - 1)
    else:
        corner_size  = (board_size[1] - 1, board_size[0] - 1)    
    criteria    = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 亚像素迭代终止条件
    objp        = np.zeros((corner_size[0] * corner_size[1], 3), np.float32)                
    objp[:, :2] = np.mgrid[0:corner_size[0], 0:corner_size[1]].T.reshape(-1, 2) * square_size
        # 构建理想棋盘, (0,0,0), (1,0,0), (2,0,0) ....,([0]-1,[1]-1,0)

    #### OpenCV逐一图像查找棋盘格角点
    objpoints  = [] # 世界坐标系, 三维点
    imgpoints  = [] # 图像坐标系, 二维点
    found_list = [0] * len(img_resize_list) # 0表示对应位置图像检测失败
    if en_time == 1: start = time.perf_counter()
    if en_SB == 1:
        for i, img_resize in enumerate(img_resize_list):
            ret, corners_sub = cv2.findChessboardCornersSB(img_resize, corner_size, None)
            if ret == True: # 检测成功
                objpoints.append(objp)        # 保存一组3D-2D数据
                imgpoints.append(corners_sub) # 保存一组3D-2D数据
                found_list[i] = 1
    else:
        for i, img_resize in enumerate(img_resize_list):
            ret, corners = cv2.findChessboardCorners(img_resize, corner_size, None)
            if ret == True: # 检测成功, 额外执行亚像素处理
                corners_sub = cv2.cornerSubPix(img_resize, corners, (11,11), (-1,-1), criteria)
                objpoints.append(objp)        # 保存一组3D-2D数据
                imgpoints.append(corners_sub) # 保存一组3D-2D数据
                found_list[i] = 1
    # 打印运行时间
    if en_time == 1: 
        end = time.perf_counter()
        print(f"OpenCV detection checkerboard time: {(end-start)*1000:.3f} ms, average time per frame: {(end-start)*1000/len(found_list):.3f} ms.")    
    # 统计检测结果
    print(f"Total images: {len(found_list)}, successfully detected: {sum(found_list)}, detection rate: {sum(found_list)/len(found_list):.2%}.")
    # 没有检测到任何棋盘格则直接终止
    if sum(found_list) == 0: print("No checkerboard detected!"); return {} 
    # 结果可视化
    if en_draw:
        draw_count = 0
        for i, img_resize in enumerate(img_resize_list):
            if found_list[i] == 1:
                img_out     = np.stack([img_resize, img_resize, img_resize], axis=-1)
                img_corners = cv2.drawChessboardCorners(img_out, corner_size, imgpoints[draw_count], True)
                draw_count  = draw_count + 1
                if draw_count > en_draw: break # 最多输出en_draw张图片
                plt.imshow(img_corners)
                plt.show() 
                 
    # OpenCV拟合标定参数
    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_resize.shape[::-1], None, None)
    # 计算重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, distortion_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        total_error += error
    # 重新缩放回原始分辨率, 并打印结果
    camera_matrix[:2, :] /= F_resize
    imgpoints  = [matrix / F_resize for matrix in imgpoints]
    mean_error = total_error / len(objpoints) / F_resize 
    print("Intrinsic parameters: ")
    print(np.round(camera_matrix, 3))
    print("Extrinsic parameters [img-0] - rotation matrix: ")
    print([float(round(val, 3)) for val in rvecs[0].flatten()])
    print("Extrinsic parameters [img-0] - translation matrix: ")
    print([float(round(val, 3)) for val in tvecs[0].flatten()])
    print(f"total re-projection error: {mean_error:.3f} pix.")

    # 构建包含所有标定结果的字典
    calib_result = {
        'found_list':          found_list,        # 是否检测成功
        'camera_matrix':       camera_matrix,     # 相机内参
        'distortion_coeffs':   distortion_coeffs, # 畸变系数
        'rotation_vectors':    rvecs,             # 外参-旋转向量(found_list中1的位置)
        'translation_vectors': tvecs,             # 外参-平移向量(found_list中1的位置)
        'mean_reproj_error':   mean_error,        # 重投影误差
        'corner_sub_list':     imgpoints          # 亚像素角点(found_list中1的位置)
    }

    return calib_result
        
# ################################################################
# # calibration_opencv_single对**单张**灰度图案按OpenCV算法进行棋盘格标定
# #     ! 注意, OpenCV工作原理为检测黑色方块, 棋盘格过小可能导致检测失败
# #     ! 注意, 根据经验, 等比例扩展至1280*640再检测, 但内参以原分辨率为准
# #     参考网址: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# #     输入I_in要求numpy或torch, H*W或H*W*3(rgb顺序), 整数或浮点
# #     输入board_size为棋盘格尺寸, 为方块的数目, 角点数目为(H-1)*(W-1)
# #     输入square_size为方格尺寸, 建议单位为mm
# #     输入draw默认为0, 设1则输出可视化图像
# #     输入factor_resize默认为2
# #     输出camera_matrix内参, rvecs旋转外参, tvecs平移外参
# #     输出corners_sub检测到的亚像素角点, 作为中间过程
# ################################################################
# def calibration_opencv_single(I_in, board_size, square_size, draw=0, factor_resize = 2):
    
#     # 检查输入: 约束为numpy, 灰度, float32. 并复制作为缓冲
#     img_in = check_image_input_forOpenCV(I_in)
#     img_H, img_W = img_in.shape # 320*640 or 160*320
#     # 扩展分辨率
#     F_resize = factor_resize # 默认图像扩大1倍
#     if img_H <= 160: F_resize = F_resize * 2
#     img_H, img_W = img_H * F_resize, img_W * F_resize
#     img_resize = cv2.resize(img_in, (img_W, img_H), interpolation=cv2.INTER_NEAREST)
#     img_resize = (img_resize*255).astype(np.uint8)  # 将图像转为255(uint8)的形式

#     # OpenCV查找棋盘格角点
#     board_size = (board_size[0] - 1, board_size[1] - 1) # OpenCV要求输入角点规模
#     found, corners = cv2.findChessboardCorners(img_resize, board_size, None)

#     # 处理棋盘格角点
#     criteria    = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)         # termination criteria
#     objp        = np.zeros((board_size[0] * board_size[1], 3), np.float32)                # prepare object points
#     objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)*square_size # prepare object points   
#     if not found:
#         print('Not found checkerboard!')
#         return None, None, None, None
#     else: 
#         # OpenCV亚像素处理
#         # corners_sub  = cv2.cornerSubPix(img_resize, corners, (11, 11), (-1, -1), criteria)
#         corners_sub = corners.copy()
#         if draw: # 将检测结果绘制在画面中
#             img_out     = np.stack([img_resize, img_resize, img_resize], axis=-1)
#             img_corners = cv2.drawChessboardCorners(img_out, board_size, corners_sub, True)
#             plt.imshow(img_corners)
#             plt.show() 
#         # OpenCV拟合标定参数
#         ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera([objp], [corners_sub], img_resize.shape[::-1], None, None)
#         imgpoints2, _ = cv2.projectPoints(objp, rvecs[0], tvecs[0], camera_matrix, distortion_coeffs) # 计算重投影点
#         error = cv2.norm(corners_sub, imgpoints2, cv2.NORM_L2)/len(imgpoints2)                       # 计算重投影误差
#         # 重新缩放回原始分辨率
#         camera_matrix[:2, :] /= F_resize
#         corners_sub = corners_sub / F_resize
#         error = error / F_resize 
#         if draw: 
#             print("[single img] Intrinsic parameters: ")
#             print(camera_matrix)
#             print("[single img] distortion coefficients: ")
#             print(distortion_coeffs)
#             print("[single img] Extrinsic parameters - rotation matrix: ")
#             print(rvecs[0])
#             print("[single img] Extrinsic parameters - translation matrix: ")
#             print(tvecs[0])
#             print( "[single img]reprojection error: {} pix.".format(error) )

#     # 构建包含所有标定结果的字典
#     calib_result = {
#         'camera_matrix':       camera_matrix,     # 相机内参
#         'distortion_coeffs':   distortion_coeffs, # 畸变系数
#         'rotation_vectors':    rvecs,             # 外参-旋转向量(长度为1)
#         'translation_vectors': tvecs,             # 外参-平移向量(长度为1)
#         'mean_reproj_error':   error,             # 重投影误差
#         'corner_sub_list':     corners_sub        # 亚像素角点(长度为1)
#     }

#     return calib_result


################################################################
# check_image_input_forOpenCV将输入图像约束为numpy, 灰度, uint8. 并复制作为缓冲
#     处理过程与checkerboard_detect类似, 为避免命名冲突而独立命名
#     输入必须是numpy或torch, H*W或H*W*3(rgb顺序), 整数或浮点
################################################################
def check_image_input_forOpenCV(I_in):

    # 检查变量类型
    if isinstance(I_in, np.ndarray): 
        img_in = I_in.copy() # 复制一份输入
    elif isinstance(I_in, torch.Tensor): # tensor转numpy
        img_in = I_in.numpy()
    else: 
        raise TypeError("I_in type must be Tensor / numpy!")

    # 检查数据类型并转换为uint8
    if img_in.dtype in (np.float32, np.float64):
        if not (np.min(img_in) >= 0.0 and np.max(img_in) <= 1.0):
            raise ValueError("Float image values must be in the range [0, 1].")
        img_in = (img_in * 255.0).astype(np.uint8)
    elif img_in.dtype == np.uint8:
        pass  # 保持原样
    else:
        raise TypeError("Image must be of type float32, float64 (0~1) or uint8 (0~255).")
       
    # 转灰度
    if len(img_in.shape) == 3 and img_in.shape[2] == 3:
        img_in = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
    elif len(img_in.shape) != 2:
        raise ValueError("Image must be H*W or H*W*3.")

    return img_in

################################################################
# evaluate_distortion评估畸变情况
#     输入为distortion_coeffs, 来自opencv标定的输出, 顺序[k1,k2,p1,p2,k1]
################################################################
def get_simple_distortion(camera_matrix, distortion_coeffs, image_size):
    # 1. 取图像边缘代表性点（避开中心，选4个角附近点计算平均畸变）
    h, w = image_size
    edge_points = np.float32([[w*0.1, h*0.1], [w*0.9, h*0.1],  # 左上、右上
                              [w*0.1, h*0.9], [w*0.9, h*0.9]]) # 左下、右下
    # 2. 计算无畸变的理想位置
    undistorted = cv2.undistortPoints(edge_points, camera_matrix, distortion_coeffs, P=camera_matrix).squeeze()
    # 3. 计算图像中心（主点）
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    # 4. 计算每个边缘点的畸变百分比
    distortions = []
    for (x1, y1), (x2, y2) in zip(edge_points, undistorted):
        actual_r = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)  # 实际距离
        ideal_r  = np.sqrt((x2 - cx)**2 + (y2 - cy)**2)   # 理想距离
        if ideal_r > 1e-3:  # 避免除以0
            dist_pct = abs((actual_r - ideal_r) / ideal_r * 100)
            distortions.append(dist_pct)

    # 5. 取平均畸变百分比，判断等级
    avg_dist = np.mean(distortions)
    if avg_dist < 0.5:
        result = f"#### 畸变百分比是{avg_dist:.2f}%，评级为微畸变"
    elif avg_dist < 1:
        result = f"#### 畸变百分比是{avg_dist:.2f}%，评级为小畸变"
    else:
        result = f"#### 畸变百分比是{avg_dist:.2f}%，建议进行畸变校准"
    
    print(result)

################################################################
# calculate_angles评估2个三维平面之间的夹角, 并直接打印
#     输入为来自opencv的旋转矢量, 尺寸(3, 1)
################################################################
def calculate_angles(rotation_vector):
    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # 从旋转矩阵提取欧拉角 (使用ZYX顺序)
    sy = np.sqrt(rotation_matrix[0,0] **2 + rotation_matrix[1,0]** 2)
    
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
        y = np.arctan2(-rotation_matrix[2,0], sy)
        z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
    else:
        x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
        y = np.arctan2(-rotation_matrix[2,0], sy)
        z = 0
    
    # 将弧度转换为度
    x_angle = np.degrees(x)
    y_angle = np.degrees(y)
    z_angle = np.degrees(z)
    
    # 计算两个平面之间的夹角
    # 旋转向量的模长等于旋转角度(弧度)
    angle_rad = np.linalg.norm(rotation_vector)
    total_angle = np.degrees(angle_rad)
    
    # 直接打印结果
    print(f"#### 成像面与显示面夹角为{total_angle:.2f}度，其中X轴{x_angle:.2f}度，Y轴{y_angle:.2f}度，Z轴{z_angle:.2f}度")


################################################################
# calculate_display_coordinates用于计算显示屏上理想棋盘格各角点坐标
#     本函数主要服务于yapeng
#     输入corner_size为角点尺寸, 例如(12, 9)
#     输入start_x, start_y为左上角起点的坐标, 单位为像素
#     输入square_size为棋盘格方格的尺寸, 单位为像素
#     返回display_coords: 形状为(rows, cols, 2)的数组，包含所有角点的显示屏坐标
################################################################
def calculate_display_coordinates(corner_size, start_x, start_y, square_size):

    rows, cols = corner_size
    display_coords = np.zeros((rows, cols, 2), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            # 计算显示屏坐标
            display_coords[i, j, 0] = start_x + i * square_size
            display_coords[i, j, 1] = start_y + j * square_size
    
    return display_coords