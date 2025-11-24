import numpy as np
import cv2
import math
import sys


###图像熵计算
def image_entropy(image_path):
    # 读取图像
    image = cv2.imread(image_path, 0)

    # 计算图像像素值的概率分布
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram /= histogram.sum()

    # 计算图像熵
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))

    return entropy


def flicker_frequency_test(freq_test_queue):
    fs = 750
    N = len(freq_test_queue)
    startTime = time.time()        
    fft_result = np.fft.fft(freq_test_queue)
    # 获取频率轴
    freq_axis = np.fft.fftfreq(len(freq_test_queue))
    # 频谱对称*2,shift
    test_freq_result = np.abs(fft_result[len(fft_result)//2+1:])*2
    # 获取分量最大的几个频率
    num_top_frequencies = 5
    top_frequencies = np.argsort(-test_freq_result)[:num_top_frequencies]
    top_amplitudes = test_freq_result[top_frequencies]
    # 输出结果
    for i in range(num_top_frequencies):
        print("TOP",i,' freq Hz', freq_axis[top_frequencies[i]] * fs, "Hz, A:",abs(top_amplitudes[i]))
    endTime = time.time()
    print('time cost:',endTime-startTime,'s','len(fft_result):',len(fft_result))
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    x = np.arange(0,N/fs,1/fs)[:len(freq_test_queue)]
    ax1.plot(x,freq_test_queue)
    ax1.set_xlabel('time/s')
    ax1.set_ylabel('sum(TD)')

    x = np.arange(0,fs,fs/N)[:len(test_freq_result)]
    ax2.set_xlabel('freq/Hz')
    ax2.set_ylabel('sum(TD)')
    ax2.plot(x,test_freq_result)
        
    plt.show()
    
    
    
#!/usr/bin/env python



def AE_RGB_MSV(cv_image,camera_state = None,desired_msv = 2.5,focus_region_mask=None):
    """
    automatic exposure control based on the paper
    "Automatic Camera Exposure Control", N. Nourani-Vatani, J. Roberts
    https://github.com/alexzzhu/auto_exposure_control/tree/master
    """
    if camera_state is None:
        camera_state = dict([])
        camera_state['err_i'] = 0
        camera_state['exp_time'] = 10
        camera_state['exp_gain'] = 1
        camera_state['max_exp_time'] = 100
        camera_state['min_exp_time'] = 1

    err_i = camera_state['err_i']
    rows, cols, channels = cv_image.shape
    
    
   #给出关注的光照区域
    if not focus_region_mask is None:
        brightness_image = brightness_image[focus_region_mask]

        
    if (channels == 3):
        brightness_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)[:,:,2]
    else:
        brightness_image = cv_image
    
    hist = cv2.calcHist([brightness_image],[0],None,[11],[0,256])
  
    mean_sample_value = 0
    for i in range(len(hist)):
        mean_sample_value += hist[i]*(i+1)
        
    mean_sample_value /= (rows*cols)
    mean_sample_value = float(mean_sample_value)

    # Gains
    k_p = 0.2
    k_i = 0.05
    # Maximum integral value
    max_i = 3
    err_p = desired_msv-mean_sample_value
    
    err_i += err_p
    if abs(err_i) > max_i:
        err_i = np.sign(err_i)*max_i
        
    #print('hist:',hist,' mean_sample_value:',mean_sample_value,' err_p:',err_p,' err_i:',err_i)
 
    exp_time = camera_state['exp_time']
    exp_gain = camera_state['exp_gain']
    # Don't change exposure if we're close enough. Changing too often slows
    # down the data rate of the camera.
    #effective_exp_time = exp_time*exp_gain

    if abs(err_p) > 0.5:
        exp_time += (k_p*err_p+k_i*err_i)/exp_gain
        
        if exp_time < 0:
            exp_time = camera_state['min_exp_time']
            exp_gain = 1
        
        if exp_time > camera_state['max_exp_time']:
            exp_time = camera_state['max_exp_time']
            exp_gain = 16
        
        if exp_time > camera_state['max_exp_time'] and exp_gain <16: 
            exp_gain *= 2
            exp_time /= 2
            
        if exp_time < camera_state['min_exp_time'] and exp_gain >1:
            exp_gain /= 2
            exp_time *= 2        
            
    camera_state['exp_time'] = exp_time 
    camera_state['exp_gain'] = exp_gain            
    return camera_state, mean_sample_value
