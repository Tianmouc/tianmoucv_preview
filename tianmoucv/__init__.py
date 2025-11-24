# package
# __init__.py
import random
import cv2
import math
import torch
import os
import sys
from importlib.metadata import version

# __init__.py
__all__ = ['random', 'cv2', 'math', 'torch','os','sys']
__author__ = 'Y. Lin'
__contributor__ = 'T. Wang, Y. Chen, Y. Li'
__authorEmail__ = '532109881@qq.com'

try:
    from tianmoucv.rdp_usb import rod_decoder_py as rdc
except:
    import subprocess
    print("WARNING: no decoder found, try to auto compile")
    print('If you still get this message,please try:\n 1. run it in a python script (only once) \n 2. installfrom source code (install.sh) to see what happened')
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    aim_path = os.path.join(current_file_path,'rdp_usb')
    os.chdir(aim_path)
    current_path = os.getcwd()
    print("Current Path:", current_path)
    subprocess.run(['sh', './compile_pybind.sh'])
    aim_path = os.path.join(current_file_path,'rdp_pcie')
    os.chdir(aim_path)
    current_path = os.getcwd()
    print("Current Path:", current_path)
    subprocess.run(['sh', './compile_pybind.sh'])
    from tianmoucv.rdp_usb import rod_decoder_py as rdc
    print('compile decoder successfully!')
    

def limit_threads():
    import multiprocessing
    # 获取CPU核心数
    total_cores = multiprocessing.cpu_count()
    desired_cores = min(max(1, total_cores // 4),32)
    # 设置环境变量
    os.environ['OMP_NUM_THREADS'] = str(desired_cores)
    os.environ['OPENBLAS_NUM_THREADS'] = str(desired_cores)
    os.environ['MKL_NUM_THREADS'] = str(desired_cores)
    # 设置OpenCV
    cv2.setNumThreads(desired_cores)
    # 设置PyTorch
    torch.set_num_threads(desired_cores)

    print(f"试验功能:TianMouCV将限制单进程中opencv与pytorch默认的线程上限为CPU总核心数的1/4(<=32): {desired_cores}/{total_cores}")
    print(f"如不想设置限制，请在环境变量中加入标记： DISABLE_TMCV_LIMIT_CORE ")

# 调用函数
print('TianMouCV™ version:',version('tianmoucv'),', via',__author__,' see DEVLOG.md')

if 'DISABLE_TMCV_LIMIT_CORE' not in os.environ:
    limit_threads()
