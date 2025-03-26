# package
# __init__.py
import random
import cv2
import math
import torch
import os
import sys

try:
    from tianmoucv.rdp_usb import rod_decoder_py as rdc
except:
    print("WARNING: no decoder found, try to auto compile")
    print('If you still get this message,please try:\n 1. run it in a python script (only once) \n 2. installfrom source code (install.sh) to see what happened')
    
    current_file_path = os.path.abspath(__file__)
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
    

# __init__.py
__all__ = ['random', 'cv2', 'math', 'torch','os','sys']
__author__ = 'Y. Lin'
__contributor__ = 'T. Wang, Y. Chen, Y. Li'
__authorEmail__ = '532109881@qq.com'

print('TianMouCV™ 0.3.4.3, via',__author__,' add codec and bw calculation')

