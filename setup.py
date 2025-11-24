import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install

with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = "TianmouCV is the official algorithm library of Tinamouc V1 (and soon will support Tianmouc V2..), including codec of the Tianmouc data (.tmdat files), deeplearning-friendly dataset reader, basic ISP tools, feature descriptions of TSD, classicle optical flow estimator and gray-sacle reconstructor, some basic but useful neural-network (reconstruction, opticla flow, deblur, instance segmentation and so on)."

'''
major.minor.patch.
主版本：当出现不兼容的 API 变动时，版本号会递增。
小版本：在以向后兼容的方式添加功能时递增。
PATCH 版本：在进行向后兼容的错误修复时递增。
'''
try:
    if not ONLY_UPDATE_PYTHON in os.environ:
        print("WARNING: no decoder found, try to auto compile")
        print('If you still get this message,please try:\n 1. run it in a python script (only once) \n 2. installfrom source code (install.sh) to see what happened')
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        aim_path = os.path.join(current_file_path,'tianmoucv','rdp_usb')
        os.chdir(aim_path)
        current_path = os.getcwd()
        print("Current Path:", current_path)
        subprocess.run(['sh', './compile_pybind.sh'])
        aim_path = os.path.join(current_file_path,'tianmoucv','rdp_pcie')
        os.chdir(aim_path)
        current_path = os.getcwd()
        print("Current Path:", current_path)
        subprocess.run(['sh', './compile_pybind.sh'])
        print('compile decoder successfully!')
except:
    print('[FATAL ERROR]: Fail to compile rdp pakage, please use ./install.sh to install tianmoucv')

setup(
    name='tianmoucv',                     # 模块的名称
    version='0.4.0.0',                    # 版本号
    author='Yihan Lin,Taoyi Wang',        # 作者名称
    author_email='532109881@qq.com',      # 作者邮箱
    description='Algorithms library for Tianmouc sensor_preview version',   # 简要描述
    url='https://github.com/Tianmouc/tianmoucv',  # 项目主页的URL
    packages=find_packages(),   # 告诉 setuptools 自动找到要安装的包
    package_data = {'':['rdp_pcie/*',
                        'rdp_usb/*',
                        'camera/*',
                        'data/blc/*',
                        'sim/*',
                        'camera/lib/*']},
    include_package_data=True,
    install_requires=install_requires,
    long_description=long_description,
    # 可选的内容
    keywords='tianmoucv',           # 关键词
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        
    ],
    python_requires='>=3.8'
)
