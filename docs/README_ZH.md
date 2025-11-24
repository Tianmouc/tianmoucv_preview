 # TianmouCV 中文文档 

 ![PyPI - Version](https://img.shields.io/pypi/v/tianmoucv) ![PyPI - Wheel](https://img.shields.io/pypi/wheel/tianmoucv) ![PyPI - License](https://img.shields.io/pypi/l/tianmoucv) ![PyPI - Downloads](https://img.shields.io/pypi/dm/tianmoucv) 

![usbmodule](/resources/usb_module.jpg)

 ## 简介
 这是首款互补视觉传感器（CVS）天眸（Tianmouc）的算法库。 
 
 更多项目详情请参阅[天眸传感器文档（需权限访问）](http://www.tianmouc.cn:30000)及[天眸CV项目主页](https://lyh983012.github.io//tianmoucv_doc_opensource/index.html) 
 
 ## 背景 
 天眸是全球首款多通路仿生视觉传感器，可同步实现高速（10000 fps）、高动态范围（130dB）、高灵敏度（530nm波长下72%，近红外波段）、高精度（10位RGB色彩，8位模式下757 fps至2位模式下10,000 fps）、低功耗及低带宽（较传统高速相机降低90%带宽）的视觉感知。其提供三种数据模态：基于帧的RGB图像、稀疏时间差分（TD）及稀疏空间差分（SD）。 
 
 天眸CV是天眸V1版（即将支持V2版）的官方算法库，包含天眸数据（.tmdat文件）编解码器、深度学习友好型数据集读取器、基础ISP工具、TSD特征描述、经典光流估计与灰度重建算法，以及一系列基础而实用的神经网络模型（涵盖重建、光流估计、去模糊、实例分割等领域）。 
 
 ## 安装指南 
 
 ### 环境准备 
 
 (1) Unix系统 
 
 安装cmake、make、g++>9及build-essential工具链 
 
 *特别说明*：对于NVIDIA Jetson模块，需安装JetPack SDK、PyTorch及TorchVision
 
 参考博客：[博客1](https://blog.csdn.net/weixin_44604409/article/details/132334866)、[博客2](https://zhuanlan.zhihu.com/p/437014069)
 
 (2) Windows系统 
 
 安装cmake、make及MinGW/Visual Studio，详情参见[文档（需权限）](http://www.tianmouc.cn:30000/tianmoucv/introduction.html)，或直接使用预编译的dll/pyd文件进行安装
 
 (3) 天眸SDK（可选）
 
 若需通过天眸CV直接读取连接PC的传感器数据，请先安装USB版天眸SDK：
 
 ```bash 
 cd tianmouc_sdk/usb/install/cyusb_linux_1.0.5
 sh install.sh 
 ``` 
 
 并通过执行install.sh脚本输入天眸SDK的绝对路径 
 
 ### 安装TianmouCV

 **建议使用anaconda提前装好pytorch-cuda环境，tianmoucv的pytorch要求很松，和其他项目耦合时可以最后安装**
 
 (1) PyPI安装（仅限稳定版0.3.5.0）：
 ```bash
 pip install tianmoucv 
 ```

 (2) 源码编译安装（推荐）：
 
 ```bash 
 git clone git@github.com:Tianmouc/tianmoucv.git 
 cd tianmoucv 
 sh install.sh 
 ``` 
 
 Windows系统请参考借助MinGW的install.bat脚本 
 
 *注：无需SDK功能时，在提示输入SDK路径时直接按回车键* 
 
 (3) 仅更新Python代码（适用于已编译库文件后的代码修改）： 
 
 ```bash 
 sh update.sh
 ``` 
 
 ## 数据准备
 
 可从[天眸文件服务器](http://www.tianmouc.cn:38328/index.php/s/2ptYY27g3eRMydG)下载演示数据片段，并参考[教程](https://github.com/Tianmouc/tianmoucv/blob/master/tianmoucv_example/introduction_to_tianmouc_data.ipynb)进行试用。 
 
 标准天眸数据结构示例如下： 
 
```
├── dataset
│   ├── matchkey
│   │   ├── cone
│   │       ├── info.txt
│   │       ├── xxx.tmdat
│   │   ├── rod
│   │       ├── info.txt
│   │       ├── xxx.tmdat
```
 
其中{matchkey}作为天眸数据读取器的过滤关键词使用 

## 应用案例 

功能示例存放于**tianmoucv_example**目录，多数示例可直接在Jupyter Notebook中运行（部分需SDK支持）： 

```bash 
conda activate [您的环境]
pip install jupyter lab
jupyter lab 
``` 

```
├── Tianmoucv_exmaple
│   ├── introduction_to_tianmouc_data: 天眸数据读取与可视化方法介绍
│   ├── >>>相机<<< 连接相机（需安装SDK）
│   │   ├── open_camera: 天眸数据接收与可视化（模板代码）
│   │   ├── qrcode_demo: 基于SD的二维码解码
│   │   ├── calibration_OpenCV: 相机标定工具
│   │   ├── deblur: （存在缺陷）使用TSD进行RGB去模糊
│   │   ├── realtime_inf: 实时实例分割
│   ├── >>>数据<<<
│   │   ├── covert_to_tmdat_and_calculate_bandwidth: 将np数组编码为tmdat格式并计算带宽
│   │   ├── rotate_tsd: 二维矢量场旋转技巧——SD处理
│   │   ├── convert_pcie_bin_to_tmdat: FPGA开发版本转换工具
│   ├── >>>处理<<<
│   │   ├── segmentation: 双通路融合实例分割
│   │   ├── feature_tracking_gray_sd: 纯SD特征追踪
│   │   ├── reconstructor:
│   │   │   ├──reconstruct_fuse_net: 最佳HDR-RGB视频神经网络重建器
│   │   │   ├──reconstruct_gray: 基于SD的灰度图像重建
│   │   │   ├──reconstruct_hdr_poisson_iter: 简易HDR融合方法
│   │   │   ├──reconstruct_original_nature_paper: 原版Nature论文中的RGB视频神经网络重建器
│   │   │   ├──reconstruct_recurrent: 基于TD训练的E2VID模型
│   │   │   ├──reconstruct_tiny_unet: 最快HDR-RGB视频神经网络重建器
│   │   ├── optical_flow:
│   │   │   ├──opticalflow_HS_method: TSD的HS光流估计器
│   │   │   ├──opticalflow_LK_method: TSD的LK光流估计器
│   │   │   ├──opticalflow_spynet: 原版Nature论文中的神经网络光流估计器
│   │   │   ├──opticalflow_RAFT: 最佳神经网络光流估计器
│   │   ├── denoise:
│   │   │   ├──denoise_tmdat_lvatf: 使用LVATF进行TSD去噪
│   │   ├── deblur
│   │   │   ├──deblur_stgdnet: 使用TSD进行RGB去模糊
│   ├── >>>模拟器<<<
│   │   │   ├──sim.ipynb: 通过视频输入运行RGB/TD/SD序列模拟器
```

## 维护团队 [@lyh983012](https://github.com/lyh983012) 

### 贡献者 

感谢所有贡献者的支持（详见贡献者列表图示）。 
特别致谢：王韬毅、陈雨过、孟亚鹏、李煜翔、杨琳 

## 许可协议 [GPLv3](LICENSE) © 林逸晗 
联系方式：linyh@xmu.edu.cn