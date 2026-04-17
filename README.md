# TianmouCV

![PyPI - Version](https://img.shields.io/pypi/v/tianmoucv) ![PyPI - Wheel](https://img.shields.io/pypi/wheel/tianmoucv) ![PyPI - License](https://img.shields.io/pypi/l/tianmoucv) ![PyPI - Downloads](https://img.shields.io/pypi/dm/tianmoucv) 

![usbmodule](/resources/usb_module.jpg)


---
- **中文文档见** [ README_ZH.md](./docs/README_ZH.md)
- **Development Log(各版本开发日志) see** [ DEVLOG.md](./docs/DEVLOG.md)
---

## Introduction

It is algorithms library for the first complementary vision sensor (CVS) Tianmouc.
The APIs of algorithms and various tools can be found in [Tianmoucv Project Page](https://lyh983012.github.io//tianmoucv_doc_opensource/index.html)
More details of the project can be found in [Tianmouc Sensor doc](http://www.tianmouc.cn:30000) 

## Table of Contents

- [Background](#background)
- [Installation](#installation)
	- [Environment Prepare](#environment-prepare)
    - [Compile and Install](#compile-and-install)
- [Usage](#Usage)
- [Examples](#Examples)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background

Tianmouc is the world's first multi-pathway brain-inspired vision sensor, which can simultaneously realize high-speed (10000 fps), high dynamics (130 dB), high-sensitivity (72%@530nm, NIR), high accuracy (10bit RGB, 757 fps@8bit - 10,000fps@2bit), low power consumption, and low bandwidth (90% bandwidth reduction compared to traditional high-speed cameras) visual perception. It provides three different data modalities: Frame-based RGB, sparse Temporal Difference and sparse Spatial Differences.

TianmouCV is the official algorithm library of Tianmouc V1 (and soon will support Tianmouc V2..), including codec of the Tianmouc data (.tmdat files), deeplearning-friendly dataset reader, basic ISP tools, feature descriptions of TSD, classical optical flow estimator and gray-scale reconstructor, some basic but useful neural-network (reconstruction, optical flow, deblur, instance segmentation and so on).

## Installation

### Environment Prepare

(1) Unix

-install cmake, make, g++>9, build-essential

(2) Windows

install cmake, make, minGW/VSstudio, details please refer to [doc(need permission)](http://www.tianmouc.cn:30000/tianmoucv/introduction.html), or you can use the pre-compiled dll/pyd for installation

(3) (optional)Tianmouc SDK

if you want to use TianmouCV to read tianmouc camera directly on your PC, please install tianomuc-SDK usb version first

```bash
cd tianmouc_sdk/usb/install/cyusb_linux_1.0.5
sh install
```

and input the absolute path of tianmoucsdk when you install TianmouCV by executing install.sh

### Compile and Install

(1) from PyPI

```bash
pip install tianmoucv
```

(2) Install from source codes:

```bash
git clone git@github.com:Tianmouc/tianmoucv_dev.git
cd tianmoucv_dev
sh install.sh
```

For windows, please refer to the "install.bat", with the help of winGW

**If you do not need to use the SDK, just press ENTER when inputing the SDK PATH**

(3) Only update python codes (If you have already compile the lib manually), it will be helpful if you want to save your time modifying some python codes in TianmouCV.

```bash
sh update.sh
```

## Prepare your TMDAT files

You can download a demo tianmouc data clip in [Tianmouc File Server](http://www.tianmouc.cn:38328/index.php/s/2ptYY27g3eRMydG), You can refer to the [tutorial](https://github.com/Tianmouc/tianmoucv_dev/blob/master/tianmoucv_example/introduction_to_tianmouc_data.ipynb) for a trial.

A standard tianmouc data structure is like:

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

The {matchkey} is the clip name used as a fileter key in Tianmouc data reader.


## Tianmouc Research paper list and public resources

- [TianmoucV1 Deblur (CVPR26)](https://tmcdeblur.github.io) [[dataset](https://huggingface.co/datasets/mypThu/SportsSlomo-CVS)]
- [TianmoucV1 Generative High-speed Reconstruction (ICCV25)](https://github.com/Tianmouc/GenRec) [[dataset](https://pan.baidu.com/share/init?surl=DFxXAAe0iVfPx-MPz2KVFQ&pwd=t6jw)]
- [Tianmouc self-supervised representation learnining](https://github.com/Tianmouc/TMC-SSL-Representation) [[dataset (preparing)]()]

For more, see [Tianmoucv Group Page](https://github.com/Tianmouc)

## Examples

For some of the function we've provided the example in **tianmoucv_example**. 

Most examples can be directly run on jupyter notebook, while some of them may need SDK support

```bash
conda activate [your environment]
pip install jupyter lab
jupyter lab
```

The examples are listed below

```
├── Tianmoucv example
│   ├── introduction_to_tianmouc_data: Introduce how to read and visualize Tianmouc data.
│   ├── >>>camera<<< connect to camera (SDK installation required).
│   │   ├── open_camera: receive and visualize Tianmouc data (template code)
│   │   ├── qrcode_demo:  Decode QR code based on SD.
│   │   ├── calibration_OpenCV: CameraCalib tool
│   │   ├── deblur: (buggy) deblur rgb using tsd
│   │   ├── realtime_inf: Realtime instance segmentation
│   ├── >>>data<<<
│   │   ├── covert_to_tmdat_and_calculate_bandwidth: Encode np array into the tmdat and calculate the bandwidth.
│   │   ├── rotate_tsd: trick for rotate 2D vector filed -- SD
│   │   ├── convert_pcie_bin_to_tmdat: tool for fpga dev version
│   ├── >>>proc<<<
│   │   ├── segmentation: dual pathway fusion instance segmentation
│   │   ├── feature_tracking_gray_sd: feature tracking based only on SD
│   │   ├── reconstructor:
│   │   │   ├──reconstruct_fuse_net: best HDR-RGB video NN-based reconstructor
│   │   │   ├──reconstruct_gray: Gray-scale image reconstruction based on SD
│   │   │   ├──reconstruct_hdr_poisson_iter: simple HDR fusion method
│   │   │   ├──reconstruct_original_nature_paper: RGB video NN-based reconstructor in original  Nature paper
│   │   │   ├──reconstruct_recurrent: E2VID trained on TD
│   │   │   ├──reconstruct_tiny_unet: fastest HDR-RGB video NN-based reconstructor
│   │   ├── optical_flow:
│   │   │   ├──opticalflow_HS_method: HS OF estimator with TSD
│   │   │   ├──opticalflow_LK_method: LK OF estimator with TSD
│   │   │   ├──opticalflow_spynet: NN-based OF estimator in original Nature paper
│   │   │   ├──opticalflow_RAFT: best NN-based OF estimator
│   │   ├── denoise:
│   │   │   ├──denoise_tmdat_lvatf:denoise TSD using LVATF
│   │   ├── deblur
│   │   │   ├──deblur_stgdnet: deblur rgb using tsd
│   ├── >>>simulator<<<
│   │   │   ├──sim.ipynb: run simulator for rgb/td/sd sequence with video input
```


We provide a example multi-thread realtime processing python script template in tianmoucv/camera/open_camera.py

you can directly run it if all the prerequisite are installed

## Maintainers

[@lyh983012](https://github.com/lyh983012).


### Contributors

This project exists thanks to all the people who contribute. 
<a href="https://github.com/Tianmouc/tianmoucv_dev/graphs/contributors"><img src="https://opencollective.com/tianmoucv/contributors.svg?width=890&button=false" /></a>

Thanks to: Taoyi Wang, Yuguo Chen, Yapeng Meng, Yuxiang Li, Lin Yang

## License

[GPLv3](LICENSE) © Yihan Lin

Please contact: linyh@xmu.edu.cn
