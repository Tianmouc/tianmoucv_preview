# TianMouCV

The algorithms library for complementary vision sensor Tianmouc.

More details of the project can be found in [doc](http://www.tianmouc.cn:38325)


## Prepare

if you want to use this lib to call tianmouc sensor v1 directly, please install tianomuc_sdk-usb version first

```bash
git clone https://git.tsinghua.edu.cn/wangtaoy20/tianmouc_sdk.git
cd tianmouc_sdk/usb/install/cyusb_linux_1.0.5
sh install
```

## Installation

(1) from PyPI

```bash
pip install tianmoucv
```

(2) Install from source codes (using pip):

```bash
#git clone https://github.com/Tianmouc/tianmoucv.git
git clone git@github.com:Tianmouc/tianmoucv.git
cd tianmoucv
sh install.sh
```

## Usage

For some of the fuction we've provided the example in tianmoucv/exmaple

Including:

(1) calculating optical flow

(2) reconstruct gray/rgb/hdr images

(3) key point matching/tracking

(4) camera calibration

(5) camera sdk api call

(6) data reeader test

...

These sample can be directly run on jupyter notebook

```bash
conda activate [your envirobnment]
pip install jupyter notebook
jupyter notebook
```

We also provide a example multi-thread realtime processing program template in tianmoucv/camera/open_camera.py

you can directly run it if all the prerequesite are installed

```bash
python tianmoucv/camera/open_camera.py
```

