![PyPI - Version](https://img.shields.io/pypi/v/tianmoucv)

# TianMouCV-preview version

**The official version will be available at [tianmoucv/tianmocv](https://github.com/Tianmouc/tianmoucv)**

The python tool for complementary vision sensor Tianmouc.

More details of the project can be found in our main page [doc](http://www.tianmouc.cn:38325)


## Installation

(1) from PyPI(not ready)

```bash
pip install tianmoucv
```

(2) Install from source codes (using pip):

**Python version should be larger than 3.8 and less than 3.12, recommand 3.10**

```bash
conda create -n [YOUR ENV NAME] --python=3.10
conda activate [YOUR ENV NAME]
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
git clone git@github.com:Tianmouc/Tianmoucv_preview.git
cd Tianmoucv_preview
sh install.sh
```

## Usage

For some of the fuction we've provided the example in tianmoucv/exmaple

Including:

(1) calculating optical flow

(2) reconstruct gray/hdr images

(3) key point matching/tracking

(4) camera calibration

(5) data reeader

...

These sample can be directly run on jupyter notebook

```bash
conda activate [your envirobnment]
pip install jupyter notebook
jupyter notebook
```

