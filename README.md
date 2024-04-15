![PyPI - Version](https://img.shields.io/pypi/v/tianmoucv) ![PyPI - Wheel](https://img.shields.io/pypi/wheel/tianmoucv) ![PyPI - License](https://img.shields.io/pypi/l/tianmoucv) ![PyPI - Downloads](https://img.shields.io/pypi/dm/tianmoucv) 

# TianMouCV-preview version

**The official version will be available at [tianmoucv/tianmocv](https://github.com/Tianmouc/tianmoucv)**

The python tool for complementary vision sensor Tianmouc.

More details of the project can be found in our main page [doc](http://www.tianmouc.cn:38325)


## Installation

(1) from PyPI

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

## Data

You can download a tianmouc data clip in [THU-sharelink](https://cloud.tsinghua.edu.cn/f/dc0d394efcb44af3b9b3/?dl=1), You can refer to tianmoucv/exmaple/data/test_data_read.ipynb for a trial

a standard tianmouc dataset structure:

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

where matchkey is the sample name used for tianmouc data reader 

## Examples

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

