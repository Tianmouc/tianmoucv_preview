![PyPI - Version](https://img.shields.io/pypi/v/tianmoucv) ![PyPI - Wheel](https://img.shields.io/pypi/wheel/tianmoucv) ![PyPI - License](https://img.shields.io/pypi/l/tianmoucv) ![PyPI - Downloads](https://img.shields.io/pypi/dm/tianmoucv) 

# TianMouCV-preview version

![usbmodule](/resources/usb_module.jpg)

**The official version will be available at [tianmoucv/tianmocv](https://github.com/Tianmouc/tianmoucv)**

This is the Python tool for the first complementary vision sensor (CVS), TianMouC.

More details about the project can be found on our project page. [Tianmouc Project](http://www.tianmouc.com:40000) and [TianmoucvProject](http://www.tianmouc.cn:40000/tianmoucv/introduction.html)
## Installation

(0) Prepare pytorch environment

**Python version should be larger than 3.8 and less than 3.12, recommend 3.10**

```bash
conda create -n [YOUR ENV NAME] --python=3.10
conda activate [YOUR ENV NAME]
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

(1) from PyPI

```bash
pip install tianmoucv
```

(2) Install from source codes (using pip):

```bash
git clone git@github.com:Tianmouc/Tianmoucv_preview.git
cd Tianmoucv_preview
sh install.sh
```

## Data

You can download a TianMouC data clip in [THU-sharelink](https://cloud.tsinghua.edu.cn/f/dc0d394efcb44af3b9b3/?dl=1), and refer to tianmoucv/exmaple/data/test_data_read.ipynb for a trial

a standard TianMouC dataset structure:

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

where matchkey is the sample name used for the TianMouC data reader 

## Examples

For some of the algorithms we've provided the example in tianmoucv/example

Including:

(1) calculating optical flow

(2) reconstruct gray/hdr images

(3) key point matching/tracking

(4) camera calibration

(5) data reeader

...

These samples can be directly run on jupyter notebook

```bash
conda activate [your environment]
pip install jupyter notebook
jupyter notebook
```

