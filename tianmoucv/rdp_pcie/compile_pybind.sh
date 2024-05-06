#!/bin/bash
PYTHON_PATH=$(which python)
PYTHON_VERSION=$(python -c 'import sys; print(sys.version_info.minor)')
# 根据Python解释器的路径获取site-packages目录
SITE_PACKAGES_PATH=$(dirname $(dirname $(echo $PYTHON_PATH)))'/lib'

g++ -O2 -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` -L$SITE_PACKAGES_PATH -lpython3'.'$PYTHON_VERSION rod_decoder_py.cpp -o rod_decoder_py`python3-config --extension-suffix`
