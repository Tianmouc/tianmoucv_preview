#!/bin/bash
which python3
g++ -O2  -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` rod_decoder_py.cpp -o rod_decoder_py`python3-config --extension-suffix`
