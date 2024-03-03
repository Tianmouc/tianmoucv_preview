
1. complile the cpp code, must use python with pybind11!

    g++ -O2 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` rod_decoder_py.cpp -o rod_decoder_py`python3-config --extension-suffix

2. copy the rod_decoder_py.cpython-XXX-XXX-linux-gnu.so to  ../../scripts/

