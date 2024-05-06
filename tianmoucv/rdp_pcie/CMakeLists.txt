cmake_minimum_required(VERSION 2.8.12)
project(rod_decoder_py)

find_package(pybind11 REQUIRED)
pybind11_add_module(rod_decoder_py rod_decoder_py.cpp)