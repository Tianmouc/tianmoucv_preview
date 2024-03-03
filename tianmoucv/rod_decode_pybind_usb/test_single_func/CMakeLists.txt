cmake_minimum_required(VERSION 3.10)
project(test_single)

set(CMAKE_CXX_STANDARD 17)

# 查找系统的文件系统库
#find_package(Filesystem REQUIRED)

# 将源文件添加到可执行目标中
add_executable(test_single lyncam_compact_data.cpp)

# 链接文件系统库
#target_link_libraries(FileList PRIVATE std::filesystem)