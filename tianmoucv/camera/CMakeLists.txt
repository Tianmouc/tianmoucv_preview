cmake_minimum_required(VERSION 3.0.0)
project("tianmoucv")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")

    # opencv
    set(OpenCV_DIR "J:/projects/tianmoucv_windows_dependency/opencv/")
    set(CMAKE_PREFIX_PATH "J:/projects/tianmoucv_windows_dependency/opencv/build/x64/vc15/lib/")

    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_DIR})
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV Libraries: " ${OpenCV_LIBS})
    message(${OpenCV_INCLUDE_DIRS})
    link_libraries(${CMAKE_PREFIX_PATH})

    # libusbCMAKE_PREFIX_PATH
    include_directories("J:/projects/tianmoucv_windows_dependency/")
    link_libraries("J:/projects/tianmoucv_windows_dependency/libusb-1.0")  # Update the path to your OpenCV installation

    # cypress lib
    include_directories("J:/projects/tianmoucv_windows_dependency/cypress/inc")  
    link_libraries("J:/projects/tianmoucv_windows_dependency/cypress/lib/x64")  

    set(inc1 ${SDK_PATH}/usb/inc)
    set(inc2 ${SDK_PATH}/usb/cyusb/lib)
    set(inc3 ${SDK_PATH}/usb/api)
    set(inc4 ${SDK_PATH}/inc)
    set(linkpath ${SDK_PATH}/usb/api)

    message(${inc1})
    message(${inc2})
    message(${inc3})

    link_directories(${linkpath})
    include_directories(${inc1})
    include_directories(${inc2})
    include_directories(${inc3})
    include_directories(${inc4})

    add_compile_definitions(USB_SDK)

    set(file1 tianmoucv.cpp)
    set(file2 ${SDK_PATH}/src/tmc_proc.cpp)
    set(file3 ${SDK_PATH}/usb/src/new_dataRecevier.cpp)
    set(file4 ${SDK_PATH}/usb/src/camera.cpp)
    set(file5 ${SDK_PATH}/usb/src/lyn_cam.cpp)
    set(file6 ${SDK_PATH}/usb/src/tools.cpp)
    set(file7 ${SDK_PATH}/src/isp.cpp)

    message(${file1})
    message(${file7})

    add_library(tianmoucv SHARED ${file1} ${file2} ${file3} ${file4} ${file5} ${file6} ${file7})
    target_link_libraries(tianmoucv PRIVATE pthread )  # Adjust library names accordingly
    target_link_libraries(tianmoucv PRIVATE ${OpenCV_LIBS} )  # Adjust library names accordingly


else()
    set(inc1 ${SDK_PATH}/usb/inc)
    set(inc2 ${SDK_PATH}/usb/cyusb/lib)
    set(inc3 ${SDK_PATH}/usb/api)
    set(inc4 ${SDK_PATH}/inc)
    set(linkpath ${SDK_PATH}/usb/api)

    message(${inc1})
    message(${inc2})
    message(${inc3})

    link_directories(${linkpath})
    
    include_directories(${inc1})
    include_directories(${inc2})
    include_directories(${inc3})
    include_directories(${inc4})
    add_compile_definitions(USB_SDK)

    set(file1 tianmoucv.cpp)
    set(file2 ${SDK_PATH}/src/tmc_proc.cpp)
    set(file3 ${SDK_PATH}/usb/src/new_dataRecevier.cpp)
    set(file4 ${SDK_PATH}/usb/src/camera.cpp)
    set(file5 ${SDK_PATH}/usb/src/lyn_cam.cpp)
    set(file6 ${SDK_PATH}/usb/src/tools.cpp)
    set(file7 ${SDK_PATH}/src/isp.cpp)
    

    include_directories("/usr/local/include/opencv4")
    find_package(OpenCV 4 REQUIRED)
    if(OpenCV_FOUND)
        message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
    else()
        message(FATAL_ERROR "OpenCV version 4.5.0 or higher is required.")
    endif()

    add_executable(tianmoucv_app ${file1} ${file2} ${file3} ${file4} ${file5} ${file6} ${file7})
    target_link_libraries(tianmoucv_app PRIVATE ${OpenCV_LIBS} pthread cyusb -lusb-1.0)

    add_library(tianmouc SHARED ${file1} ${file2} ${file3} ${file4} ${file5} ${file6} ${file7})
    target_link_libraries(tianmouc PRIVATE ${OpenCV_LIBS} cyusb pthread -lusb-1.0)


endif()

