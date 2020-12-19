cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(box)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 14)
find_library(darknet libdarknet.so /home/ethan/Documents/robocon/code/darknet/)
# 注意：如果想要在自己的机器上跑代码记得将路径改为自己电脑上darknet安装的位置
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/home/ethan/Documents/robocon/code/darknet/include/)
include_directories(.)
add_executable(box test.cpp box.h box.cpp)
target_compile_options(${PROJECT_NAME} PRIVATE -fPIC)
# -fPIC是动态链接库常用的编译参数，可以使动态链接库不需要根据当前程序加载到的位置重定位
# 不加-fPIC就意味着每一次动态链接库都需要重定位到当前程序所在内存位置，当代码与位置有关的时候我们这样做
# 当库不大的时候或者库需要很高的效率的时候，我们就使用-fPIC取消重定位操作，使得.so在内存中真正共享，而不用每使用一次都copy一份以重定位
target_link_libraries(box
        ${OpenCV_LIBS}
        ${darknet}
        )