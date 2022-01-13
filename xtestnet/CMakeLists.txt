CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

project(xtestnet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

# create executable
#file(GLOB SOURCE_FILES "src/*.h" "src/*.cpp")
#file(GLOB SOURCE_FILES "*.h" "*.cpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# plugin
cuda_add_library(myplugin SHARED ${PROJECT_SOURCE_DIR}/src/mPlugin.cpp)
target_link_libraries(myplugin nvinfer cudart)

cuda_add_executable(xtestnet ${PROJECT_SOURCE_DIR}/src/utils.cpp ${PROJECT_SOURCE_DIR}/src/layers.cpp ${PROJECT_SOURCE_DIR}/src/xtestnet.cpp main.cpp)

target_link_libraries(xtestnet nvinfer)
target_link_libraries(xtestnet cudart)
#target_link_libraries(xtestnet nvinfer_plugin)
target_link_libraries(xtestnet myplugin)

add_definitions(-O2 -pthread)