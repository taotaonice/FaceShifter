cmake_minimum_required(VERSION 2.8)
project(FaceShifter_Accelerations)

set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(pybind11 REQUIRED)
find_package(CUDA)
find_package(OpenCV)

include_directories(
  /usr/local/cuda-10.1/include/
  ${OpenCV_INCLUDE_DIRS}
  )
link_directories(
  /usr/local/cuda-10.1/lib64/
  )

cuda_add_library(culib SHARED cuda_kernels.cu)

pybind11_add_module(cuda_postprocess cuda_postprocess.cpp)
target_link_libraries(cuda_postprocess
  culib
  ${OpenCV_LIBS}
  )

