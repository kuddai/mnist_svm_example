cmake_minimum_required(VERSION 2.8)

project(mnist_svm) 

set(CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wall -g")

add_library(vlfeat 
    lib/lib_vlfeat/vl/host.c
    lib/lib_vlfeat/vl/random.c
    lib/lib_vlfeat/vl/svm.c
    lib/lib_vlfeat/vl/svmdataset.c
    lib/lib_vlfeat/vl/homkermap.c
    lib/lib_vlfeat/vl/mathop.c
    lib/lib_vlfeat/vl/generic.c)

add_subdirectory(code)
