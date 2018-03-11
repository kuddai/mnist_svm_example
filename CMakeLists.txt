cmake_minimum_required(VERSION 2.8)

project(vlfeat_slic_example) 

set(CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -g")

# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_library(vlfeat_slic lib_vlfeat/vl/host.c
lib_vlfeat/vl/random.c
lib_vlfeat/vl/svm.c
lib_vlfeat/vl/svmdataset.c
lib_vlfeat/vl/homkermap.c
lib_vlfeat/vl/mathop.c
lib_vlfeat/vl/generic.c
lib_vlfeat/vl/slic.c)
add_subdirectory(vlfeat_slic_cli)
