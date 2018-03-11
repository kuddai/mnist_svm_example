include_directories(../lib_vlfeat/)
include_directories(.)

find_package(OpenCV REQUIRED)
find_package(Threads)


add_library(mnist_svm
    ./binary_svm.cpp)

target_link_libraries(mnist_svm
    ${OpenCV_LIBS}
    vlfeat_slic
    pthread)

add_executable(main
    ./main.cpp)

target_link_libraries(main
    mnist_svm)

add_executable(test_ml
    ./test/test_binary_svm.cpp
    ./catch_main.cpp)

target_link_libraries(test_ml
    mnist_svm)

