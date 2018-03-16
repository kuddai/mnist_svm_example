include_directories(../lib/lib_vlfeat/)
include_directories(../lib/catch2/)
include_directories(./ml/)

find_package(OpenCV REQUIRED)
find_package(Threads)


add_library(mnist_svm
    ./ml/binary_svm.cpp
    ./ml/multiclass_svm.cpp
    ./ml/util.cpp)

target_link_libraries(mnist_svm
    ${OpenCV_LIBS}
    vlfeat
    pthread)

add_executable(main
    ./main.cpp)

target_link_libraries(main
    mnist_svm)

add_executable(test_ml
    ./test/test_binary_svm.cpp
    ./test/test_multiclass_svm.cpp
    ../lib/catch2/catch_main.cpp)

target_link_libraries(test_ml
    mnist_svm)

