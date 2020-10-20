cmake_minimum_required (VERSION 3.0)

add_executable (test-qrkit test-qrkit.cpp)
add_executable (test-utils test-utils.cpp)

target_link_libraries (test-utils Eigen3::Eigen)
target_link_libraries (test-qrkit Eigen3::Eigen)
