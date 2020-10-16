cmake_minimum_required(VERSION 3.0.0)
project(QRKit VERSION 0.1.0)

set(Eigen3_DIR "/home/awf/dev/eigen/awf-build")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories("src")

add_subdirectory(test)
# add_subdirectory(src)
