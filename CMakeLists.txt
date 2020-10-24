cmake_minimum_required(VERSION 3.0.0)
project(QRKit VERSION 0.1.0)

include(user.cmake) # Create this from user.cmake.in
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++17 -O2 -fpermissive")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -fpermissive")

include_directories("src")

add_subdirectory(test)
# add_subdirectory(src)
