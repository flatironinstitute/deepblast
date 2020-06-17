# include_directories(.)
#
# set(INNER_LIBRARIES
#     nw.cpp)

# Barebones cmake file
cmake_minimum_required(VERSION 3.0)
project(nw)

find_package(Torch REQUIRED)
set(CMAKE_CXX_STANDARD 14)

# https://github.com/pytorch/examples/tree/master/cpp/autograd


add_executable(nw nw.cpp)
target_link_libraries(nw "${TORCH_LIBRARIES}")
set_property(TARGET nw PROPERTY CXX_STANDARD 14)
