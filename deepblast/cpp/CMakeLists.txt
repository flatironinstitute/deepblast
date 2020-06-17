# include_directories(.)
#
# set(INNER_LIBRARIES
#     nw.cpp)

# Barebones cmake file
cmake_minimum_required(VERSION 3.0)
project(nw)

find_package(Torch REQUIRED)
set(CMAKE_CXX_STANDARD 14)

# Useful tutorials for inspiration
# https://github.com/pytorch/examples/tree/master/cpp/autograd
# https://www.eriksmistad.no/getting-started-with-google-test-on-ubuntu/


# Locate Gtest
# find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIRS})


set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")


# Build 1 only use executable
add_executable(nw nw.cpp)
# add_executable(test_nw nw.cpp test_nw.cpp)
target_link_libraries(nw ${TORCH_LIBRARIES})
set_property(TARGET nw PROPERTY CXX_STANDARD 14)

# Build 2 attempt to use Gtest
# add_library(nw nw.cpp)
# add_executable(test_nw nw.cpp test_nw.cpp)
# target_link_libraries(nw ${TORCH_LIBRARIES})
# target_link_libraries(test_nw ${TORCH_LIBRARIES} ${GTEST_LIBRARIES} pthread)
# set_property(TARGET nw PROPERTY CXX_STANDARD 14)
# set_property(TARGET test_nw PROPERTY CXX_STANDARD 14)
