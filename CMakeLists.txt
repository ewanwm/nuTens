cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(nuTens)

find_package(Torch REQUIRED)
find_package(Protobuf REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(nuTens)
add_subdirectory(tests)