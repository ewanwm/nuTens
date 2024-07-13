cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)

OPTION(TEST_COVERAGE "TEST_COVERAGE" OFF)
IF(TEST_COVERAGE)
    message("Adding flags to check test coverage")
    SET(CMAKE_CXX_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
    SET(CMAKE_C_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
    SET(CMAKE_EXE_LINKER_FLAGS "-fprofile-arcs -ftest-coverage")
ELSE()
    message("Won't check test coverage")
ENDIF()

project(nuTens)
enable_testing()

find_package(Torch REQUIRED)
find_package(Protobuf REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(nuTens)
add_subdirectory(tests)