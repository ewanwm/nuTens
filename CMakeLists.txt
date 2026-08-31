cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# utilities for getting version from git
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/nuTens-versioning.cmake)

# set the project name and version
get_version_from_git()
message(INFO "nuTens PROJECT_VERSION: ${PROJECT_VERSION}")
project(nuTens VERSION ${PROJECT_VERSION})

# compiler options
set(CMAKE_CXX_STANDARD 17)
set(CUDA_STANDARD 14)
set(CMAKE_CUDA_COMPILER "nvcc")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# configure the version.h file with all above version info
configure_file(${PROJECT_SOURCE_DIR}/cmake/templates/version.h.in
               ${PROJECT_BINARY_DIR}/include/version.h)

# Changes default install path to be a subdirectory of the build dir. Can set
# build dir at configure time with -DCMAKE_INSTALL_PREFIX=/install/path
if((NOT DEFINED CMAKE_INSTALL_PREFIX)
   OR (CMAKE_INSTALL_PREFIX STREQUAL "")
   OR (CMAKE_INSTALL_PREFIX STREQUAL "/usr/local"))
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}")
endif()

# set up user provided command line options
include(${PROJECT_SOURCE_DIR}/cmake/nuTens-options.cmake)

# add dependencies
include(cmake/CPM.cmake)
include(cmake/nuTens-dependencies.cmake)

# ##############################################################################
# Go configure the actual code ####
# ##############################################################################

add_subdirectory(nuTens)
add_subdirectory(tests)

if(NT_ENABLE_PYTHON)
  add_subdirectory(python)
endif()

if(NT_ENABLE_BENCHMARKING)
  add_subdirectory(benchmarks)
endif()

install(
  EXPORT nuTens-targets
  FILE nuTensTargets.cmake
  DESTINATION lib/cmake/nuTens)

include(CMakePackageConfigHelpers)

# generate the config file that includes the exports
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/templates/nuTensConfig.cmake.in
  "${PROJECT_BINARY_DIR}/nuTensConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/nuTens"
  NO_SET_AND_CHECK_MACRO NO_CHECK_REQUIRED_COMPONENTS_MACRO)

write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/nuTensConfigVersion.cmake
  VERSION ${nuTens_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES ${PROJECT_BINARY_DIR}/nuTensConfig.cmake
              ${PROJECT_BINARY_DIR}/nuTensConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/nuTens)

export(EXPORT nuTens-targets FILE "${PROJECT_BINARY_DIR}/nuTensTargets.cmake")

include(cmake/nuTens-config.cmake)
