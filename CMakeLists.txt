cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CUDA_STANDARD 14)
set( CMAKE_CUDA_COMPILER "nvcc" )

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# set the project name and version
project(nuTens VERSION 0.4.0)

# Changes default install path to be a subdirectory of the build dir.
# Can set build dir at configure time with -DCMAKE_INSTALL_PREFIX=/install/path
if(CMAKE_INSTALL_PREFIX STREQUAL "" OR CMAKE_INSTALL_PREFIX STREQUAL
  "/usr/local")
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}")
elseif(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}")
endif()

# user options
option(NT_USE_TORCH "Use torch as the backend for dealing with tensors" ON)
option(NT_TORCH_FROM_PIP "If it is not found, torch will be installed using pip" ON)
option(NT_TORCH_FROM_SCRATCH "If it is not found, torch will be compiled from scratch using CPM (very slow but maybe useful for debugging builds)" OFF)
option(NT_ALLOW_GLOBAL_PYTHON_ENV "Allow installing pip packages in global python environment" OFF)
option(NT_ENABLE_BENCHMARKING "enable benchmarking using google benchmark" OFF)
option(NT_ENABLE_PYTHON "enable python interface" OFF)
option(NT_COMPILE_TESTS "whether or not to compile unit and integration tests" ON)
option(NT_TEST_COVERAGE "produce code coverage reports when running tests" OFF)
option(NT_BUILD_TIMING "output time to build each target" OFF)
option(NT_USE_PCH "NT_USE_PCH" OFF)
option(BUILD_SHARED_LIBS "Build using shared libs" ON)

## to build the python library we require to build with the pic flag
if(NT_ENABLE_PYTHON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

    ## for now we also need to build with static libs
    message(WARNING "BUILD_SHARED_LIBS cannot be used when building python interface, setting BUILD_SHARED_LIBS=OFF")
    set(BUILD_SHARED_LIBS OFF)
endif()

# Need to add some special compile flags to check the code test coverage 
if(NT_TEST_COVERAGE)
    message("Adding flags to check test coverage")
    add_compile_options("--coverage")
    add_link_options("--coverage")
else()
    message("Won't check test coverage")
endif()

# enable ctest
if(NT_COMPILE_TESTS)
    message("Compiling tests")
    enable_testing()
else()
    message("Won't compile tests")
endif()

##########################
#### add dependencies ####
##########################

include(cmake/CPM.cmake)
include(cmake/nuTens-dependencies.cmake)


## check build times
## have this optional as it's not supported on all CMake platforms
if(NT_BUILD_TIMING)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
endif()


######################################
#### Go configure the actual code ####
######################################

add_subdirectory(nuTens)
add_subdirectory(tests)

if(NT_ENABLE_PYTHON)
    add_subdirectory(python)
endif()

if(NT_ENABLE_BENCHMARKING)
    add_subdirectory(benchmarks)
endif()

# Print out a handy message to more easily see the config options
message( STATUS "The following variables have been used to configure the build: " )
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    unset(MATCHED)
    string(REGEX MATCH "^NT_*" MATCHED ${_variableName})
    if (NOT MATCHED)
        continue()
    endif()
    
    message(STATUS "  ${_variableName}=${${_variableName}}")
endforeach()
message(STATUS "  BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}")

install(EXPORT nuTens-targets
  FILE nuTensTargets.cmake
  DESTINATION lib/cmake/nuTens
)

include(CMakePackageConfigHelpers)

# generate the config file that includes the exports
configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/templates/nuTensConfig.cmake.in
  "${PROJECT_BINARY_DIR}/nuTensConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/nuTens"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

write_basic_package_version_file(${PROJECT_BINARY_DIR}/nuTensConfigVersion.cmake
  VERSION ${nuTens_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES
  ${PROJECT_BINARY_DIR}/nuTensConfig.cmake
  ${PROJECT_BINARY_DIR}/nuTensConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/nuTens)
  
 export(EXPORT nuTens-targets
  FILE "${PROJECT_BINARY_DIR}/nuTensTargets.cmake"
)

include(cmake/nuTens-config.cmake)
