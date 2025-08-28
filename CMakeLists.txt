cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CUDA_STANDARD 14)
set( CMAKE_CUDA_COMPILER "nvcc" )

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# function to get the project version from the most recent git tag
function(get_version_from_git)
    find_package(Git QUIET)
    if(NOT Git_FOUND)
        message(WARNING "Git not found")
        return()
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --always
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE GIT_RESULT
    )

    if(NOT GIT_RESULT EQUAL 0)
        message(WARNING "Failed to get git tag")
        return()
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short=7 HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_SHORT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    string(REGEX REPLACE "^v" "" CLEAN_TAG "${GIT_TAG}")
    if(CLEAN_TAG MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(-.*)?$")

        set(PROJECT_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(PROJECT_VERSION_MAJOR ${CMAKE_MATCH_1} PARENT_SCOPE)
        set(PROJECT_VERSION_MINOR ${CMAKE_MATCH_2})
        set(PROJECT_VERSION_MINOR ${CMAKE_MATCH_2} PARENT_SCOPE)
        set(PROJECT_VERSION_PATCH ${CMAKE_MATCH_3})
        set(PROJECT_VERSION_PATCH ${CMAKE_MATCH_3} PARENT_SCOPE)

        set(FULL_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}+${GIT_COMMIT_SHORT_HASH}")
        set(FULL_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}+${GIT_COMMIT_SHORT_HASH}" PARENT_SCOPE)
        set(PROJECT_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}") 
        set(PROJECT_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}" PARENT_SCOPE)
    else()
        message(WARNING "Tag '${CLEAN_TAG}' does not match semver format")
    endif()
endfunction()

# set the project name and version
get_version_from_git()
message(INFO "nuTens PROJECT_VERSION: ${PROJECT_VERSION}")
project(nuTens VERSION ${PROJECT_VERSION})

# configure the version.h file with all above version info
configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/templates/version.h.in
    ${PROJECT_BINARY_DIR}/include/version.h
)


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

if(NT_COMPILE_TESTS)
    add_subdirectory(tests)
endif()

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
