cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)

project(nuTens)

# Need to add some special compile flags to check the code test coverage 
OPTION(NT_TEST_COVERAGE "produce code coverage reports when running tests" OFF)
IF(NT_TEST_COVERAGE)
    message("Adding flags to check test coverage")
    add_compile_options("--coverage")
    add_link_options("--coverage")
ELSE()
    message("Won't check test coverage")
ENDIF()

enable_testing()

##########################
#### add dependencies ####
##########################

find_package(Torch REQUIRED)
find_package(Protobuf REQUIRED)
message("Torch cxx flags: ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wabi-tag ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(cmake/CPM.cmake)

CPMAddPackage("gh:gabime/spdlog@1.8.2")

# If user wants to enable benchmarking we need to set up google benchmark dependency
OPTION(NT_ENABLE_BENCHMARKING "enable benchmarking using google benchmark" OFF)
IF(NT_ENABLE_BENCHMARKING)
    message("Enabling benchmarking")
    CPMAddPackage(
        GITHUB_REPOSITORY "google/benchmark"
        VERSION 1.8.5 
        OPTIONS "BENCHMARK_DOWNLOAD_DEPENDENCIES ON"
    )
ELSE()
    message("Won't benchmark")
ENDIF()


## check build times
## have this optional as it's not supported on all CMake platforms
OPTION(NT_BUILD_TIMING "output time to build each target" OFF)
IF(NT_BUILD_TIMING)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
ENDIF()


######################################
#### Go configure the actual code ####
######################################

add_subdirectory(nuTens)
add_subdirectory(tests)

IF(NT_ENABLE_BENCHMARKING)
    add_subdirectory(benchmarks)
ENDIF()


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
