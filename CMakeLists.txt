cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)

project(nuTens)

OPTION(NT_TEST_COVERAGE "produce code coverage reports when running tests" OFF)

IF(NT_TEST_COVERAGE)
    message("Adding flags to check test coverage")
    add_compile_options("--coverage")
    add_link_options("--coverage")
ELSE()
    message("Won't check test coverage")
ENDIF()

enable_testing()

# add dependencies
include(cmake/CPM.cmake)

CPMAddPackage("gh:gabime/spdlog@1.8.2")

## check build times
## have this optional as it's not supported on all CMake platforms
OPTION(NT_BUILD_TIMING "output time to build each target" OFF)
IF(NT_BUILD_TIMING)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
ENDIF()

find_package(Torch REQUIRED)
find_package(Protobuf REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_subdirectory(nuTens)
add_subdirectory(tests)


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