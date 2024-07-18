cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
project(nuTens)
enable_testing()

## check build times
## have this optional as it's not supported on all CMake platforms
OPTION(BUILD_TIMING "output time to build each target" OFF)
IF(BUILD_TIMING)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
ENDIF()

find_package(Torch REQUIRED)
find_package(Protobuf REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


## if user wants to use pch then we use the pch
## people, especially if developing, might want to use this as including tensor related things
## can be excruciatingly sloow when building
OPTION(USE_PCH "USE_PCH" OFF)
IF(USE_PCH)
    message("Using precompiled header")
    
    add_library(nuTens-pch nuTens-pch.hpp)

    ## the headers included in the PCH will (at some point) depend on which tensor library is being used
    IF(TORCH_FOUND)
        message( "Including PyTorch includes in the PCH")
        message( "TORCH_LIBRARIES = ")
        message( "${TORCH_LIBRARIES}")

        target_compile_definitions(nuTens-pch PUBLIC USE_PYTORCH)
        SET(PCH_LIBS "${TORCH_LIBRARIES}")
    ENDIF()

    target_link_libraries(nuTens-pch PUBLIC "${PCH_LIBS}")
    target_precompile_headers(nuTens-pch PUBLIC nuTens-pch.hpp)
    set_target_properties(nuTens-pch PROPERTIES LINKER_LANGUAGE CXX)

ENDIF() ## end USE_PCH block

add_subdirectory(nuTens)
add_subdirectory(tests)