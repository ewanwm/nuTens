

add_library( constants STATIC constants.hpp)
set_target_properties(constants PROPERTIES LINKER_LANGUAGE CXX)

add_library(
    propagator STATIC 
    propagator.hpp propagator.cpp 
    const-density-solver.hpp const-density-solver.cpp
)

target_link_libraries(
    propagator PUBLIC 
    tensor 
    constants 
    utils
    )

if(NT_USE_PCH)
    target_precompile_headers(propagator REUSE_FROM nuTens-pch)
endif()

target_include_directories(propagator PUBLIC "${CMAKE_SOURCE_DIR}")
set_target_properties(propagator PROPERTIES LINKER_LANGUAGE CXX)
