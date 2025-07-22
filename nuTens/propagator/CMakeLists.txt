

add_library( constants constants.hpp)
add_library( units units.hpp)
set_target_properties(constants PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(units PROPERTIES LINKER_LANGUAGE CXX)

add_library(
    propagator
    propagator.hpp propagator.cpp 
    const-density-solver.hpp const-density-solver.cpp
    base-mixing-matrix.hpp
    pmns-matrix.hpp
    DP-propagator.hpp DP-propagator.cpp
)

target_link_libraries(
    propagator PRIVATE
    tensor 
    constants 
    units
    nt-logging
    instrumentation
    tensor-backend
    )

if(NT_USE_PCH)
    target_precompile_headers(propagator REUSE_FROM nuTens-pch)
endif()

target_include_directories(propagator PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

set_target_properties(propagator PROPERTIES LINKER_LANGUAGE CXX)

## install headers to "include" directory
set(HEADERS "constants.hpp;units.hpp;propagator.hpp;const-density-solver.hpp;base-matter-solver.hpp")
install(FILES ${HEADERS} DESTINATION include/propagator)

## install libraries to the "lib" directory
install(
    TARGETS propagator constants units
    DESTINATION lib
    EXPORT nuTens-targets
)
