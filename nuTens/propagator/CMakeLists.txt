

add_library( constants constants.hpp)
add_library( units units.hpp)
set_target_properties(constants PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(units PROPERTIES LINKER_LANGUAGE CXX)

function( add_resource input target)
    string( MAKE_C_IDENTIFIER ${input} input_identifier )
    set( output "${input_identifier}.o" )
    set( ${input}_object ${output} PARENT_SCOPE)

    add_custom_command(
        PRE_BUILD
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${output}
        COMMAND ${CMAKE_LINKER} --relocatable --format binary --output ${output} ${input}
        DEPENDS ${input}
    )

    add_library( ${target} OBJECT IMPORTED )
    
    set_target_properties( ${target} PROPERTIES
        IMPORTED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/${output}"
    )

endfunction()

set(
    FILES 
    propagator.hpp propagator.cpp 
    const-density-solver.hpp const-density-solver.cpp
    base-mixing-matrix.hpp
    pmns-matrix.hpp
    DP-propagator.hpp DP-propagator.cpp
)

if(NT_USE_TORCH)
    add_custom_command(
        PRE_BUILD
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/precompiledDPpropagator-cpu.pt2
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/precompiled-DP-propagator.py ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS precompiled-DP-propagator.py
    )
    list(APPEND FILES precompiled-DP-propagator.hpp)
endif()

add_library(
    propagator
    ${FILES}
)

add_resource(precompiledDPpropagator-cpu.pt2 precompDPpropCPU)
message(${precompiledDPpropagator-cpu.pt2_object})
target_link_libraries(propagator PRIVATE precompDPpropCPU)

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
