
if(NT_USE_TORCH)
    add_library(tensor tensor.hpp torch-tensor.cpp dtypes.hpp)
    target_compile_definitions(tensor PUBLIC USE_PYTORCH)
endif()

target_include_directories(tensor
                           PUBLIC
                            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
                            $<INSTALL_INTERFACE:include>
                           )

## when new tensor backends are added will add coresponding <library>-tensor.cpp and compile accordingly

if(NT_USE_PCH)
    target_precompile_headers(tensor REUSE_FROM nuTens-pch)
endif()

target_link_libraries(
    tensor PUBLIC
    nt-logging
    instrumentation
    tensor-backend)

set_target_properties(tensor PROPERTIES LINKER_LANGUAGE CXX)

## install headers to "include" directory
set(HEADERS "tensor.hpp")
install(FILES ${HEADERS} DESTINATION include/tensor)

## install libraries to the "lib" directory
install(
    TARGETS tensor 
    DESTINATION lib
    EXPORT nuTens-targets
)
