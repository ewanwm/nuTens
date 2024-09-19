
if(TORCH_FOUND)
    add_library(tensor STATIC tensor.hpp torch-tensor.cpp)
    target_compile_definitions(tensor PUBLIC USE_PYTORCH)
endif()

## when new tensor backends are added will add coresponding <library>-tensor.cpp and compile accordingly

if(NT_USE_PCH)
    target_precompile_headers(tensor REUSE_FROM nuTens-pch)
endif()

target_link_libraries(tensor PUBLIC utils)
set_target_properties(tensor PROPERTIES LINKER_LANGUAGE CXX)