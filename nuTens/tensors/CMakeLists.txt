
IF(TORCH_FOUND)
    message( "Found tensor library: PyTorch")
    
    add_library(tensor STATIC tensor.hpp torch-tensor.cpp)
    target_compile_definitions(tensor PUBLIC USE_PYTORCH)
    target_link_libraries(tensor PUBLIC "${TORCH_LIBRARIES}")

ELSE()
    message( FATAL_ERROR "No library found to deal with tensors. Currently only pytorch is available, please install this and try again." )
ENDIF()


target_include_directories(tensor PUBLIC "${CMAKE_SOURCE_DIR}")
set_target_properties(tensor PROPERTIES LINKER_LANGUAGE CXX)