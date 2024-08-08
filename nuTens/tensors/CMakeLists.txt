

IF(TORCH_FOUND)
    add_library(tensor STATIC tensor.hpp torch-tensor.cpp)
    target_compile_definitions(tensor PUBLIC USE_PYTORCH)
ENDIF()

IF(NT_USE_PCH)
    target_link_libraries(tensor PUBLIC nuTens-pch)

ELSE()
    IF(TORCH_FOUND)

        target_link_libraries(tensor PUBLIC "${TORCH_LIBRARIES}")

    ELSE()
        message( FATAL_ERROR "No library found to deal with tensors. Currently only pytorch is available, please install this and try again." )
    ENDIF()
ENDIF()

target_link_libraries(tensor PUBLIC instrumentation)
target_include_directories(tensor PUBLIC "${CMAKE_SOURCE_DIR}")
set_target_properties(tensor PROPERTIES LINKER_LANGUAGE CXX)