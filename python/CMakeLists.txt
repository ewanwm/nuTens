
pybind11_add_module( 
    _pyNuTens MODULE 
    binding.cpp 
)

if(NT_USE_PCH)
    target_precompile_headers( _pyNuTens REUSE_FROM nuTens-pch )
endif()
target_link_libraries( _pyNuTens PUBLIC nuTens test-utils )

# This is passing in the version as a define just as an example
target_compile_definitions( _pyNuTens PRIVATE VERSION_INFO=${CMAKE_PROJECT_VERSION} )
target_compile_definitions( _pyNuTens PUBLIC USE_PYTORCH)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries( _pyNuTens PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

install( TARGETS _pyNuTens DESTINATION nuTens/ )
install( DIRECTORY nuTens DESTINATION . )