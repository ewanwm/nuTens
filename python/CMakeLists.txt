
pybind11_add_module( 
    _binding MODULE 
    binding.cpp 
)

if(NT_USE_PCH)
    target_precompile_headers( _binding REUSE_FROM nuTens-pch )
endif()
target_link_libraries( _binding PUBLIC nuTens )

# This is passing in the version as a define just as an example
target_compile_definitions( _binding PRIVATE VERSION_INFO=${PROJECT_VERSION} )

install( TARGETS _binding DESTINATION pyNuTens/ )