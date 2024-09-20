
pybind11_add_module( 
    pyNuTens MODULE 
    binding.cpp 
)

if(NT_USE_PCH)
    target_precompile_headers( pyNuTens REUSE_FROM nuTens-pch )
endif()
target_link_libraries( pyNuTens PUBLIC nuTens )

# This is passing in the version as a define just as an example
target_compile_definitions( pyNuTens PRIVATE VERSION_INFO=${PROJECT_VERSION} )

install( TARGETS pyNuTens DESTINATION pyNuTens/ )