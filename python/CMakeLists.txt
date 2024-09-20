
pybind11_add_module( 
    pyNuTens MODULE 
    pyNuTens.cpp 
)

if(NT_USE_PCH)
    target_precompile_headers(pyNuTens REUSE_FROM nuTens-pch)
endif()
target_link_libraries( pyNuTens PUBLIC nuTens )

install( TARGETS pyNuTens DESTINATION pyNuTens/)