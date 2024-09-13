
pybind11_add_module( 
    pyNuTens MODULE 
    pyNuTens.cpp 
)

target_link_libraries( pyNuTens PUBLIC tensor propagator )

install( TARGETS pyNuTens DESTINATION pyNuTens/)