
add_subdirectory(utils)
add_subdirectory(tensors)
add_subdirectory(propagator)

add_library(nuTens INTERFACE)
target_link_libraries(nuTens INTERFACE tensor propagator)