

add_library(propagator STATIC vacuum-propagator.hpp vacuum-propagator.cpp)
target_link_libraries(propagator PUBLIC tensor)

target_include_directories(propagator PUBLIC "${CMAKE_SOURCE_DIR}")
set_target_properties(propagator PROPERTIES LINKER_LANGUAGE CXX)
