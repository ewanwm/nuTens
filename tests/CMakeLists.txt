
add_executable(basic basic.cpp)
target_link_libraries(basic PUBLIC tensor)
target_include_directories(basic PUBLIC "${CMAKE_SOURCE_DIR}")

add_test(NAME basic-test COMMAND basic)


add_executable(two-flavour-osc two-flavour-osc.cpp)
target_link_libraries(two-flavour-osc PUBLIC propagator)

add_test(NAME two-flavour-osc-test COMMAND two-flavour-osc)


add_executable(two-flavour-const-matter two-flavour-const-matter.cpp)
target_link_libraries(two-flavour-const-matter PUBLIC propagator)

add_test(NAME two-flavour-const-matter-test COMMAND two-flavour-const-matter)