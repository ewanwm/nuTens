
add_library(test-utils test-utils.hpp barger-propagator.hpp)
target_link_libraries(test-utils PUBLIC constants)
set_target_properties(test-utils PROPERTIES LINKER_LANGUAGE CXX)

add_executable(tensor-basic tensor-basic.cpp)
target_link_libraries(tensor-basic PUBLIC tensor test-utils)
target_include_directories(tensor-basic PUBLIC "${CMAKE_SOURCE_DIR}")

add_test(NAME tensor-basic-test COMMAND tensor-basic)


add_executable(barger barger-test.cpp)
target_link_libraries(barger PUBLIC test-utils)
target_include_directories(barger PUBLIC "${CMAKE_SOURCE_DIR}")

add_test(NAME barger-test COMMAND basic)


add_executable(two-flavour-osc two-flavour-osc.cpp)
target_link_libraries(two-flavour-osc PUBLIC propagator test-utils)

add_test(NAME two-flavour-osc-test COMMAND two-flavour-osc)


add_executable(two-flavour-const-matter two-flavour-const-matter.cpp)
target_link_libraries(two-flavour-const-matter PUBLIC propagator test-utils)

add_test(NAME two-flavour-const-matter-test COMMAND two-flavour-const-matter)