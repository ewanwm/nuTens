
add_executable(basic basic.cpp)
target_link_libraries(basic PUBLIC tensor)
target_include_directories(basic PUBLIC "${CMAKE_SOURCE_DIR}")

add_test(NAME basicTest COMMAND basic)
