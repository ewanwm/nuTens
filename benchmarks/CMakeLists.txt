
add_executable(benchmarks benchmarks.cpp)
target_link_libraries(benchmarks benchmark::benchmark tensor propagator benchmark::benchmark_main)
