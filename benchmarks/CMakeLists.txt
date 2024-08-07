
add_executable(benchmarks benchmarks.cpp)
target_link_libraries(benchmarks benchmark::benchmark benchmark::benchmark_main tensor propagator )
