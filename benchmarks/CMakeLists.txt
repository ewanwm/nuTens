
add_executable(benchmarks benchmarks.cpp)

if(NT_USE_PCH)
    target_precompile_headers(benchmarks REUSE_FROM nuTens-pch)
endif()

target_link_libraries(benchmarks benchmark::benchmark benchmark::benchmark_main tensor propagator )
