
add_executable(benchmarks benchmarks.cpp)
add_executable(benchmarks-gpu benchmarks-gpu.cpp)

if(NT_ENABLE_GPU_BENCHMARKING)
    target_compile_definitions(benchmarks PUBLIC BENCHMARK_GPU)
endif()

if(NT_USE_PCH)
    target_precompile_headers(benchmarks REUSE_FROM nuTens-pch)
endif()

target_link_libraries(benchmarks benchmark::benchmark benchmark::benchmark_main tensor propagator )

if (NT_USE_TORCH)
    target_compile_definitions(benchmarks PUBLIC USE_PYTORCH)
endif()
