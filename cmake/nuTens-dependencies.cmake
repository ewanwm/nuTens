## define dependencies of nutens which will be included using cpm where possible

## ==== Pytorch ====
find_package(Torch)
if( NOT Torch_FOUND )
    message( "didn't find pytorch, will try installing using cpm" )

    CPMAddPackage("gh:pytorch/pytorch@2.7.1")
    find_package(Torch REQUIRED)
  #set( Torch_FOUND TRUE)
endif()

message("Torch cxx flags: ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

## ==== spdlog ====
CPMAddPackage("gh:gabime/spdlog@1.8.2")

# ==== google benchmark ====
if(NT_ENABLE_BENCHMARKING)
    message("Enabling benchmarking")
    CPMAddPackage(
        GITHUB_REPOSITORY "google/benchmark"
        VERSION 1.8.5 
        OPTIONS "BENCHMARK_DOWNLOAD_DEPENDENCIES ON"
    )
else()
    message("Won't benchmark")
endif()

# ==== pybind11 ====
if(NT_ENABLE_PYTHON)
    message("Enabling python")
    CPMAddPackage(
        GITHUB_REPOSITORY "pybind/pybind11"
        VERSION 2.13.5 
    )

else()
    message("Won't enable python interface")
endif()
