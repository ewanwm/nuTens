## define dependencies of nutens which will be included using cpm where possible

## ==== Pytorch ====
if(NT_USE_TORCH)
    find_package(Torch)

    if(Torch_FOUND)
        message(STATUS "Found pre-installed Torch at ${Torch_DIR}")
    
    elseif(NT_TORCH_FROM_PIP)
        
        message(STATUS "Torch not found - installing with pip")
        
        ## check if we are in a virtual environment before pip installing
        if(DEFINED ENV{VIRTUAL_ENV} OR DEFINED ENV{CONDA_PREFIX})
            set(_pip_args)
 
        else()
            ## we're not in a virtual env, make sure the user is really sure that's ok
            if(NT_ALLOW_GLOBAL_PYTHON_ENV)
                set(_pip_args "--user")
            else()
                message(FATAL_ERROR "\
You're not in a python virtual environment but trying to install pytorch! \n \
This is ill advised, but if you're sure that's what you want to do you can set the option \n \
  -D NT_ALLOW_GLOBAL_PYTHON_ENV")

            endif() ## end of if(NT_ALLOW_GLOBAL_PYTHON_ENV)

        endif() ## done checking for virtual env 
        
        find_package(Python COMPONENTS Interpreter REQUIRED)
        
        ## install torch Python package using pip
        execute_process(COMMAND ${Python_EXECUTABLE} -m pip install -r ${PROJECT_SOURCE_DIR}/PyTorch_requirements.txt)
        
        ## need to do some absolute tomfoolery to get the path to the torch shared library
        execute_process(
            COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/cmake/torch-cmake-prefix.py
            OUTPUT_VARIABLE _TORCH_CMAKE_PREFIX
        )
        string(REPLACE ' "" TORCH_CMAKE_PREFIX ${_TORCH_CMAKE_PREFIX})

        message(STATUS "TORCH CMAKE PREFIX: ${TORCH_CMAKE_PREFIX}")
        list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
    
    elseif(NT_TORCH_FROM_SCRATCH)

        CPMAddPackage(
            NAME Torch
            GIT_REPOSITORY git@github.com:pytorch/pytorch.git
            VERSION 2.7.1
        )

    else()
        message(FATAL_ERROR "\
Torch not found and no method specified to install it. \n \
either go install torch and try again or set one of\n \
    -DNT_TORCH_FROM_PIP=ON\n \
    -DNT_TORCH_FROM_SCRATCH=ON 
")

    endif() ## end of if(Torch_FOUND)

endif() ## end of if(NT_USE_TORCH)

message("CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
execute_process(COMMAND ls ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

message(STATUS "Torch DIR ${TORCH_DIR}")
message(STATUS "Torch libs ${TORCH_LIBS}")
message("Torch cxx flags: ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

## ==== spdlog ====
CPMFindPackage(
    NAME spdlog
    GITHUB_REPOSITORY gabime/spdlog
    VERSION 1.8.2
)

# ==== google tests ====
if(NT_COMPILE_TESTS)
    CPMAddPackage(
        GITHUB_REPOSITORY "google/googletest"
        VERSION 1.17.0
        OPTIONS 
        "BUILD_GMOCK OFF"
    )
endif()

# ==== google benchmark ====
if(NT_ENABLE_BENCHMARKING)
    message("Enabling benchmarking")
    CPMAddPackage(
        GITHUB_REPOSITORY "google/benchmark"
        VERSION 1.8.5 
        OPTIONS 
        "BENCHMARK_ENABLE_TESTING OFF"
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
