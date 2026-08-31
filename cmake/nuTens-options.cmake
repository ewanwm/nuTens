# ##############################################################################
# Basic user options ####
# ##############################################################################

option(NT_USE_TORCH "Use torch as the backend for dealing with tensors" ON)

option(NT_TORCH_FROM_PIP
       "If it is not found, torch will be installed using pip" ON)

option(NT_ALLOW_GLOBAL_PYTHON_ENV
       "Allow installing pip packages in global python environment" OFF)

option(NT_ENABLE_PYTHON "enable python interface" OFF)

option(NT_COMPILE_TESTS "whether or not to compile unit and integration tests"
       ON)

# ##############################################################################
# advanced user options ####
# ##############################################################################

option(
  NT_TORCH_FROM_SCRATCH
  "If it is not found, torch will be compiled from scratch using CPM (very slow but maybe useful for debugging builds)"
  OFF)
mark_as_advanced(NT_TORCH_FROM_SCRATCH)

option(NT_ENABLE_BENCHMARKING "enable benchmarking using google benchmark" OFF)
mark_as_advanced(NT_ENABLE_BENCHMARKING)

option(NT_ENABLE_GPU_BENCHMARKING "enable benchmarking on GPU" OFF)
mark_as_advanced(NT_ENABLE_GPU_BENCHMARKING)

option(
  NT_COMPILE_GPU_TESTS
  "whether or not to compile unit and integration tests for GPU bound tensors"
  ON)
mark_as_advanced(NT_COMPILE_GPU_TESTS)

option(NT_TEST_COVERAGE "produce code coverage reports when running tests" OFF)
mark_as_advanced(NT_TEST_COVERAGE)

option(NT_BUILD_TIMING "output time to build each target" OFF)
mark_as_advanced(NT_BUILD_TIMING)

option(NT_USE_PCH "NT_USE_PCH" OFF)
mark_as_advanced(NT_USE_PCH)

option(NT_PROFILING "enable profiling of the code" OFF)
mark_as_advanced(NT_PROFILING)

option(BUILD_SHARED_LIBS "Build using shared libs" ON)
mark_as_advanced(BUILD_SHARED_LIBS)

# ##############################################################################
# Apply flags and other global vars implied by user options ####
# ##############################################################################

# to build the python library we require to build with the pic flag
if(NT_ENABLE_PYTHON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

  # for now we also need to build with static libs
  message(
    WARNING
      "BUILD_SHARED_LIBS cannot be used when building python interface, setting BUILD_SHARED_LIBS=OFF"
  )
  set(BUILD_SHARED_LIBS OFF)
  set(SPDLOG_BUILD_SHARED OFF)
endif()

# Need to add some special compile flags to check the code test coverage
if(NT_TEST_COVERAGE)
  message("Adding flags to check test coverage")
  add_compile_options("-O0")
  add_compile_options("--coverage")
  add_link_options("--coverage")
else()
  message("Won't check test coverage")
endif()

# enable ctest
if(NT_COMPILE_TESTS)
  message("Compiling tests")
  enable_testing()
else()
  message("Won't compile tests")
endif()

# have this optional as it's not supported on all CMake platforms
if(NT_BUILD_TIMING)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
endif()

# ##############################################################################
# Print out user options and values ####
# ##############################################################################

# Print out a handy message to more easily see the config options
message(
  STATUS "The following variables have been used to configure the build: ")
get_cmake_property(variable_names VARIABLES)
list(SORT variable_names)
foreach(variable_name ${variable_names})
  unset(MATCHED)
  string(REGEX MATCH "^NT_*" MATCHED ${variable_name})
  if(NOT MATCHED)
    continue()
  endif()

  message(STATUS "  ${variable_name}=${${variable_name}}")
endforeach()
message(STATUS "  BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}")

# ##############################################################################
# Now string valued variables ####
# ##############################################################################

# get the log level specified by the user via -DNT_LOG_LEVEL
set(NT_LOG_LEVEL
    "INFO"
    CACHE STRING "the level of detail to log to the console")

# Convert NT_LOG_LEVEL to all upper case so that we aren't case sensitive to
# user input
string(TOUPPER "${NT_LOG_LEVEL}" NT_LOG_LEVEL)

# Check the specified log level is valid
set(VALID_LOG_OPTIONS SILENT ERROR WARNING INFO DEBUG TRACE)
list(FIND VALID_LOG_OPTIONS ${NT_LOG_LEVEL} index)
if(${index} GREATER -1)
  message(STATUS "Setting log level to ${NT_LOG_LEVEL}")
else()
  message(
    FATAL_ERROR
      "Invalid log level specified: ${NT_LOG_LEVEL} \n Should be one of: ${VALID_LOG_OPTIONS}"
  )
endif()
