SET(nuTens_LIB_LIST "-lnuTens -libtensor -libpropagator -libinstrumentation -libnt-logging -libconstants -libunits")

SET(nuTens_FEATURES_LIST)

if(NT_ENABLE_PYTHON EQUAL 1)
  LIST(APPEND nuTens_FEATURES_LIST "python")
endif()
if(NT_USE_TORCH EQUAL 1)
  LIST(APPEND nuTens_FEATURES_LIST "torch")
endif()
if(NT_ENABLE_BENCHMARKING EQUAL 1)
  LIST(APPEND nuTens_FEATURES_LIST "benchmarks")
endif()
if(NT_COMPILE_TESTS EQUAL 1)
  LIST(APPEND nuTens_FEATURES_LIST "tests")
endif()
if(NT_TEST_COVERAGE EQUAL 1)
  LIST(APPEND nuTens_FEATURES_LIST "test-coverage")
endif()

# Set the creation date
string(TIMESTAMP CREATION_DATE "%d-%m-%Y")

string(REPLACE ";" " " nuTens_FEATURES "${nuTens_FEATURES_LIST}")
configure_file(${CMAKE_CURRENT_LIST_DIR}/templates/nuTens-config.in
  "${PROJECT_BINARY_DIR}/nuTens-config" @ONLY)
install(PROGRAMS
  "${PROJECT_BINARY_DIR}/nuTens-config" DESTINATION
  bin)
