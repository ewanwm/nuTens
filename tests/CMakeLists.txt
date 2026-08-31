if(NT_COMPILE_TESTS)

  include(GoogleTest)

  foreach(
    test_name
    test-barger
    test-tensor
    test-matter-solver
    test-two-flavour-osc
    test-three-flavour-osc
    test-pmns-matrix
    test-propagator
    test-DP-propagator)

    add_executable("${test_name}" "${test_name}.cpp" "${test_name}.hpp")

    if(NT_USE_PCH)
      target_precompile_headers("${test_name}" REUSE_FROM nuTens-pch)
    endif()

    target_link_libraries("${test_name}" PRIVATE nuTens::All m
                                                 GTest::gtest_main)

    if(NT_COMPILE_GPU_TESTS)
      target_compile_definitions("${test_name}" PUBLIC COMPILE_GPU_TESTS)
    endif()

    gtest_discover_tests("${test_name}")

  endforeach()

  if(NT_ENABLE_PYTHON)
    install(DIRECTORY python DESTINATION ./tests)
  endif()

endif()
