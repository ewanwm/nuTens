enable_testing()

add_library(
    test-utils 
    barger-propagator.hpp
    printing.hpp
    utils.hpp
)

target_link_libraries(test-utils PUBLIC constants m)
set_target_properties(test-utils PROPERTIES LINKER_LANGUAGE CXX)

include(GoogleTest)

foreach(TESTNAME 
    test-barger 
    test-tensor
    test-matter-solver
    test-two-flavour-osc
    test-three-flavour-osc
    test-pmns-matrix
    test-propagator
    test-DP-propagator
)

    add_executable("${TESTNAME}" "${TESTNAME}.cpp" "${TESTNAME}.hpp")
    
    IF(NT_USE_PCH)
        target_precompile_headers("${TESTNAME}" REUSE_FROM nuTens-pch)
    ENDIF()

    target_link_libraries(
        "${TESTNAME}" PUBLIC 
        nuTens 
        test-utils 
        m 
        GTest::gtest_main
    )

    if(NT_COMPILE_GPU_TESTS)
        target_compile_definitions("${TESTNAME}" PUBLIC COMPILE_GPU_TESTS)
    endif()
    
    target_include_directories("${TESTNAME}" PUBLIC "${PROJECT_SOURCE_DIR}")
    
    gtest_discover_tests("${TESTNAME}")

endforeach()

install( DIRECTORY python DESTINATION ./tests )
