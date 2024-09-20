
add_library(test-utils test-utils.hpp barger-propagator.hpp)
target_link_libraries(test-utils PUBLIC constants m)
set_target_properties(test-utils PROPERTIES LINKER_LANGUAGE CXX)

foreach(TESTNAME 
    barger tensor-basic two-flavour-vacuum two-flavour-const-matter
    )

    add_executable("${TESTNAME}" "${TESTNAME}.cpp")
    
    IF(NT_USE_PCH)
        target_precompile_headers("${TESTNAME}" REUSE_FROM nuTens-pch)
    ENDIF()

    target_link_libraries("${TESTNAME}" PUBLIC nuTens test-utils m)
    target_include_directories("${TESTNAME}" PUBLIC "${CMAKE_SOURCE_DIR}")
    add_test(NAME "${TESTNAME}-test" COMMAND "${TESTNAME}")

endforeach()
