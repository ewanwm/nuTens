# ##############################################################################
# set up logging ####
# ##############################################################################
add_library(nt-logging STATIC logging.hpp)
target_link_libraries(nt-logging PRIVATE spdlog)
target_include_directories(
  nt-logging PUBLIC $<BUILD_INTERFACE:${spdlog_SOURCE_DIR}/include>
                    $<INSTALL_INTERFACE:include>)

set_target_properties(nt-logging PROPERTIES LINKER_LANGUAGE CXX)

# set the log level that will be used inside the logging.hpp file
target_compile_definitions(nt-logging
                           PUBLIC NT_LOG_LEVEL=NT_LOG_LEVEL_${NT_LOG_LEVEL})

# ##############################################################################
# set up instrumentation ####
# ##############################################################################
add_library(instrumentation instrumentation.hpp)
set_target_properties(instrumentation PROPERTIES LINKER_LANGUAGE CXX)

if(NT_PROFILING)
  target_compile_definitions(instrumentation PUBLIC USE_PROFILING)
endif()

# install headers to "include" directory
set(HEADERS "instrumentation.hpp")
install(FILES ${HEADERS} DESTINATION include/utils)

# install libraries to the "lib" directory
install(
  TARGETS instrumentation nt-logging spdlog
  DESTINATION lib
  EXPORT nuTens-targets)
