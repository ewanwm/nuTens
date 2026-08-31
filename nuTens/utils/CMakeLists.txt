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

# if user wants to use pch then we use the pch people, especially if developing,
# might want to use this as including tensor related things can be
# excruciatingly sloow when building
if(NT_USE_PCH)
  message("Using precompiled header")

  file(
    GENERATE
    OUTPUT ${PROJECT_BINARY_DIR}/utils/nuTens-pch.cpp
    CONTENT "")

  add_library(nuTens-pch OBJECT ${PROJECT_BINARY_DIR}/utils/nuTens-pch.cpp)
  target_include_directories(nuTens-pch PUBLIC "${PROJECT_SOURCE_DIR}")

  set(PCH_LIBS "${PCH_LIBS};nt-logging;instrumentation")

  # the headers included in the PCH will (at some point) depend on which tensor
  # library is being used
  if(NT_USE_TORCH)
    target_compile_definitions(nuTens-pch PUBLIC USE_PYTORCH)
    set(PCH_LIBS "${PCH_LIBS};${TORCH_LIBRARIES}")
  endif()

  target_link_libraries(nuTens-pch PUBLIC "${PCH_LIBS}")
  target_precompile_headers(nuTens-pch PUBLIC nuTens-pch.hpp)
  set_target_properties(nuTens-pch PROPERTIES LINKER_LANGUAGE CXX)

endif() # end NT_USE_PCH block

# install headers to "include" directory
set(HEADERS "instrumentation.hpp")
install(FILES ${HEADERS} DESTINATION include/utils)

# install libraries to the "lib" directory
install(
  TARGETS instrumentation nt-logging spdlog
  DESTINATION lib
  EXPORT nuTens-targets)
