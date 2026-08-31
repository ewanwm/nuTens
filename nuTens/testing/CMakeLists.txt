add_library(testing barger-propagator.hpp printing.hpp utils.hpp)

target_link_libraries(testing PUBLIC constants m)
set_target_properties(testing PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(
  testing PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
                $<INSTALL_INTERFACE:include>)

target_link_libraries(testing PUBLIC nt-logging instrumentation tensor-backend)

# install headers to "include" directory
set(HEADERS barger-propagator.hpp;nuFast.hpp;printing.hpp;utils.hpp)
install(FILES ${HEADERS} DESTINATION include/testing)

# install libraries to the "lib" directory
install(
  TARGETS testing
  DESTINATION lib
  EXPORT nuTens-targets)
