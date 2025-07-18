
add_subdirectory(utils)
add_subdirectory(tensors)
add_subdirectory(propagator)

add_library(nuTens INTERFACE)
target_link_libraries(nuTens INTERFACE tensor propagator)

target_include_directories(nuTens
                           INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                            $<INSTALL_INTERFACE:include>
                           )

install(
  TARGETS nuTens
  DESTINATION lib
  EXPORT nuTens-targets
)