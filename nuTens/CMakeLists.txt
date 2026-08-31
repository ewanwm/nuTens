add_subdirectory(utils)

# if user wants to use pch then we use the pch. People, especially if developing,
# might want to use this as including tensor related things can be
# excruciatingly sloow when building
if(NT_USE_PCH)
  include(${PROJECT_SOURCE_DIR}/cmake/nuTens-pch.cmake)
  setup_nutens_pch()
endif()

add_subdirectory(tensors)
add_subdirectory(propagator)

add_library(nuTens INTERFACE)
target_link_libraries(nuTens INTERFACE tensor propagator)

target_include_directories(
  nuTens INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                   $<INSTALL_INTERFACE:include>)

install(
  TARGETS nuTens
  DESTINATION lib
  EXPORT nuTens-targets)
