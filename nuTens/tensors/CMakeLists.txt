# interface to the tensor backend
add_library(tensor-backend INTERFACE)

if(NT_USE_TORCH)
  add_library(tensor tensor.hpp torch-tensor.cpp dtypes.hpp autograd.hpp
                     torch-autograd.cpp)
  target_compile_definitions(tensor PUBLIC USE_PYTORCH)
  target_link_libraries(tensor-backend INTERFACE "${TORCH_LIBRARIES}")
else()
  message(
    FATAL_ERROR
      "No library found to deal with tensors. Currently only pytorch is available, please install this and try again."
  )
endif()
# when new tensor backends are added will add coresponding <library>-tensor.cpp
# and compile accordingly

target_include_directories(
  tensor PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
                $<INSTALL_INTERFACE:include>)

if(NT_USE_PCH)
  target_precompile_headers(tensor REUSE_FROM nuTens-pch)
endif()

target_link_libraries(tensor PUBLIC nt-logging instrumentation tensor-backend)

# install headers to "include" directory
set(HEADERS tensor.hpp;dtypes.hpp;autograd.hpp)
install(FILES ${HEADERS} DESTINATION include/tensor)

# install libraries to the "lib" directory
install(
  TARGETS tensor tensor-backend
  DESTINATION lib
  EXPORT nuTens-targets)
