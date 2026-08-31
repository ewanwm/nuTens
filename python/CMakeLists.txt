pybind11_add_module(pyNuTens MODULE binding.cpp)

if(NT_USE_PCH)
  target_precompile_headers(pyNuTens REUSE_FROM nuTens-pch)
endif()
target_link_libraries(pyNuTens PUBLIC nuTens test-utils)

# This is passing in the version as a define just as an example
target_compile_definitions(pyNuTens
                           PRIVATE VERSION_INFO=${CMAKE_PROJECT_VERSION})
target_compile_definitions(pyNuTens PUBLIC USE_PYTORCH)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH
             "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(pyNuTens PRIVATE ${TORCH_LIBRARIES}
                                        ${TORCH_PYTHON_LIBRARY})

install(TARGETS pyNuTens DESTINATION nuTens/)
install(DIRECTORY nuTens DESTINATION .)
