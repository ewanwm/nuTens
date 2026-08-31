# sets up the nuTens-pch target containing precompiled headers for nuTens
function(setup_nutens_pch)

  message("Using precompiled header")

  file(
    GENERATE
    OUTPUT ${PROJECT_BINARY_DIR}/nuTens-pch.cpp
    CONTENT "")

  add_library(nuTens-pch OBJECT ${PROJECT_BINARY_DIR}/nuTens-pch.cpp)
  target_include_directories(
    nuTens-pch PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
                      $<INSTALL_INTERFACE:include>)

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

endfunction()
