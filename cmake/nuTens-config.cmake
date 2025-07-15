SET(nuTens_LIB_LIST "-libtensor -libpropagator -libinstrumentation -libnt-logging -libconstants -libunits")

SET(nuTens_FEATURES_LIST)

if(UseGPU EQUAL 1)
  LIST(APPEND NUOSCILLATOR_FEATURES_LIST "GPU")
endif()
LIST(APPEND NUOSCILLATOR_FEATURES_LIST ${NuOscillator_Engines_Enabled})

# Set the creation date
string(TIMESTAMP CREATION_DATE "%d-%m-%Y")

string(REPLACE ";" " " nuTens_FEATURES "${nuTens_FEATURES_LIST}")
configure_file(${CMAKE_CURRENT_LIST_DIR}/templates/nuTens-config.in
  "${PROJECT_BINARY_DIR}/nuTens-config" @ONLY)
install(PROGRAMS
  "${PROJECT_BINARY_DIR}/nuTens-config" DESTINATION
  bin)
