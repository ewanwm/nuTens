

## set up logging
add_library(logging logging.hpp)
set_target_properties(logging PROPERTIES LINKER_LANGUAGE CXX)

set( LOG_LEVEL "INFO" CACHE STRING "the level of detail to log to the console" )

## Convert LOG_LEVEL to all upper case so that we aren't case sensitive to user input
string( TOUPPER "${LOG_LEVEL}" LOG_LEVEL_UPPER )

## Check the specified log level is valid
set(VALID_LOG_OPTIONS SILENT ERROR WARNING INFO DEBUG TRACE)
list(FIND VALID_LOG_OPTIONS ${LOG_LEVEL_UPPER} index)
if(${index} GREATER -1)
    message(STATUS "Setting log level to ${LOG_LEVEL_UPPER}")
else()
    message(FATAL_ERROR "Invalid log level specified: ${LOG_LEVEL_UPPER} \n Should be one of: ${VALID_LOG_OPTIONS}")
endif()

## set the log level that will be used inside the logging.hpp file
target_compile_definitions(logging PUBLIC NT_LOG_LEVEL="NT_LOG_LEVEL_${LOG_LEVEL_UPPER}")

add_subdirectory(tensors)
add_subdirectory(propagator)