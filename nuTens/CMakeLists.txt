
########################
#### set up logging ####
########################
add_library(logging logging.hpp)
target_link_libraries(logging spdlog::spdlog)
set_target_properties(logging PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(logging PUBLIC "${SPDLOG_INCLUDE_DIRS}")

set( NT_LOG_LEVEL "INFO" CACHE STRING "the level of detail to log to the console" )

## Convert NT_LOG_LEVEL to all upper case so that we aren't case sensitive to user input
string( TOUPPER "${NT_LOG_LEVEL}" LOG_LEVEL_UPPER )

## Check the specified log level is valid
set(VALID_LOG_OPTIONS SILENT ERROR WARNING INFO DEBUG TRACE)
list(FIND VALID_LOG_OPTIONS ${LOG_LEVEL_UPPER} index)
if(${index} GREATER -1)
    message(STATUS "Setting log level to ${LOG_LEVEL_UPPER}")
else()
    message(FATAL_ERROR "Invalid log level specified: ${LOG_LEVEL_UPPER} \n Should be one of: ${VALID_LOG_OPTIONS}")
endif()

## set the log level that will be used inside the logging.hpp file
target_compile_definitions(logging PUBLIC NT_LOG_LEVEL=NT_LOG_LEVEL_${LOG_LEVEL_UPPER})



################################
#### set up instrumentation ####
################################
add_library(instrumentation instrumentation.hpp)
set_target_properties(instrumentation PROPERTIES LINKER_LANGUAGE CXX)

option( NT_PROFILING "enable profiling of the code" OFF )
if( NT_PROFILING )
    target_compile_definitions( instrumentation PUBLIC USE_PROFILING )
endif()


## if user wants to use pch then we use the pch
## people, especially if developing, might want to use this as including tensor related things
## can be excruciatingly sloow when building
OPTION(NT_USE_PCH "NT_USE_PCH" OFF)
IF(NT_USE_PCH)
    message("Using precompiled header")
    
    add_library(nuTens-pch nuTens-pch.hpp)
    
    SET(PCH_LIBS "${PCH_LIBS};logging;instrumentation")

    ## the headers included in the PCH will (at some point) depend on which tensor library is being used
    IF(TORCH_FOUND)
        message( "Including PyTorch includes in the PCH")
        message( "TORCH_LIBRARIES = ")
        message( "${TORCH_LIBRARIES}")

        target_compile_definitions(nuTens-pch PUBLIC USE_PYTORCH)
        SET(PCH_LIBS "${PCH_LIBS};${TORCH_LIBRARIES}")
    ENDIF()

    target_link_libraries(nuTens-pch PUBLIC "${PCH_LIBS}")
    target_precompile_headers(nuTens-pch PUBLIC nuTens-pch.hpp)
    set_target_properties(nuTens-pch PROPERTIES LINKER_LANGUAGE CXX)

ENDIF() ## end NT_USE_PCH block





add_subdirectory(tensors)
add_subdirectory(propagator)