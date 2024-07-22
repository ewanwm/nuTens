#pragma once 

/*! \file logging.hpp
    \brief Define the logging utilities for nuTens
    
    Basically just use spdlog interface. 
    However we define our own log levels and macros and pass them through to the logging library.
    This is juuust in case we ever want to change the logging library we use. This way only this file would need to change.
*/

#define NT_LOG_LEVEL_TRACE 0
#define NT_LOG_LEVEL_DEBUG 1
#define NT_LOG_LEVEL_INFO 2
#define NT_LOG_LEVEL_WARNING 3
#define NT_LOG_LEVEL_ERROR 4
#define NT_LOG_LEVEL_SILENT 5


// define the log level in spdlogger
#if NT_LOG_LEVEL == NT_LOG_LEVEL_TRACE
    #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_DEBUG
    #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_INFO
    #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_WARNING
    #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_WARNING

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_ERROR
    #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_ERROR

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_SILENT
    #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_OFF

#endif


// #include "spdlog.h" has to happen *AFTER* we set SPDLOG_ACTIVE_LEVEL
#include "spdlog.h"


namespace ntlogging{

}