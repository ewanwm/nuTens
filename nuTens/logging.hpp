#pragma once

/*! \file logging.hpp
    \brief Define the logging utilities for nuTens

    Basically just use spdlog interface.
    However we define our own log levels and macros and pass them through to
   the logging library. This is juuust in case we ever want to change the
   logging library we use. This way only this file would need to change.
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
#include "spdlog/spdlog.h"

// Now define the runtime log level which we will use to set the default log
// level This is needed since for trace or debug, we need to alter the default
// value at runtime see
// https://github.com/gabime/spdlog/wiki/1.-QuickStart#:~:text=Notice%20that%20spdlog%3A%3Aset_level%20is%20also%20necessary%20to%20print%20out%20debug%20or%20trace%20messages.
#if NT_LOG_LEVEL == NT_LOG_LEVEL_TRACE
static spdlog::level::level_enum runtimeLogLevel = spdlog::level::trace;

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_DEBUG
static spdlog::level::level_enum runtimeLogLevel = spdlog::level::debug;

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_INFO
static spdlog::level::level_enum runtimeLogLevel = spdlog::level::info;

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_WARNING
static spdlog::level::level_enum runtimeLogLevel = spdlog::level::warning;

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_ERROR
static spdlog::level::level_enum runtimeLogLevel = spdlog::level::error;

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_SILENT
static spdlog::level::level_enum runtimeLogLevel = spdlog::level::off;

#endif

static std::once_flag once;

/// @brief Set up the logger at runtime, should only be invoked once the very
/// first time any of the logging macros below are called
inline void setup_logging()
{
    std::call_once(once, []() {
        std::cout << ":::::::: INFO: Setting default spdlog logging level to "
                  << spdlog::level::to_string_view(runtimeLogLevel).data() << " ::::::::" << std::endl;
        spdlog::set_level(runtimeLogLevel);
    });
}

/// @brief Trace message that will only be displayed if NT_LOG_LEVEL ==
/// NT_LOG_LEVEL_TRACE
/// @param[in] ... The message to print. This can consist of just a simple
/// string, or a format string and subsequent variables to format.
#define NT_TRACE(...)                                                                                                  \
    setup_logging();                                                                                                   \
    SPDLOG_TRACE(__VA_ARGS__)

/// @brief Debug message that will only be displayed if NT_LOG_LEVEL <=
/// NT_LOG_LEVEL_DEBUG
/// @param[in] ... The message to print. This can consist of just a simple
/// string, or a format string and subsequent variables to format.
#define NT_DEBUG(...)                                                                                                  \
    setup_logging();                                                                                                   \
    SPDLOG_DEBUG(__VA_ARGS__)

/// @brief Information message that will only be displayed if NT_LOG_LEVEL <=
/// NT_LOG_LEVEL_INFO
/// @param[in] ... The message to print. This can consist of just a simple
/// string, or a format string and subsequent variables to format.
#define NT_INFO(...)                                                                                                   \
    setup_logging();                                                                                                   \
    SPDLOG_INFO(__VA_ARGS__)

/// @brief Warning message that will only be displayed if NT_LOG_LEVEL <=
/// NT_LOG_LEVEL_WARNING
/// @param[in] ... The message to print. This can consist of just a simple
/// string, or a format string and subsequent variables to format.
#define NT_WARN(...)                                                                                                   \
    setup_logging();                                                                                                   \
    SPDLOG_WARN(__VA_ARGS__)

/// @brief Error message that will only be displayed if NT_LOG_LEVEL <=
/// NT_LOG_LEVEL_ERROR
/// @param[in] ... The message to print. This can consist of just a simple
/// string, or a format string and subsequent variables to format.
#define NT_ERROR(...)                                                                                                  \
    setup_logging();                                                                                                   \
    SPDLOG_ERROR(__VA_ARGS__)
