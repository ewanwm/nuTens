#pragma once

#include <cstddef>
#include <iostream>

/*! \file logging.hpp
    \brief Define the logging utilities for nuTens

    Basically just use spdlog interface.
    However we define our own log levels and macros and pass them through to
   the logging library. This is juuust in case we ever want to change the
   logging library we use. This way only this file would need to change.
*/

constexpr size_t NT_LOG_LEVEL_TRACE = 0;
constexpr size_t NT_LOG_LEVEL_DEBUG = 1;
constexpr size_t NT_LOG_LEVEL_INFO = 2;
constexpr size_t NT_LOG_LEVEL_WARNING = 3;
constexpr size_t NT_LOG_LEVEL_ERROR = 4;
constexpr size_t NT_LOG_LEVEL_SILENT = 5;

// define the log level in spdlogger
#if NT_LOG_LEVEL == NT_LOG_LEVEL_TRACE
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE // NOLINT

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_DEBUG
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG // NOLINT

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_INFO
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO // NOLINT

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_WARNING
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_WARNING // NOLINT

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_ERROR
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_ERROR // NOLINT

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_SILENT
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_OFF // NOLINT

#endif

// #include "spdlog.h" has to happen *AFTER* we set SPDLOG_ACTIVE_LEVEL
#include <spdlog/spdlog.h>

// Now define the runtime log level which we will use to set the default log
// level This is needed since for trace or debug, we need to alter the default
// value at runtime see
// https://github.com/gabime/spdlog/wiki/1.-QuickStart#:~:text=Notice%20that%20spdlog%3A%3Aset_level%20is%20also%20necessary%20to%20print%20out%20debug%20or%20trace%20messages.
#if NT_LOG_LEVEL == NT_LOG_LEVEL_TRACE
const static spdlog::level::level_enum runtimeLogLevel = spdlog::level::trace;

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_DEBUG
const static spdlog::level::level_enum runtimeLogLevel = spdlog::level::debug;

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_INFO
const static spdlog::level::level_enum runtimeLogLevel = spdlog::level::info;

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_WARNING
const static spdlog::level::level_enum runtimeLogLevel = spdlog::level::warning;

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_ERROR
const static spdlog::level::level_enum runtimeLogLevel = spdlog::level::error;

#elif NT_LOG_LEVEL == NT_LOG_LEVEL_SILENT
const static spdlog::level::level_enum runtimeLogLevel = spdlog::level::off;

#endif

namespace ntlogging
{
static std::once_flag
    logLevelOnceFlag; // NOLINT: Linter gets angry that this is globally accessible and non-const. Needs to be non-const
                      // so it can be flipped by the call to std::call_once. Could wrap it up so that it's not global
                      // but that feels like a bit much for what we're doing here

/// @brief Set up the logger at runtime, should only be invoked once the very
/// first time any of the logging macros below are called
inline void setup_logging()
{
    std::call_once(logLevelOnceFlag, []() {
        std::cout << ":::::::: INFO: Setting default spdlog logging level to "
                  << spdlog::level::to_string_view(runtimeLogLevel).data() << " ::::::::" << std::endl;
        spdlog::set_level(runtimeLogLevel);
    });
}
} // namespace ntlogging

/// @brief Trace message that will only be displayed if NT_LOG_LEVEL ==
/// NT_LOG_LEVEL_TRACE
/// @param[in] ... The message to print. This can consist of just a simple
/// string, or a format string and subsequent variables to format.
// NOLINTNEXTLINE
#define NT_TRACE(...)                                                                                                  \
    ntlogging::setup_logging();                                                                                        \
    SPDLOG_TRACE(__VA_ARGS__)

/// @brief Debug message that will only be displayed if NT_LOG_LEVEL <=
/// NT_LOG_LEVEL_DEBUG
/// @param[in] ... The message to print. This can consist of just a simple
/// string, or a format string and subsequent variables to format.
// NOLINTNEXTLINE
#define NT_DEBUG(...)                                                                                                  \
    ntlogging::setup_logging();                                                                                        \
    SPDLOG_DEBUG(__VA_ARGS__)

/// @brief Information message that will only be displayed if NT_LOG_LEVEL <=
/// NT_LOG_LEVEL_INFO
/// @param[in] ... The message to print. This can consist of just a simple
/// string, or a format string and subsequent variables to format.
// NOLINTNEXTLINE
#define NT_INFO(...)                                                                                                   \
    ntlogging::setup_logging();                                                                                        \
    SPDLOG_INFO(__VA_ARGS__)

/// @brief Warning message that will only be displayed if NT_LOG_LEVEL <=
/// NT_LOG_LEVEL_WARNING
/// @param[in] ... The message to print. This can consist of just a simple
/// string, or a format string and subsequent variables to format.
// NOLINTNEXTLINE
#define NT_WARN(...)                                                                                                   \
    ntlogging::setup_logging();                                                                                        \
    SPDLOG_WARN(__VA_ARGS__)

/// @brief Error message that will only be displayed if NT_LOG_LEVEL <=
/// NT_LOG_LEVEL_ERROR
/// @param[in] ... The message to print. This can consist of just a simple
/// string, or a format string and subsequent variables to format.
// NOLINTNEXTLINE
#define NT_ERROR(...)                                                                                                  \
    ntlogging::setup_logging();                                                                                        \
    SPDLOG_ERROR(__VA_ARGS__)
