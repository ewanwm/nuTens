#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <utility>

/*!
 * @file instrumentation.hpp
 * @brief Define utilities for instrumentation of the code
 *
 * This is the home of anything that gets placed inside other classes or functions in order to instrument the code.
 * e.g. for profiling or debugging. Everything should ideally be macro-fied so it can be included only for certain
 * builds, or specified by build time options.
 */

struct ProfileResult
{
    /// @struct ProfileResult
    /// @brief Hold the results of a profiled function to be written out.

    std::string name;
    long start;
    long end;
    uint32_t threadID;
};

class ProfileWriter
{
    /*!
     * @class ProfileWriter
     * @brief Singleton class to collect timing information for functions and write out to a file that can be inspected
     * later with visual profiling tool
     *
     * Writes out profiling information in a json format readable by chrome tracing
     * (https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/) Use the macros provided to instrument
     * the source code like:
     *
     * @code{.cpp}
     * \\ header.hpp
     *
     * class ClassName
     * {
     *   returnType func(args);
     * }
     *
     *
     * \\ implementation.cpp
     *
     * ClassName::func(args)
     * {
     *   NT_PROFILE();
     *
     *   \\ implementation code
     *
     * }
     * @endcode
     *
     * In order to instantiate the ProfileWriter in an application you will then need to use NT_PROFILE_BEGINSESSION()
     * and NT_PROFILE_ENDSESSION() like:
     *
     * @code{.cpp}
     *
     * \\ application.cpp
     *
     * void main()
     * {
     *   NT_PROFILE_BEGINSESSION(sessionName);
     *
     *   \\ ... code ...
     *   ClassName instance;
     *
     *   instance.func(args);
     *
     *   \\ ... code ...
     *
     *   NT_PROFILE_ENDSSION();
     * }
     *
     * @endcode
     *
     * This will save a json file called <sessionName>-profile.json.
     * Then you can open up your favourite chromium based browser and go to chrome://tracing. You can then just drag and
     * drop the profiling json file and should see a lovely display of the collected profile information.
     */

    /// @todo currently only suppor the format used by chrome tracing. Would be nice to support other formats too.
    /// Should just be a case of adding additional option for writeProfile and header and footer

  public:
    /// @brief Constructor
    ProfileWriter() = default;

    /// @brief Set up the session
    /// @param[in] name The name of the timer
    /// @param[in] filePath The destination of the output file
    void beginSession(const std::string &name, const std::string &filePath = "results.json")
    {
        _outputStream.open(filePath);
        writeHeader();
        _name = name;
    }

    /// @brief Close the session and clean up
    void endSession()
    {
        writeFooter();
        _outputStream.close();
        _name = "";
        _profileCount = 0;
    }

    /// @brief Write out the results of a profiled function
    /// @param[in] result The result to write
    void writeProfile(const ProfileResult &result)
    {
        if (_profileCount++ > 0)
        {
            _outputStream << ",";
        }

        std::string name = result.name;
        std::replace(name.begin(), name.end(), '"', '\'');

        _outputStream << "{";
        _outputStream << R"("cat":"function",)";
        _outputStream << "\"dur\":" << (result.end - result.start) << ',';
        _outputStream << R"("name":")" << name << "\",";
        _outputStream << R"("ph":"X",)";
        _outputStream << "\"pid\":0,";
        _outputStream << "\"tid\":" << result.threadID << ",";
        _outputStream << "\"ts\":" << result.start;
        _outputStream << "}";

        _outputStream.flush();
    }

    /// @brief Write the file header
    void writeHeader()
    {
        _outputStream << R"({"otherData": {},"traceEvents":[)";
        _outputStream.flush();
    }

    /// @brief Write the file footer
    void writeFooter()
    {
        _outputStream << "]}";
        _outputStream.flush();
    }

    /// @brief Get a reference to the ProfileWriter, if it has not yet been instantiated, this will do so
    static ProfileWriter &get()
    {
        static ProfileWriter instance; // this will be instantiated the first time ProfileWriter::get() is called and
                                       // killed at the end of the program
        return instance;
    }

  private:
    std::string _name;
    std::ofstream _outputStream;
    uint _profileCount{0};
};

class InstrumentationTimer
{
    /*!
     * @class InstrumentationTimer
     * @brief Class to perform timing
     *
     * Gets created at the start of the scope to time then will be deleted when the scope ends.
     * When deleted, will write out timing information to the output stream defined by ProfileWriter.
     *
     *
     */

  public:
    /// @brief Construct an InstrumentationTimer object and start the clock
    /// @param[in] name The name of the profile. Typically use __PRETTY_FUNCTION__ so it's clear which part of the code
    /// is being profiled.
    InstrumentationTimer(std::string name) : _name(std::move(name)), _stopped(false)
    {
        _startTimepoint = std::chrono::high_resolution_clock::now();
    }

    /// @brief Destroy the timer object and stop the timer by calling stop()
    ~InstrumentationTimer()
    {
        if (!_stopped)
        {
            stop();
        }
    }

    InstrumentationTimer(const InstrumentationTimer &) = delete;
    /// @brief Stop the timer and write out the profile result using the ProfileWriter
    void stop()
    {
        auto endTimepoint = std::chrono::high_resolution_clock::now();

        long long start =
            std::chrono::time_point_cast<std::chrono::microseconds>(_startTimepoint).time_since_epoch().count();
        long long end =
            std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();

        uint32_t threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());
        ProfileWriter::get().writeProfile({_name, start, end, threadID});

        _stopped = true;
    }

  private:
    std::string _name;
    std::chrono::time_point<std::chrono::high_resolution_clock> _startTimepoint;
    bool _stopped;
};

/// @brief Begin a profiling session
/// Will open up the results json file and set things up.
/// If USE_PROFILING not defined will be empty so that it can be stripped from non-debug builds
/// @param[in] sessionName The name of the session
#ifdef USE_PROFILING
// NOLINTNEXTLINE
#define NT_PROFILE_BEGINSESSION(sessionName)                                                                           \
    ProfileWriter::get().beginSession(sessionName, std::string(sessionName) + "-profile.json")
#else
#define NT_PROFILE_BEGINSESSION(sessionName)
#endif

/// @brief Profile the current scope
/// Shold always be used at the very start of the scope.
#ifdef USE_PROFILING
// NOLINTNEXTLINE
#define NT_PROFILE() InstrumentationTimer timer##__LINE__(std::string(__PRETTY_FUNCTION__))
#else
#define NT_PROFILE()
#endif

/// @brief End the profiling session
/// Should be used at the very end of an application, after all functions containing a NT_PROFILE() have been called
#ifdef USE_PROFILING
// NOLINTNEXTLINE
#define NT_PROFILE_ENDSESSION() ProfileWriter::get().endSession()
#else
#define NT_PROFILE_ENDSESSION()
#endif