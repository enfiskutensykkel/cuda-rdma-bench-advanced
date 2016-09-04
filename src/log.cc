#include <cstdio>
#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <ctime>
#include "util.h"
#include "log.h"


static FILE* logFile = stderr;


static Log::Level logLevel = Log::Level::ERROR;


/* Helper function to format the log string */
static inline void report(const char* prefix, size_t len, const char* format, va_list arguments)
{
    char buffer[1024];
    size_t length;

    // If not printing to stderr, include timestamp in log
    if (logFile != stderr)
    {
        time_t now = time(nullptr);
        length = strftime(buffer, sizeof(buffer), "[%Y-%m-%d %H:%M:%S] ", localtime(&now));
        fwrite(buffer, length, 1, logFile);
    }

    // Make pretty print stuff
    fputc('<', logFile);
    fwrite(prefix, len, 1, logFile);
    fputc('>', logFile);
    for (size_t i = 0; i < 6 - len; ++i)
    {
        fputc(' ', logFile);
    }

    // Now print the actual message
    length = vsnprintf(buffer, sizeof(buffer), format, arguments);

    fwrite(buffer, length, 1, logFile);
    fwrite("\n", 1, 1, logFile);
    fflush(logFile);
}


void Log::error(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    report("ERROR", 5, format, args);
    va_end(args);
}


void Log::warn(const char* format, ...)
{
    if (logLevel >= Log::Level::WARN || logFile != stderr)
    {
        va_list args;
        va_start(args, format);
        report("WARN", 4, format, args);
        va_end(args);
    }
}


void Log::info(const char* format, ...)
{
    if (logLevel >= Log::Level::INFO || logFile != stderr)
    {
        va_list args;
        va_start(args, format);
        report("INFO", 4, format, args);
        va_end(args);
    }
}


void Log::debug(const char* format, ...)
{
    if (logLevel >= Log::Level::DEBUG)
    {
        va_list args;
        va_start(args, format);
        report("DEBUG", 5, format, args);
        va_end(args);
    }
}


void Log::init(FILE* file, Level level)
{
    logFile = file;
    logLevel = level;
}

