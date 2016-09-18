#include <mutex>
#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <ctime>
#include "util.h"
#include "log.h"


static FILE* logFile = stderr;


static Log::Level logLevel = Log::Level::ERROR;


static std::mutex logLock;


/* Known SISCI error codes */
static unsigned error_codes[] = {
    SCI_ERR_OK,
    SCI_ERR_BUSY,
    SCI_ERR_FLAG_NOT_IMPLEMENTED,
    SCI_ERR_ILLEGAL_FLAG,
    SCI_ERR_NOSPC,
    SCI_ERR_API_NOSPC,
    SCI_ERR_HW_NOSPC,
    SCI_ERR_NOT_IMPLEMENTED,
    SCI_ERR_ILLEGAL_ADAPTERNO,
    SCI_ERR_NO_SUCH_ADAPTERNO,
    SCI_ERR_TIMEOUT,
    SCI_ERR_OUT_OF_RANGE,
    SCI_ERR_NO_SUCH_SEGMENT,
    SCI_ERR_ILLEGAL_NODEID,
    SCI_ERR_CONNECTION_REFUSED,
    SCI_ERR_SEGMENT_NOT_CONNECTED,
    SCI_ERR_SIZE_ALIGNMENT,
    SCI_ERR_OFFSET_ALIGNMENT,
    SCI_ERR_ILLEGAL_PARAMETER,
    SCI_ERR_MAX_ENTRIES,
    SCI_ERR_SEGMENT_NOT_PREPARED,
    SCI_ERR_ILLEGAL_ADDRESS,
    SCI_ERR_ILLEGAL_OPERATION,
    SCI_ERR_ILLEGAL_QUERY,
    SCI_ERR_SEGMENTID_USED,
    SCI_ERR_SYSTEM,
    SCI_ERR_CANCELLED,
    SCI_ERR_NOT_CONNECTED,
    SCI_ERR_NOT_AVAILABLE,
    SCI_ERR_INCONSISTENT_VERSIONS,
    SCI_ERR_COND_INT_RACE_PROBLEM,
    SCI_ERR_OVERFLOW,
    SCI_ERR_NOT_INITIALIZED,
    SCI_ERR_ACCESS,
    SCI_ERR_NOT_SUPPORTED,
    SCI_ERR_DEPRECATED,
    SCI_ERR_NO_SUCH_NODEID,
    SCI_ERR_NODE_NOT_RESPONDING,
    SCI_ERR_NO_REMOTE_LINK_ACCESS,
    SCI_ERR_NO_LINK_ACCESS,
    SCI_ERR_TRANSFER_FAILED,
    SCI_ERR_EWOULD_BLOCK,
    SCI_ERR_SEMAPHORE_COUNT_EXCEEDED,
    SCI_ERR_IRQL_ILLEGAL,
    SCI_ERR_REMOTE_BUSY,
    SCI_ERR_LOCAL_BUSY,
    SCI_ERR_ALL_BUSY
}; 


/* Corresponding error strings */
static const char* error_strings[] = {
    "OK",
    "Resource busy",
    "Flag option is not implemented",
    "Illegal flag option",
    "Out of local resources",
    "Out of local API resources",
    "Out of hardware resources",
    "Not implemented",
    "Illegal adapter number",
    "Adapter not found",
    "Operation timed out",
    "Out of range",
    "Segment ID not found",
    "Illegal node ID",
    "Connection to remote node is refused",
    "No connection to segment",
    "Size is not aligned",
    "Offset is not aligned",
    "Illegal function parameter",
    "Maximum possible physical mapping is exceeded",
    "Segment is not prepared",
    "Illegal address",
    "Illegal operation",
    "Illegal query operation",
    "Segment ID already used",
    "Could not get requested resource from the system",
    "Operation cancelled",
    "Host is not connected to remote host",
    "Operation not available",
    "Inconsistent driver version",
    "Out of local resources",
    "Host not initialized",
    "No local or remote access for requested operation",
    "Request not supported",
    "Function deprecated",
    "Node ID not found",
    "Node does not respond",
    "Remote link is not operational",
    "Local link is not operational",
    "Transfer failed",
    "Illegal interrupt line",
    "Remote host is busy",
    "Local host is busy",
    "System is busy"
};


static inline size_t scierridx(sci_error_t code)
{
    const size_t num = sizeof(error_codes) / sizeof(error_codes[0]);

    size_t idx;

    for (idx = 0; idx < num; ++idx)
    {
        if (error_codes[idx] == code)
        {
            return idx;
        }
    }

    return idx;
}


const char* scierrstr(sci_error_t code)
{
    const size_t idx = scierridx(code);

    if (idx < sizeof(error_codes) / sizeof(error_codes[0]))
    {
        return error_strings[idx];
    }

    return "Undocumented error";
}


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

    std::lock_guard<std::mutex> lock(logLock);

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


void Log::abort(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    report("ABORT", 5, format, args);
    va_end(args);
}


void Log::error(const char* format, ...)
{
    if (logLevel >= Log::Level::ERROR || logFile != stderr)
    {
        va_list args;
        va_start(args, format);
        report("ERROR", 5, format, args);
        va_end(args);
    }
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

