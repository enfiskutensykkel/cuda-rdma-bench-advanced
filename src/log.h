#ifndef __RDMA_BENCH_LOG_H__
#define __RDMA_BENCH_LOG_H__

#include <cstdio>
#include <cstdarg>
#include <sisci_types.h>


namespace Log
{
    /* Different log levels */
    enum Level : unsigned int
    {
        ABORT   = 0,
        ERROR   = 1,
        WARN    = 2,
        INFO    = 3,
        DEBUG   = 4
    };

    /* Initialize log file */
    void init(FILE* logfile, Level loglevel);

    /* Report a critical error */
    void abort(const char* format, ...);

    /* Report an error condition */
    void error(const char* format, ...);


    /* Warn user about a potential problem */
    void warn(const char* format, ...);


    /* Inform user about something */
    void info(const char* format, ...);


    /* Debug information */
    void debug(const char* format, ...);
};


/* Translate a SISCI error code to a sensible string */
const char* scierrstr(sci_error_t error);


#endif
