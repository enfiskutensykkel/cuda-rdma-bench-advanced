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
        ERROR   = 0,
        WARN    = 1,
        INFO    = 2,
        DEBUG   = 3
    };


    /* Initialize log file */
    void init(FILE* logfile, Level loglevel);


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
