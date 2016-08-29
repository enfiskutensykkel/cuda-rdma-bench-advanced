#ifndef __UTIL_H__
#define __UTIL_H__

#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <sisci_api.h>


/* Global log utility class */
struct LogUtil;
extern LogUtil logger;


/* Get current timestamp (microseconds) */
uint64_t current_usecs();


/* Translate a SISCI error code to a sensible string */
const char* scierrstr(sci_error_t error);


/* Log utility class definition */
struct LogUtil
{
    public:
        unsigned int level;
        const uint64_t start;

        LogUtil();
        ~LogUtil();

        /* Report an error condition */
        static void error(const char* format, ...);

        /* Warn user about a potential problem */
        static void warn(const char* format, ...);

        /* Inform user about something */
        static void info(const char* format, ...);

        /* Debug information */
        static void debug(const char* format, ...);

        void setLogFile(const char* filename);

    private:
        size_t error_count;
        FILE* file;
};


#endif
