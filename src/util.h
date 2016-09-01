#ifndef __UTIL_H__
#define __UTIL_H__

#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <sisci_api.h>


/* Get current timestamp (microseconds) */
uint64_t current_usecs();


/* Translate a SISCI error code to a sensible string */
const char* scierrstr(sci_error_t error);


void initLog(FILE* file, uint level);


/* Report an error condition */
void error(const char* format, ...);


/* Warn user about a potential problem */
void warn(const char* format, ...);


/* Inform user about something */
void info(const char* format, ...);


/* Debug information */
void debug(const char* format, ...);


#endif
