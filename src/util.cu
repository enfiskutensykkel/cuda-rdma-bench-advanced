#include <cstdio>
#include <cstdarg>
#include <stdexcept>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <sisci_types.h>
#include <sisci_api.h>
#include "log.h"
#include "util.h"



uint64_t currentTime()
{
    timespec ts;

    if (clock_gettime(CLOCK_REALTIME, &ts) < 0)
    {
        throw std::runtime_error("Failed to get realtime clock");
    }

    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}


sci_error_t openDescriptor(sci_desc_t& desc)
{
    sci_error_t err;

    SCIOpen(&desc, 0, &err);
    if (err != SCI_ERR_OK)
    {
        Log::error("Failed to open descriptor: %s", scierrstr(err));
    }

    return err;
}


sci_error_t closeDescriptor(sci_desc_t desc)
{
    sci_error_t err;

    do
    {
        SCIClose(desc, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    if (err != SCI_ERR_OK)
    {
        Log::error("Failed to close descriptor: %s", scierrstr(err));
    }

    return err;
}


uint64_t IOAddress(sci_local_segment_t segment)
{
    sci_error_t err;
    sci_query_local_segment_t query;

    query.subcommand = SCI_Q_LOCAL_SEGMENT_IOADDR;
    query.segment = segment;

    SCIQuery(SCI_Q_LOCAL_SEGMENT, &query, 0, &err);
    if (err != SCI_ERR_OK)
    {
        Log::warn("Failed to query local segment: %s", scierrstr(err));
        return 0;
    }

    return query.data.ioaddr;
}


uint64_t IOAddress(sci_remote_segment_t segment)
{
    sci_error_t err;
    sci_query_remote_segment_t query;

    query.subcommand = SCI_Q_REMOTE_SEGMENT_IOADDR;
    query.segment = segment;

    SCIQuery(SCI_Q_REMOTE_SEGMENT, &query, 0, &err);
    if (err != SCI_ERR_OK)
    {
        Log::warn("Failed to query remote segment: %s", scierrstr(err));
        return 0;
    }

    return query.data.ioaddr;
}


uint64_t physicalAddress(sci_local_segment_t segment)
{
    sci_error_t err;
    sci_query_local_segment_t query;

    query.subcommand = SCI_Q_LOCAL_SEGMENT_PHYS_ADDR;
    query.segment = segment;

    SCIQuery(SCI_Q_LOCAL_SEGMENT, &query, 0, &err);
    if (err != SCI_ERR_OK)
    {
        Log::warn("Failed to query local segment: %s", scierrstr(err));
        return 0;
    }

    return query.data.ioaddr;
}