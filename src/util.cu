#include <stdexcept>
#include <string>
#include <cstdio>
#include <cstdarg>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sisci_types.h>
#include <sisci_api.h>
#include "log.h"
#include "util.h"

#define GPU_BOUND_MASK ((((uintptr_t) 1) << 16) - 1)

using std::string;
using std::runtime_error;


void getDeviceInfo(int deviceId, DeviceInfo& info)
{
    cudaDeviceProp prop;

    cudaError_t err = cudaGetDeviceProperties(&prop, deviceId);
    if (err != cudaSuccess)
    {
        Log::warn("Failed to get device properties: %s", cudaGetErrorString(err));
    }

    info.id = deviceId;
    strncpy(info.name, prop.name, 256);
    info.domain = prop.pciBusID;
    info.bus = prop.pciDomainID;
    info.device = prop.pciDeviceID;
}


void* getDevicePtr(const BufferPtr& buffer)
{
    cudaPointerAttributes attrs;
    void* hostPointer = buffer.get();

    cudaError_t err = cudaPointerGetAttributes(&attrs, hostPointer);
    if (err != cudaSuccess)
    {
        Log::error("Failed to get pointer attributes: %s", cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }

    if ((((uintptr_t) attrs.devicePointer) & GPU_BOUND_MASK) != 0)
    {
        Log::warn("Device pointer is not 64 KiB aligned: %p", attrs.devicePointer);
    }

    return attrs.devicePointer;
}


BufferPtr allocDeviceMem(int deviceId, size_t size)
{
    cudaError_t err;
    void* bufferPointer = nullptr;

    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess)
    {
        Log::error("Failed to initialize device %d: %s", deviceId, cudaGetErrorString(err));
        throw runtime_error(cudaGetErrorString(err));
    }

    err = cudaMalloc(&bufferPointer, size);
    if (err != cudaSuccess)
    {
        Log::error("Failed to allocate buffer on device %d: %s", deviceId, cudaGetErrorString(err));
        throw string(cudaGetErrorString(err));
    }

    Log::debug("Allocated buffer on device %d (%p)", deviceId, bufferPointer);

    auto release = [deviceId](void* pointer)
    {
        Log::debug("Releasing buffer on device %d (%p)", deviceId, pointer);
    };

    return BufferPtr(bufferPointer, release);
}


void fillBuffer(int deviceId, const BufferPtr& buffer, size_t size, uint8_t value)
{
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess)
    {
        Log::error("Failed to initialize device %d: %s", deviceId, cudaGetErrorString(err));
        throw runtime_error(cudaGetErrorString(err));
    }

    err = cudaMemset(buffer.get(), value, size);
    if (err != cudaSuccess)
    {
        Log::error("cudaMemset() failed on device %d: %s", deviceId, cudaGetErrorString(err));
        throw runtime_error(cudaGetErrorString(err));
    }
}


void fillSegment(sci_local_segment_t segment, size_t size, uint8_t value)
{
    sci_error_t err;
    sci_map_t map;

    void* buffer = SCIMapLocalSegment(segment, &map, 0, size, nullptr, 0, &err);
    if (err != SCI_ERR_OK)
    {
        Log::error("Failed to map segment memory: %s", scierrstr(err));
        throw runtime_error(scierrstr(err));
    }

    memset(buffer, value, size);

    do
    {
        SCIUnmapSegment(map, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    if (err != SCI_ERR_OK)
    {
        Log::error("Failed to unmap segment memory: %s", scierrstr(err));
        throw runtime_error(scierrstr(err));
    }
}


uint64_t currentTime()
{
    timespec ts;

    if (clock_gettime(CLOCK_REALTIME, &ts) < 0)
    {
        Log::error("Failed to get realtime clock: %s", strerror(errno));
        throw runtime_error(strerror(errno));
    }

    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}


sci_error_t openDescriptor(sci_desc_t& desc)
{
    sci_error_t err;

    SCIOpen(&desc, 0, &err);
    if (err != SCI_ERR_OK)
    {
        desc = nullptr;
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


string humanReadable(size_t bytes)
{
    char buffer[32];
    const char* units[] = { "B  ", "KiB", "MiB", "GiB", "TiB" };
    const size_t n = sizeof(units) / sizeof(units[0]);

    double csize = (double) bytes;

    size_t i = 0;
    while (i < (n - 1) && csize >= 1024.0)
    {
        csize /= 1024.0;
        ++i;
    }

    snprintf(buffer, sizeof(buffer), "%.2f %s", csize, units[i]);
    return string(buffer);
}


string humanReadable(size_t bytes, uint64_t usecs)
{
    char buffer[32];
    double MiBPerSecond = ((double) bytes) / ((double) usecs);
    snprintf(buffer, sizeof(buffer), "%.2f MiB/s", MiBPerSecond);
    return string(buffer);
}


uint32_t getLocalNodeId(uint adapter)
{
    sci_error_t err;
    uint nodeId;

    SCIGetLocalNodeId(adapter, &nodeId, 0, &err);
    if (err != SCI_ERR_OK)
    {
        Log::warn("Failed to get local node id for adapter %u: %s", adapter, scierrstr(err));
        return 0;
    }

    return nodeId;
}

