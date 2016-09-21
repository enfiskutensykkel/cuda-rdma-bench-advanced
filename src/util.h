#ifndef __RDMA_BENCH_UTIL_H__
#define __RDMA_BENCH_UTIL_H__

#include <map>
#include <memory>
#include <string>
#include <cstddef>
#include <cstdint>
#include <sisci_types.h>



/* CUDA device info descriptor */
struct DeviceInfo 
{
    int     id;         // CUDA device ID
    char    name[256];  // CUDA device name
    int     domain;     // PCI domain
    int     bus;        // PCI bus
    int     device;     // PCI device
};


/* Convenience type for CUDA buffer */
typedef std::shared_ptr<void> BufferPtr;


/* Get information about a CUDA device */
void getDeviceInfo(int deviceId, DeviceInfo& info);


/* Allocate a buffer on a CUDA device with cudaMalloc */
BufferPtr allocDeviceMem(int deviceId, size_t size);


/* Fill buffer with value */
void fillBuffer(int deviceId, const BufferPtr& buffer, size_t size, uint8_t value);


/* Fill segment with value */
void fillSegment(sci_local_segment_t segment, size_t size, uint8_t value);


/* Get a CUDA device pointer from a pointer allocated with cudaMalloc */
void* getDevicePtr(const BufferPtr& buffer);


/* Get current timestamp (microseconds) */
uint64_t currentTime();


/* Helper function for opening a SISCI descriptor */
sci_error_t openDescriptor(sci_desc_t& descriptor);


/* Helper function for closing a SISCI descriptor */
sci_error_t closeDescriptor(sci_desc_t descriptor);


/* Get the IO address of a local segment */
uint64_t IOAddress(sci_local_segment_t segment);


/* Get the IO address of a remote segment */
uint64_t IOAddress(sci_remote_segment_t segment);


/* Get the physicl address of a segment */
uint64_t physicalAddress(sci_local_segment_t segment);


/* Convert bytes into a convenient human readable form */
std::string humanReadable(size_t bytes);


/* Convert transfer speed into a convenient human readable form */
std::string humanReadable(size_t bytes, uint64_t usecs);


/* Get local node id on an adapter */
uint32_t getLocalNodeId(uint adapter);

#endif
