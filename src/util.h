#ifndef __RDMA_BENCH_UTIL_H__
#define __RDMA_BENCH_UTIL_H__

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


/* Get information about a CUDA device */
void getDeviceInfo(int deviceId, DeviceInfo& info);


/* Get a CUDA device pointer from a pointer allocated with cudaMalloc */
void* getDevicePtr(void* hostPointer);


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


#endif
