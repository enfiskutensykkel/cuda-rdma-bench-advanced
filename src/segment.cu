#include <cstddef>
#include <cstdlib>
#include <memory>
#include <map>
#include <stdexcept>
#include <cuda.h>
#include <sisci_api.h>
#include <sisci_types.h>
#include "segment.h"
#include "log.h"
#include "util.h"

using std::runtime_error;


Segment::Segment() : 
    segmentId(0),
    deviceId(NO_DEVICE),
    size(0),
    sci_desc(nullptr),
    segment(nullptr),
    buffer(nullptr)
{
}


Segment::~Segment()
{
    sci_error_t err;

    if (segment != nullptr)
    {
        for (ExportMap::const_iterator it = exports.begin(); it != exports.end(); ++it)
        {
            if (it->second)
            {
                Log::debug("Unexporting segment %u on adapter %u...", segmentId, it->first);

                do
                {
                    SCISetSegmentUnavailable(segment, it->first, 0, &err);
                }
                while (err == SCI_ERR_BUSY);

                if (err != SCI_ERR_OK)
                {
                    Log::error("Failed to unexport local segment %u: %s", segmentId, scierrstr(err));
                }
            }
        }
    }

    if (segment != nullptr)
    {
        Log::debug("Removing local segment %u...", segmentId);
        do
        {
            SCIRemoveSegment(segment, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        if (err != SCI_ERR_OK)
        {
            Log::error("Failed to remove local segment %u", segmentId);
        }
    }

    if (buffer != nullptr)
    {
        if (deviceId != NO_DEVICE)
        {
            cudaFree(buffer);
            Log::debug("Free'd GPU buffer");
        }
    }

    if (sci_desc != nullptr)
    {
        closeDescriptor(sci_desc);
    }

    Log::info("Segment %u was removed", segmentId);
}


static sci_error_t fillSegment(Segment& segment)
{
    sci_error_t err;

    sci_map_t map;
    void* ptr = SCIMapLocalSegment(segment.segment, &map, 0, segment.size, nullptr, 0, &err);
    if (err != SCI_ERR_OK)
    {
        Log::error("Failed to map local segment: %s", scierrstr(err));
        return err;
    }

    srand(usecs());
    uint8_t pseudo_random = (rand() & 0xfe) + 1; // should never be 0x00 or 0xff

    for (size_t i = 0; i < segment.size; ++i)
    {
        *((uint8_t*) ptr) = pseudo_random;
    }

    SCIUnmapSegment(map, 0, &err);
    if (err != SCI_ERR_OK)
    {
        Log::error("Failed to unmap local segment: %s", scierrstr(err));
        return err;
    }

    return SCI_ERR_OK;
}


static void* getDevicePtr(int device, void* ptr)
{
    cudaError_t err;
   
    err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        Log::error("Failed to set GPU: %s", cudaGetErrorString(err));
        throw err;
    }

    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess)
    {
        Log::error("Failed to get pointer attributes: %s", cudaGetErrorString(err));
        throw err;
    }

    return attrs.devicePointer;
}


static void* allocGpuBuffer(int device, size_t size)
{
    cudaError_t err;
   
    err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        Log::error("Failed to set GPU: %s", cudaGetErrorString(err));
        throw err;
    }

    void* buffer = nullptr;
    err = cudaMalloc(&buffer, size);
    if (err != cudaSuccess)
    {
        Log::error("Failed to allocated GPU memory: %s", cudaGetErrorString(err));
        throw err;
    }

    try
    {
        void* devicePtr = getDevicePtr(device, buffer);
        unsigned int flag = 1;
        CUresult res = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr) devicePtr);
        if (res != CUDA_SUCCESS)
        {
            Log::warn("Failed to set CU_POINTER_ATTRIBUTE_SYNC_MEMOPS");
        }
    }
    catch (cudaError_t err)
    {
        Log::warn("Failed to set CU_POINTER_ATTRIBUTE_SYNC_MEMOPS");
    }

    return buffer;
}


void Segment::createSegment(bool fill)
{
    sci_error_t err;

    if ((err = openDescriptor(sci_desc)) != SCI_ERR_OK)
    {
        throw runtime_error(scierrstr(err));
    }

    // Create local segment
    unsigned int flag = deviceId != NO_DEVICE ? SCI_FLAG_EMPTY : 0;
    SCICreateSegment(sci_desc, &segment, segmentId, size, NULL, NULL, flag, &err);
    if (err != SCI_ERR_OK)
    {
        Log::error("Failed to create segment %u: %s", segmentId, scierrstr(err));
        throw runtime_error(scierrstr(err));
    }

    // Attach physical memory if GPU buffer
    if (deviceId != NO_DEVICE)
    {
        try
        {
            buffer = allocGpuBuffer(deviceId, size);
        }
        catch (cudaError_t err)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        void *devicePtr = getDevicePtr(deviceId, buffer);
        SCIAttachPhysicalMemory(0, devicePtr, 0, size, segment, SCI_FLAG_CUDA_BUFFER, &err);
        if (err != SCI_ERR_OK)
        {
            Log::error("Failed to attach memory on segment %u: %s", segmentId, scierrstr(err));
            throw runtime_error(scierrstr(err));
        }

        Log::debug("Created segment %u with device pointer 0x%016x", segmentId, (uint64_t) devicePtr);
    }
    else
    {
        Log::debug("Created segment %u with physical address 0x%016x", segmentId, physicalAddress(segment));
    }

    if (fill)
    {
        fillSegment(*this);
    }

    // Prepare segment on all adapters
    for (ExportMap::const_iterator it = exports.begin(); it != exports.end(); ++it)
    {
        SCIPrepareSegment(segment, it->first, 0, &err);
        if (err != SCI_ERR_OK)
        {
            Log::error("Failed to prepare segment %u on adapter %u: %s", segmentId, it->first, scierrstr(err));
            throw runtime_error(scierrstr(err));
        }

        Log::debug("Prepared segment %u on adapter %u with IO address 0x%016x", segmentId, it->first, ioAddress(segment));
    }
}


void Segment::exportSegment()
{
    for (ExportMap::iterator it = exports.begin(); it != exports.end(); ++it)
    {
        sci_error_t err;

        // Get local node identifier
        unsigned int localNodeId;
        SCIGetLocalNodeId(it->first, &localNodeId, 0, &err);
        if (err != SCI_ERR_OK)
        {
            Log::warn("Failed to get local node id for adapter %u", it->first);
        }

        // Set segment available
        SCISetSegmentAvailable(segment, it->first, 0, &err);
        if (err != SCI_ERR_OK)
        {
            Log::error("Failed to set segment %u available on adapter %u: %s", segmentId, it->first, scierrstr(err));
            throw err;
        }

        // Indicate that segment is available
        it->second = true;

        Log::info("Connect to segment %u on node %u", segmentId, localNodeId);
    }
}
