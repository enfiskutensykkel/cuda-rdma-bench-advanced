#include <cstddef>
#include <memory>
#include <map>
#include <stdexcept>
#include <sisci_api.h>
#include <sisci_types.h>
#include "segment.h"
#include "log.h"
#include "util.h"

using std::runtime_error;
using std::shared_ptr;
using std::map;


/* Definition of implementation class */
struct SegmentImpl
{
    uint                    id;         // segment id
    sci_desc_t              sd;         // SISCI descriptor
    sci_local_segment_t     segment;    // SISCI segment descriptor
    map<uint, bool>         exports;    // map over exports ordered by adapter number
    void*                   buffer;     // pointer to memory buffer
    sci_map_t               memoryMap;  // segment map
};


static void releaseSegment(SegmentImpl* seginfo)
{
    sci_error_t err;

    // Unmap segment memory
    if (seginfo->memoryMap != nullptr)
    {
        do 
        {
            SCIUnmapSegment(seginfo->memoryMap, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        if (err != SCI_ERR_OK)
        {
            Log::error("Failed to unmap segment %u memory: %s", seginfo->id, scierrstr(err));
        }
    }

    // Unexport segment
    for (auto it = seginfo->exports.begin(); it != seginfo->exports.end(); ++it)
    {
        if (it->second == true)
        {
            Log::debug("Unexporting segment %u on adapter %u...", seginfo->id, it->first);

            do
            {
                SCISetSegmentUnavailable(seginfo->segment, it->first, 0, &err);
            }
            while (err == SCI_ERR_BUSY);

            if (err != SCI_ERR_OK)
            {
                Log::error("Failed to unexport local segment %u: %s", seginfo->id, scierrstr(err));
            }
        }
    }

    // Remove segment descriptor
    if (seginfo->segment != nullptr)
    {
        Log::debug("Removing local segment %u...", seginfo->id);
        do
        {
            SCIRemoveSegment(seginfo->segment, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        if (err != SCI_ERR_OK)
        {
            Log::error("Failed to remove local segment %u", seginfo->id);
        }
    }

    // Close SISCI descriptor
    if (seginfo->sd != nullptr)
    {
        closeDescriptor(seginfo->sd);
    }

    // Delete segment holder
    delete seginfo;
}


Segment::Segment(const SegmentInfo& info) :
    id(info.segmentId),
    size(info.size),
    adapters(info.adapters),
    impl(nullptr)
{
    sci_error_t err;

    // Create initial segment holder
    shared_ptr<SegmentImpl> impl(new SegmentImpl, &releaseSegment);
    impl->sd = nullptr;
    impl->segment = nullptr;
    impl->buffer = info.deviceBuffer;
    impl->memoryMap = nullptr;
    impl->id = id;

    // Open SISCI descriptor
    if ((err = openDescriptor(impl->sd)) != SCI_ERR_OK)
    {
        throw runtime_error(scierrstr(err));
    }

    // Create local segment
    const uint flag = info.deviceId != NO_DEVICE ? SCI_FLAG_EMPTY : 0;

    SCICreateSegment(impl->sd, &(impl->segment), id, size, NULL, NULL, flag, &err);
    if (err != SCI_ERR_OK)
    {
        Log::error("Failed to create segment %u: %s", id, scierrstr(err));
        throw runtime_error(scierrstr(err));
    }

    // Attach pre-allocated memory to segment
    if (info.deviceId != NO_DEVICE)
    {
        SCIAttachPhysicalMemory(0, info.deviceBuffer, 0, size, impl->segment, SCI_FLAG_CUDA_BUFFER, &err);
        if (err != SCI_ERR_OK)
        {
            Log::error("Failed to attach memory on segment %u: %s", id, scierrstr(err));
            throw runtime_error(scierrstr(err));
        }
    }
    else if (info.deviceBuffer != nullptr)
    {
        // TODO: SCIRegisterMemory
    }

    // Prepare segment on all adapters
    for (uint adapter: info.adapters)
    {
        SCIPrepareSegment(impl->segment, adapter, 0, &err);
        if (err != SCI_ERR_OK)
        {
            Log::error("Failed to prepare segment %u on adapter %u: %s", id, adapter, scierrstr(err));
            throw runtime_error(scierrstr(err));
        }

        impl->exports[adapter] = false;

        Log::debug("Prepared segment %u on adapter %u with IO address 0x%x", id, adapter, IOAddress(impl->segment));
    }

    Log::info("Created local segment %u with physical address 0x%x", id, physicalAddress(impl->segment));
    this->impl = impl;
}


void Segment::setAvailable(uint adapter)
{
    sci_error_t err;

    // Get local node id
    uint localId;
    SCIGetLocalNodeId(adapter, &localId, 0, &err);
    if (err != SCI_ERR_OK)
    {
        Log::warn("Failed to get local node id for adapter %u", adapter);
    }

    // Set segment available
    SCISetSegmentAvailable(impl->segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        Log::error("Failed to set segment %u available on adapter %u: %s", id, adapter, scierrstr(err));
        throw runtime_error(scierrstr(err));
    }

    // Save export state
    impl->exports[adapter] = true;

    Log::info("Segment %u is available on node %u", id, localId);
}


void* Segment::getPointer() const
{
    if (impl->buffer != nullptr)
    {
        return impl->buffer;
    }

    // Map segment into virtual address space
    sci_error_t err;
    impl->buffer = SCIMapLocalSegment(impl->segment, &impl->memoryMap, 0, size, nullptr, 0, &err);
    if (err != SCI_ERR_OK)
    {
        Log::error("Failed to map segment %u memory: %s", id, scierrstr(err));
        throw runtime_error(scierrstr(err));
    }

    return impl->buffer;
}

