#include <mutex>
#include <memory>
#include <vector>
#include <map>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <cstdint>
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
    uint                    id;             // segment identifier
    size_t                  size;           // segment size
    sci_desc_t              sd;             // SISCI descriptor
    sci_local_segment_t     segment;        // SISCI segment descriptor
    map<uint, bool>         exports;        // map over exports ordered by adapter number
    DeviceInfo*             devInfo;        // device info
    std::mutex              segmentLock;    // lock to handle concurrency
    map<uint, uint>         connections;    // current connections
};


static sci_callback_action_t 
connectEvent(SegmentImpl* info, sci_local_segment_t, sci_segment_cb_reason_t reason, uint nodeId, uint adapter, sci_error_t err)
{
    if (err != SCI_ERR_OK)
    {
        Log::warn("Error condition in segment callback: %s", scierrstr(err));
    }

    // Report state
    switch (reason)
    {
        case SCI_CB_CONNECT:
            Log::info("Remote node %u connected to segment %u on adapter %u", nodeId, info->id, adapter);
            {
                std::lock_guard<std::mutex> lock(info->segmentLock);
                info->connections[nodeId]++;
            }
            break;

        case SCI_CB_DISCONNECT:
            Log::info("Remote node %u disconnected from segment %u", nodeId, info->id);
            {
                std::lock_guard<std::mutex> lock(info->segmentLock);
                info->connections[nodeId]--;
            }
            break;

        default:
            break;
    }

    return SCI_CALLBACK_CONTINUE;
}


static void releaseSegment(SegmentImpl* seginfo)
{
    sci_error_t err;

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
    if (seginfo->devInfo != nullptr)
    {
        delete seginfo->devInfo;
    }

    delete seginfo;
}


static shared_ptr<SegmentImpl> createSegmentImpl(uint id, size_t size)
{
    shared_ptr<SegmentImpl> ptr(new SegmentImpl, &releaseSegment);
    ptr->sd = nullptr;
    ptr->segment = nullptr;
    ptr->devInfo = nullptr;
    
    ptr->id = id;
    ptr->size = size;

    // Open SISCI descriptor
    sci_error_t err;
    if ((err = openDescriptor(ptr->sd)) != SCI_ERR_OK)
    {
        throw runtime_error(scierrstr(err));
    }

    return ptr;
}


Segment::Segment(shared_ptr<SegmentImpl> impl, const std::set<uint>& adapters, uint flags) 
    :
    id(impl->id),
    size(impl->size),
    adapters(adapters),
    flags(flags),
    impl(impl)
{
    sci_error_t err;

    // Prepare segment on adapters
    for (uint adapter: adapters)
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

    Log::debug("Created local segment %u with physical address 0x%x and size %s", 
            id, physicalAddress(impl->segment), humanReadable(size).c_str());
}


SegmentPtr Segment::create(uint id, size_t size, const std::set<uint>& adapters, uint flags)
{
    sci_error_t err;

    // Create initial segment holder
    shared_ptr<SegmentImpl> impl = createSegmentImpl(id, size);

    // Create local segment
    SCICreateSegment(impl->sd, &impl->segment, id, size, (sci_cb_local_segment_t) &connectEvent, impl.get(), flags | SCI_FLAG_USE_CALLBACK, &err);
    if (err != SCI_ERR_OK)
    {
        impl->segment = nullptr;
        Log::error("Failed to create segment %u: %s", id, scierrstr(err));
        throw runtime_error(scierrstr(err));
    }

    return SegmentPtr(new Segment(impl, adapters, flags));
}


SegmentPtr Segment::createWithPhysMem(uint id, size_t size, const std::set<uint>& adapters, int gpu, void* ptr, uint flags)
{
    sci_error_t err;

    // Create initial segment holder
    shared_ptr<SegmentImpl> impl = createSegmentImpl(id, size);

    // Create local segment
    flags |= SCI_FLAG_EMPTY;
    SCICreateSegment(impl->sd, &impl->segment, id, size, (sci_cb_local_segment_t) &connectEvent, impl.get(), flags | SCI_FLAG_USE_CALLBACK, &err);
    if (err != SCI_ERR_OK)
    {
        impl->segment = nullptr;
        Log::error("Failed to create segment %u: %s", id, scierrstr(err));
        throw runtime_error(scierrstr(err));
    }

    // Attach physical memory to segment
    SCIAttachPhysicalMemory(0, ptr, 0, size, impl->segment, SCI_FLAG_CUDA_BUFFER, &err);
    if (err != SCI_ERR_OK)
    {
        Log::error("Failed to attach memory on segment %u: %s", id, scierrstr(err));
        throw runtime_error(scierrstr(err));
    }

    // Store device info
    impl->devInfo = new DeviceInfo;
    getDeviceInfo(gpu, *impl->devInfo);

    return SegmentPtr(new Segment(impl, adapters, flags));
}


void Segment::setAvailable(uint adapter)
{
    auto exported = impl->exports.find(adapter);
    if (exported == impl->exports.end())
    {
        throw std::string("Segment is not prepared on adapter ") + std::to_string(adapter);
    }

    if (!exported->second)
    {
        sci_error_t err;

        // Get local node id
        uint32_t localId = getLocalNodeId(adapter);

        // Set segment available
        SCISetSegmentAvailable(impl->segment, adapter, 0, &err);
        if (err != SCI_ERR_OK)
        {
            Log::error("Failed to set segment %u available on adapter %u: %s", id, adapter, scierrstr(err));
            throw runtime_error(scierrstr(err));
        }

        // Save export state
        exported->second = true;

        Log::info("Segment %u (%s) is available on node %u", 
                id, humanReadable(size).c_str(), localId);
    }
}


void Segment::setUnavailable(uint adapter)
{
    auto exported = impl->exports.find(adapter);
    if (exported == impl->exports.end())
    {
        throw std::string("Segment is not prepared on adapter ") + std::to_string(adapter);
    }

    if (exported->second)
    {
        sci_error_t err;

        // Set unavailable
        Log::debug("Setting segment %u unavailable on adapter %u...", id, adapter);
        do
        {
            SCISetSegmentUnavailable(impl->segment, adapter, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        if (err != SCI_ERR_OK)
        {
            Log::error("Failed to set segment %u unavailable on adapter %u: %s", id, adapter, scierrstr(err));
        }

        // Update export state
        exported->second = false;
    }
}


sci_local_segment_t Segment::getSegment() const
{
    return impl->segment;
}


void Segment::getConnections(std::vector<uint>& conns) const
{
    for (map<uint, uint>::const_iterator it = impl->connections.begin(); it != impl->connections.end(); ++it)
    {
        for (size_t i = 0; i < it->second; ++i)
        {
            conns.push_back(it->first);
        }
    }
}

