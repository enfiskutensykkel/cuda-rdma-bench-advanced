#include <stdexcept>
#include <vector>
#include <memory>
#include <cstddef>
#include <sisci_api.h>
#include <sisci_types.h>
#include "segment.h"
#include "transfer.h"
#include "util.h"
#include "log.h"

using std::runtime_error;
using std::shared_ptr;
using std::vector;


/* Definition of implementation class */
struct TransferImpl
{
    sci_desc_t              sd;                 // SISCI descriptor
    SegmentPtr              localSegment;       // local segment pointer
    uint                    remoteNodeId;       // remote node identifier
    uint                    remoteSegmentId;    // remote segment identifier
    uint                    adapter;            // local adapter number
    sci_remote_segment_t    remoteSegment;      // SISCI remote segment descriptor
    size_t                  remoteSegmentSize;  // size of remote segment
    sci_dma_queue_t         dmaQueue;           // SISCI DMA queue
    vector<dis_dma_vec_t>   dmaVector;          // DIS DMA vector
    uint                    flags;              // additional SISCI flags to SCIStartDmaTransferVec
};


static void releaseResources(TransferImpl* handle)
{
    sci_error_t err;

    // Remove DMA queue
    if (handle->dmaQueue != nullptr)
    {
        SCIRemoveDMAQueue(handle->dmaQueue, 0, &err);
        if (err != SCI_ERR_OK)
        {
            Log::error("Failed to remove DMA queue: %s", scierrstr(err));
        }
        Log::debug("Removed DMA queue");
    }

    // Disconnect from remote segment
    if (handle->remoteSegment != nullptr)
    {
        Log::debug("Disconnecting remote segment %u on node %u...", handle->remoteSegmentId, handle->remoteNodeId);
        do
        {
            SCIDisconnectSegment(handle->remoteSegment, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        if (err != SCI_ERR_OK)
        {
            Log::error("Failed to disconnect remote segment %u: %s", handle->remoteSegmentId, scierrstr(err));
        }
    }


    // Close SISCI descriptor
    if (handle->sd != nullptr)
    {
        closeDescriptor(handle->sd);
    }

    delete handle;
}


static shared_ptr<TransferImpl> createTransferImpl(SegmentPtr localSegment, uint remoteNodeId, uint remoteSegmentId, uint adapter)
{
    shared_ptr<TransferImpl> ptr(new TransferImpl, &releaseResources);
    ptr->sd = nullptr;
    ptr->localSegment = localSegment;
    ptr->remoteNodeId = remoteNodeId;
    ptr->remoteSegmentId = remoteSegmentId;
    ptr->adapter = adapter;
    ptr->remoteSegment = nullptr;
    ptr->remoteSegmentSize = 0;
    ptr->dmaQueue = nullptr;
    ptr->flags = 0;

    // Open SISCI descriptor
    sci_error_t err;
    if ((err = openDescriptor(ptr->sd)) != SCI_ERR_OK)
    {
        throw runtime_error(scierrstr(err));
    }
    
    return ptr;
}


Transfer::Transfer(shared_ptr<TransferImpl> impl)
    :
    adapter(impl->adapter),
    remoteNodeId(impl->remoteNodeId),
    remoteSegmentId(impl->remoteSegmentId),
    remoteSegmentSize(impl->remoteSegmentSize),
    localSegmentId(impl->localSegment->id),
    localSegmentSize(impl->localSegment->size),
    flags(impl->flags),
    impl(impl)
{
    sci_error_t err;

    // Allocate DMA queue on adapter
    SCICreateDMAQueue(impl->sd, &impl->dmaQueue, adapter, 1, 0, &err);
    if (err != SCI_ERR_OK)
    {
        impl->dmaQueue = nullptr;
        Log::error("Failed to create DMA queue: %s", scierrstr(err));
        throw runtime_error(scierrstr(err));
    }
}


TransferPtr Transfer::create(const SegmentPtr segment, uint nodeId, uint segmentId, uint adapter, uint flags)
{
    sci_error_t err;

    // Create transfer implementation
    shared_ptr<TransferImpl> impl(createTransferImpl(segment, nodeId, segmentId, adapter));

    // Connect to remote segment
    Log::debug("Connecting to remote segment %u on node %u...", segmentId, nodeId);
    SCIConnectSegment(impl->sd, &impl->remoteSegment, nodeId, segmentId, adapter, nullptr, nullptr, 5000, 0, &err);
    if (err != SCI_ERR_OK)
    {
        impl->remoteSegment = nullptr;
        Log::error("Failed to connect to remote segment %u on node %u: %s", segmentId, nodeId, scierrstr(err));
        throw runtime_error(scierrstr(err));
    }

    // Get remote segment size
    impl->remoteSegmentSize = SCIGetRemoteSegmentSize(impl->remoteSegment);

    // Set SISCI flags
    impl->flags = flags;

    return TransferPtr(new Transfer(impl));
}


void Transfer::addVectorEntry(size_t localOffset, size_t remoteOffset, size_t size)
{
    dis_dma_vec_t entry;
    entry.size = size;
    entry.local_offset = localOffset;
    entry.remote_offset = remoteOffset;
    entry.flags = 0;

    addVectorEntry(entry);
}


void Transfer::addVectorEntry(const dis_dma_vec_t& entry)
{
    impl->dmaVector.push_back(entry);
}


sci_remote_segment_t Transfer::getRemoteSegment() const
{
    return impl->remoteSegment;
}


sci_local_segment_t Transfer::getLocalSegment() const
{
    return impl->localSegment->getSegment();
}


const DmaVector& Transfer::getDmaVector() const
{
    return impl->dmaVector;
}


sci_dma_queue_t Transfer::getDmaQueue() const
{
    return impl->dmaQueue;
}


const SegmentPtr Transfer::getLocalSegmentPtr() const
{
    return impl->localSegment;
}
