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
    sci_dma_queue_t         dmaQueue;           // SISCI DMA queue
    vector<dis_dma_vec_t>   dmaVector;          // DIS DMA vector
    bool                    pull;               // should we read data?
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
    ptr->dmaQueue = nullptr;
    ptr->pull = false;

    // Open SISCI descriptor
    sci_error_t err;
    if ((err = openDescriptor(ptr->sd)) != SCI_ERR_OK)
    {
        throw runtime_error(scierrstr(err));
    }
    
    return ptr;
}


Transfer::Transfer(const SegmentPtr segment, uint nodeId, uint segmentId, uint adapter)
    :
    impl(createTransferImpl(segment, nodeId, segmentId, adapter))
{
    sci_error_t err;

    // Connect to remote segment
    SCIConnectSegment(impl->sd, &impl->remoteSegment, nodeId, segmentId, adapter, nullptr, nullptr, 5000, 0, &err);
    if (err != SCI_ERR_OK)
    {
        impl->remoteSegment = nullptr;
        Log::error("Failed to connect to remote segment %u on node %u: %s", segmentId, nodeId, scierrstr(err));
        throw runtime_error(scierrstr(err));
    }

    // Allocate DMA queue on adapter
    SCICreateDMAQueue(impl->sd, &impl->dmaQueue, adapter, 1, 0, &err);
    if (err != SCI_ERR_OK)
    {
        impl->dmaQueue = nullptr;
        Log::error("Failed to create DMA queue: %s", scierrstr(err));
        throw runtime_error(scierrstr(err));
    }
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


void Transfer::setDirection(bool pull)
{
    impl->pull = pull;
}
