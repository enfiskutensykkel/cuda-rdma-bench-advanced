#ifndef __RDMA_BENCH_TRANSFER_H__
#define __RDMA_BENCH_TRANSFER_H__

#include <cstddef>
#include <memory>
#include <vector>
#include <sisci_types.h>
#include "segment.h"


/* Convenience type for DMA vectors */
typedef std::vector<dis_dma_vec_t> DmaVector;


/* Forward declaration of implementation class */
struct TransferImpl;


/* Actual transfer representation */
class Transfer
{
    public:
        Transfer(const SegmentPtr localSegment, uint remoteNodeId, uint remoteSegmentId, uint adapter, uint flags);

        void addVectorEntry(size_t localOffset, size_t remoteOffset, size_t size);

        void addVectorEntry(const dis_dma_vec_t& entry);

        size_t getLocalSegmentSize() const;

        sci_local_segment_t getLocalSegment() const;

        uint getLocalSegmentId() const;

        size_t getRemoteSegmentSize() const;

        sci_remote_segment_t getRemoteSegment() const;

        uint getRemoteSegmentId() const;

        uint getRemoteNodeId() const;

    private:
        std::shared_ptr<TransferImpl> impl;
};


/* Convenience type for a transfer pointer */
typedef std::shared_ptr<Transfer> TransferPtr;


/* Convenience type for a collection of transfers */
typedef std::vector<TransferPtr> TransferList;


#endif
