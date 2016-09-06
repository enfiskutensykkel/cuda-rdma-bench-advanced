#ifndef __RDMA_BENCH_TRANSFER_H__
#define __RDMA_BENCH_TRANSFER_H__

#include <cstddef>
#include <memory>
#include <vector>
#include <sisci_types.h>
#include "segment.h"

/* Forward declaration of implementation class */
struct TransferImpl;


/* Actual transfer representation */
class Transfer
{
    public:
        Transfer(const SegmentPtr localSegment, uint remoteNodeId, uint remoteSegmentId, uint adapter);

        void addVectorEntry(size_t localOffset, size_t remoteOffset, size_t size);

        void addVectorEntry(const dis_dma_vec_t& entry);

        void setDirection(bool read);

//        void setRemoteGlobal(bool global);
//
//        void setLocalGlobal(bool global);

    private:
        mutable std::shared_ptr<TransferImpl> impl;
};


/* Convenience type for a transfer pointer */
typedef std::shared_ptr<Transfer> TransferPtr;


/* Convenience type for a collection of transfers */
typedef std::vector<TransferPtr> TransferList;


#endif
