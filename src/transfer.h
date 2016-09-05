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
        Transfer(const Segment& localSegment, uint remoteNodeId, uint remoteSegmentId, uint adapter, bool pull);

        void execute(FILE* reportFile) const;

    private:
        mutable std::shared_ptr<TransferImpl> impl;
};


/* Convenience type for a collection of transfers */
typedef std::vector<Transfer> TransferList;


#endif
