#ifndef __RDMA_BENCH_TRANSFER_H__
#define __RDMA_BENCH_TRANSFER_H__

#include <cstddef>
#include <memory>
#include <vector>
#include <sisci_types.h>


/* Convenience type for DIS DMA vectors */
typedef std::shared_ptr<dis_dma_vec_t> DmaVecPtr;


/* DMA transfer descriptor */
struct Transfer
{
    uint                    remoteNodeId;       // remote node identifier
    uint                    remoteSegmentId;    // remote segment identifier
    uint                    localAdapterNo;     // local adapter number
    uint                    localSegmentId;     // local segment identifier
    size_t                  size;               // total transfer size
    size_t                  localOffset;        // offset into local segment
    size_t                  remoteOffset;       // offset into remote segment
    bool                    pull;               // read data from remote segment
    size_t                  repeat;             // repeat transfer N times
    sci_desc_t              sci_desc;           // SISCI descriptor
    sci_remote_segment_t    remoteSegment;      // SISCI remote segment descriptor
    sci_dma_queue_t         dmaQueue;           // SISCI DMA queue
    DmaVecPtr               dmaVector;          // DIS DMA vector pointer
    size_t                  vectorLength;       // length of vector

    // Constructor
    Transfer();

    // Destructor
    ~Transfer();
};


/* Convenience type for a transfer pointer */
typedef std::shared_ptr<Transfer> TransferPtr;


/* Convenience type for a vector of transfers */
typedef std::vector<TransferPtr> TransferVec;


#endif
