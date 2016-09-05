#ifndef __RDMA_BENCH_ARGS_H__
#define __RDMA_BENCH_ARGS_H__

#include <map>
#include <list>
#include "log.h"
#include "segment.h"
#include "transfer.h"

#define NO_DEVICE   -1


/* Describe a local segment and how to create it */
struct SegmentInfo
{
    uint            segmentId;      // segment identifier
    int             deviceId;       // CUDA device the buffer is allocated on
    void*           deviceBuffer;   // buffer memory pointer
    size_t          size;           // segment size
    std::set<uint>  adapters;       // list of local adapters to export the segment on
};


/* Convenience type for a segment info map ordered by segment id */
typedef std::map<uint, SegmentInfo> SegmentInfoMap;


/* Describe a transfer task */
struct TransferInfo
{
    uint                        localSegmentId;     // local segment identifier
    uint                        remoteNodeId;       // remote node identifier
    uint                        remoteSegmentId;    // remote segment identifier
    uint                        localAdapterNo;     // local adapter number
    bool                        pullData;           // read data from remote segment
    size_t                      repeat;             // repeat transfer N times
    std::vector<dis_dma_vec_t>  vector;             // DMA transfer vector
    bool                        verify;             // verify transfer
};


/* Convenience type for a transfer info list */
typedef std::list<TransferInfo> TransferInfoList;


/* Parse command line options and load settings */
void parseArguments(int argc, char** argv, SegmentInfoMap& segments, TransferInfoList& transfers, Log::Level& logLevel);


#endif
