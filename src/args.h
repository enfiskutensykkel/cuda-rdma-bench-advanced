#ifndef __RDMA_BENCH_ARGS_H__
#define __RDMA_BENCH_ARGS_H__

#include <map>
#include <memory>
#include "log.h"
#include "segment.h"
#include "transfer.h"

#define NO_DEVICE   -1


/* Describe a local segment and how to create it */
struct SegmentSpec
{
    uint            segmentId;  // segment identifier
    int             deviceId;   // CUDA device the buffer is allocated on
    size_t          size;       // segment size
    std::set<uint>  adapters;   // list of local adapters to export the segment on
    uint            flags;      // SISCI flags
};


/* Convenience type for a segment specification pointer */
typedef std::shared_ptr<SegmentSpec> SegmentSpecPtr;


/* Convenience type for a segment spec map ordered by segment id */
typedef std::map<uint, SegmentSpecPtr> SegmentSpecMap;


/* Describe a DMA transfer task */
struct DmaJob
{
    uint       localSegmentId;  // local segment identifier
    uint       remoteNodeId;    // remote node identifier
    uint       remoteSegmentId; // remote segment identifier
    uint       localAdapterNo;  // local adapter number
    DmaVector  vector;          // DMA transfer vector
    uint       flags;           // SISCI flags
};


/* Convenience type for a transfer info pointer */
typedef std::shared_ptr<DmaJob> DmaJobPtr;


/* Convenience type for a transfer info list */
typedef std::vector<DmaJobPtr> DmaJobList;


/* Parse command line options and load settings */
void parseArguments(int argc, char** argv, SegmentSpecMap& segments, DmaJobList& transfers, Log::Level& logLevel, bool& verify);


#endif
