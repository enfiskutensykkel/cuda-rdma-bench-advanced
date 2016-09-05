#ifndef __RDMA_BENCH_SERVER_H__
#define __RDMA_BENCH_SERVER_H__

#include "segment.h"
#include "transfer.h"


int runBenchmarkServer(SegmentList& segments);


int runBenchmarkClient(const SegmentList& segments, TransferVec& transfers);

#endif
