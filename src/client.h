#ifndef __RDMA_BENCH_CLIENT_H__
#define __RDMA_BENCH_CLIENT_H__

#include "segment.h"
#include "transfer.h"


int runBenchmarkClient(const SegmentList& segments, TransferVec& transfers);

#endif
