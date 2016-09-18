#ifndef __RDMA_BENCH_SERVER_H__
#define __RDMA_BENCH_SERVER_H__

#include "interrupt.h"
#include "segment.h"
#include "transfer.h"


int runBenchmarkServer(SegmentList& segments, Callback interruptHandler);


int runBenchmarkClient(const TransferList& transfers);

#endif
