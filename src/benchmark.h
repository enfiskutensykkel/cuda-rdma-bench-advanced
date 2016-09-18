#ifndef __RDMA_BENCH_SERVER_H__
#define __RDMA_BENCH_SERVER_H__

#include <cstdio>
#include "datachannel.h"
#include "segment.h"
#include "transfer.h"
#include "util.h"


int runBenchmarkServer(SegmentMap& segments, ChecksumCallback calculateChecksum);


int runBenchmarkClient(const TransferList& transfers, FILE* reportFile);

#endif
