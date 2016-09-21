#ifndef __RDMA_BENCH_SERVER_H__
#define __RDMA_BENCH_SERVER_H__

#include <cstdio>
#include "rpc.h"
#include "segment.h"
#include "transfer.h"
#include "util.h"


int runBenchmarkServer(SegmentMap& segments, ChecksumCallback calculateChecksum);


void runBenchmarkClient(const TransferList& transfers, FILE* reportFile);


int validateTransfers(const TransferList& transfers, ChecksumCallback calculateChecksum, FILE* reportFile);


#endif
