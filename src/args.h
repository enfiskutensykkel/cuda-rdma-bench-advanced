#ifndef __RDMA_BENCH_ARGS_H__
#define __RDMA_BENCH_ARGS_H__

#include <map>
#include "log.h"
#include "segment.h"
#include "transfer.h"


extern Log::Level logLevel;

extern std::string logFilename;


/* Convenience type for a segment info map ordered by segment id */
typedef std::map<uint, SegmentInfo> SegmentInfoMap;


/* Parse command line options and load settings */
void parseArguments(int argc, char** argv, SegmentInfoMap& segments, TransferVec& transfers, Log::Level& logLevel);

#endif
