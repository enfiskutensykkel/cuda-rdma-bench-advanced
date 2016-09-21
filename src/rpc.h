#ifndef __RDMA_BENCH_RPC_H__
#define __RDMA_BENCH_RPC_H__

#include <functional>
#include <memory>
#include <cstddef>
#include <cstdint>
#include "segment.h"
#include "interrupt.h"
#include "util.h"


typedef std::function<bool (const Segment& segment, size_t offset, size_t size, uint32_t& checksum)> ChecksumCallback;


struct RpcServerImpl;

class RpcServer
{
    public:
        /* 
         * Host an RPC server for segment on specified adapter
         * Note that it doesn't check if the segment is exported on that specific adapter
         */
        RpcServer(uint adapter, const SegmentPtr& segment, ChecksumCallback callback);

    private:
        std::shared_ptr<RpcServerImpl> impl;
};


struct SegmentInfo
{
    uint    id;
    size_t  size;
    bool    isGlobal;
    bool    isDeviceMem;
};


struct RpcClientImpl;

class RpcClient
{
    public:
        RpcClient(uint adapter, uint id);

        bool getRemoteSegmentInfo(uint nodeId, uint segmentId, SegmentInfo& info);

        bool calculateChecksum(uint nodeId, uint segmentId, size_t offset, size_t size, uint32_t& checksum);

    private:
        std::shared_ptr<RpcClientImpl> impl;
};


#endif
