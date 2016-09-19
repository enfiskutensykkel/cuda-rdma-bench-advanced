#ifndef __RDMA_BENCH_RPC_H__
#define __RDMA_BENCH_RPC_H__

#include <functional>
#include <memory>
#include <cstddef>
#include <cstdint>
#include "segment.h"
#include "interrupt.h"
#include "util.h"


typedef std::function<bool (const Segment& segment, uint32_t& checksum)> ChecksumCallback;


class RpcServer
{
    public:
        /* 
         * Host an RPC server for segment on specified adapter
         * Note that it doesn't check if the segment is exported on that specific adapter
         */
        RpcServer(uint adapter, const SegmentPtr& segment, ChecksumCallback callback);

    private:
        void handleRequest(const InterruptEvent&, const void*, size_t);

        const SegmentPtr segment;
        ChecksumCallback callback;
        InterruptPtr interrupt;
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

    private:
        std::shared_ptr<RpcClientImpl> impl;
};


#endif
