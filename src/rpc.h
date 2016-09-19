#ifndef __RDMA_BENCH_RPC_H__
#define __RDMA_BENCH_RPC_H__

#include <functional>
#include <cstddef>
#include <cstdint>
#include "segment.h"
#include "interrupt.h"
#include "util.h"


typedef std::function<bool (const Segment& segment, uint32_t& checksum)> ChecksumCallback;


class RpcServer
{
    public:
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




#endif
