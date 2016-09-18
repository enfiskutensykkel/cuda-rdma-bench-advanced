#ifndef __RDMA_BENCH_DATACHANNEL_H__
#define __RDMA_BENCH_DATACHANNEL_H__

#include <functional>
#include <mutex>
#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>
#include "segment.h"
#include "interrupt.h"
#include "util.h"


struct SegmentInfo
{
    bool        isGlobal;
    bool        isDeviceMemory;
};


typedef std::function<bool (const Segment& segment, uint32_t& checksum)> ChecksumCallback;


class DataChannelClient
{
    public:
        DataChannelClient(uint adapter, uint id);

        bool getRemoteSegmentInfo(uint nodeId, uint segmentId, SegmentInfo& info);

    private:
        std::shared_ptr<std::mutex> channelLock;
        InterruptPtr interrupt;
        IntrCallback callback;
};


class DataChannelServer
{
    public:
        DataChannelServer(const SegmentPtr& localSegment, ChecksumCallback callback);

    private:
        const SegmentPtr localSegment;
        ChecksumCallback calculateChecksum;
        std::vector<InterruptPtr> interrupts;
};

#endif
