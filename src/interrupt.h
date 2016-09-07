#ifndef __RDMA_BENCH_INTERRUPT_H__
#define __RDMA_BENCH_INTERRUPT_H__

#include <cstddef>
#include <cstdint>
#include <memory>


struct InterruptEvent
{
    uint        interruptNo;
    uint        localAdapterNo;
    uint        remoteNodeId;
    uint64_t    timestamp;
};


typedef std::function<void (const InterruptEvent& event, void* userData, const void* data, size_t length)> Callback;


struct InterruptImpl;

class Interrupt
{
    public:
        const uint no;
        const uint adapter;

        Interrupt(uint adapter, uint number, Callback callback, void* userData);

    private:
        std::shared_ptr<InterruptImpl> impl;
};

#endif
