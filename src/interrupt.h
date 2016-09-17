#ifndef __RDMA_BENCH_INTERRUPT_H__
#define __RDMA_BENCH_INTERRUPT_H__

#include <cstddef>
#include <cstdint>
#include <memory>


/* Interrupt event description */
struct InterruptEvent
{
    uint        interruptNo;    // interrupt number of the triggered interrupt
    uint        localAdapterNo; // the local adapter the interrupt was triggered on
    uint        remoteNodeId;   // node identifier of the node that triggered the interrupt
    uint64_t    timestamp;      // timestamp of the interrupt
};


/* Convenience type for interrupt callback routines */
typedef std::function<void (const InterruptEvent& event, const void* data, size_t length)> Callback;


/* Forward declaration of implementation class */
struct InterruptImpl;


/* Local data interrupt handle */
class Interrupt
{
    public:
        const uint no;      // interrupt number
        const uint adapter; // local adapter number

        Interrupt(uint adapter, uint number, Callback callback);

    private:
        std::shared_ptr<InterruptImpl> impl;
};


/* Convenience type for interrupt pointer */
typedef std::shared_ptr<Interrupt> InterruptPtr;


#endif
