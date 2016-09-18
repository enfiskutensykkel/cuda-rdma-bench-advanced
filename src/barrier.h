#ifndef __RDMA_BENCH_THREAD_BARRIER_H__
#define __RDMA_BENCH_THREAD_BARRIER_H__

#include <memory>

struct BarrierImpl;


/* Generic thread barrier implemented as a monitor */
class Barrier
{
    public:
        /* Create a barrier for N threads */
        explicit Barrier(uint numThreads);

        /* Wait until all threads reach this execution point */
        void wait();

    private:
        std::shared_ptr<BarrierImpl> impl;
};

#endif
