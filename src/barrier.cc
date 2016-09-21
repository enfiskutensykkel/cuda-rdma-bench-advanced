#include <mutex>
#include <condition_variable>
#include "barrier.h"


struct BarrierImpl
{
    uint numThreads;
    uint numRemaining;
    uint resetCount;
    std::mutex monitorLock;
    std::condition_variable waitingQueue;
};


Barrier::Barrier(uint numThreads)
    : impl(new BarrierImpl)
{
    impl->numThreads = numThreads;
    impl->numRemaining = numThreads;
    impl->resetCount = 0;
}


void Barrier::wait()
{
    std::unique_lock<std::mutex> lock(impl->monitorLock);

    // Should we wait for others?
    if (--impl->numRemaining > 0)
    {    
        uint countCopy = impl->resetCount;
        auto ready = [this, countCopy] { 
            return impl->resetCount > countCopy;
        };

        // Wait for other threads
        impl->waitingQueue.wait(lock, ready);
    }
    else
    {
        // We are the last thread, lets wake everyone up
        impl->resetCount++;
        impl->numRemaining = impl->numThreads;
        impl->waitingQueue.notify_all();
    }
}
