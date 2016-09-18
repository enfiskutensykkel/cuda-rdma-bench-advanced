#include <mutex>
#include <condition_variable>
#include "barrier.h"

struct BarrierImpl
{
    uint numThreads;
    uint numRemaining;
    std::mutex monitorLock;
    std::condition_variable waitingQueue;
};


Barrier::Barrier(uint numThreads)
    : impl(new BarrierImpl)
{
    impl->numThreads = numThreads;
    impl->numRemaining = numThreads;
}


void Barrier::wait()
{
    std::unique_lock<std::mutex> lock(impl->monitorLock);

    // Should we wait for others?
    if (--impl->numRemaining > 0)
    {    
        // The lambda *should* handle spurious wakeups */
        auto ready = [this] { 
            return impl->numRemaining == impl->numThreads; 
        };

        // Wait for other threads
        impl->waitingQueue.wait(lock, ready);
    }
    else
    {
        // We are the last threads, lets wake everyone up
        impl->numRemaining = impl->numThreads;
        impl->waitingQueue.notify_all();
    }
}
