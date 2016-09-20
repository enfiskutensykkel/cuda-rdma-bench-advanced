#include <string>
#include <stdexcept>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <signal.h>
#include "rpc.h"
#include "segment.h"
#include "benchmark.h"
#include "log.h"


static bool keepRunning = true;
static std::condition_variable_any doneSignal;
static std::mutex mutex;


static void stopServer(int)
{
    keepRunning = false;
    doneSignal.notify_all();
}


int runBenchmarkServer(SegmentMap& segments, ChecksumCallback callback)
{
    // Vector of information request handlers
    std::vector<RpcServer> rpcHandlers;

    try
    {
        // Loop through all segments and export on adapters
        for (auto it = segments.begin(); it != segments.end(); ++it)
        {
            SegmentPtr& segment = it->second;

            // Export segments on all adapters
            for (uint adapter: segment->adapters)
            {
                // Handle information requests for segment
                rpcHandlers.push_back(RpcServer(adapter, segment, callback));

                // Set available on adapter
                Log::debug("Exporting segment %u on adapter %u...", segment->id, adapter);
                segment->setAvailable(adapter);
            }
        }
    } 
    catch (const std::string& error)
    {
        Log::error("%s", error.c_str());
    }
    catch (const std::runtime_error& error)
    {
        Log::error("Unexpected error caused server to abort: %s", error.what());
        return 2;
    }

    // Catch ctrl + c from terminal
    auto oldTerm = signal(SIGTERM, (sig_t) stopServer);
    auto oldInt = signal(SIGINT, (sig_t) stopServer);

    // Run server
    Log::info("Running server...");
    mutex.lock();
    doneSignal.wait(mutex, []() { return !keepRunning; });
    mutex.unlock();    

    // Stop server
    Log::info("Shutting down server...");
    for (auto it = segments.begin(); it != segments.end(); ++it)
    {
        SegmentPtr segment = it->second;

        for (uint adapter: segment->adapters)
        {
            segment->setUnavailable(adapter);
        }
    }

    // Restore signal catchers
    signal(SIGTERM, oldTerm);
    signal(SIGINT, oldInt);

    return 0;
}

