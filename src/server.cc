#include <string>
#include <stdexcept>
#include <signal.h>
#include <vector>
#include "rpc.h"
#include "segment.h"
#include "benchmark.h"
#include "log.h"


static bool keepRunning = true;


static void stopServer(int)
{
    keepRunning = false;
}


int runBenchmarkServer(SegmentMap& segments, ChecksumCallback callback)
{
    std::vector<RpcServer> rpcHandlers;

    try
    {
        for (auto it = segments.begin(); it != segments.end(); ++it)
        {
            SegmentPtr& segment = it->second;

            // Export segments on all adapters
            for (uint adapter: segment->adapters)
            {
                // Handle RPC for segment
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
    while (keepRunning);

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
