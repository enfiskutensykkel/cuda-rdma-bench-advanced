#include <string>
#include <stdexcept>
#include <signal.h>
#include "benchmark.h"
#include "segment.h"
#include "log.h"


static bool keepRunning = true;


static void stopServer(int)
{
    keepRunning = false;
}


int runBenchmarkServer(SegmentList& segments)
{
    try
    {
        for (SegmentPtr segment: segments)
        {
            // Export segments on all adapters
            for (uint adapter: segment->adapters)
            {
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
    signal(SIGTERM, (sig_t) stopServer);
    signal(SIGINT, (sig_t) stopServer);

    // Run server
    Log::info("Running server...");
    while (keepRunning);

    Log::info("Shutting down server");

    return 0;
}
