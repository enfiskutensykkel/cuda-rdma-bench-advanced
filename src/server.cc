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
        for (Segment& segment: segments)
        {
            // Export segments on all adapters
            for (uint adapter: segment.adapters)
            {
                Log::debug("Exporting segment %u on adapter %u...", segment.id, adapter);
                segment.setAvailable(adapter);
            }

            // TODO: create interrupts, one per segment id
        }
    } 
    catch (const std::runtime_error& error)
    {
        Log::error("Unexpected error caused server to abort: %s", error.what());
        return -1;
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
