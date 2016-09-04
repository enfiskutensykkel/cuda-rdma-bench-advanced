#include <stdexcept>
#include "segment.h"
#include "server.h"
#include "log.h"


static bool run = true;


int runBenchmarkServer(SegmentMap& segments)
{
    try
    {
        // Create local segments
        for (SegmentMap::iterator it = segments.begin(); it != segments.end(); ++it)
        {
            it->second->createSegment(true);
        }

        // Export local segments
        for (SegmentMap::iterator it = segments.begin(); it != segments.end(); ++it)
        {
            it->second->exportSegment();
        }
    }
    catch (const std::runtime_error& err)
    {
        Log::error("Aborting server");
        return -1;
    }

    // Run server
    Log::info("Running server...");
    while (run);

    Log::info("Shutting down server");

    return 0;
}


void stopBenchmarkServer()
{
    run = false;
}
