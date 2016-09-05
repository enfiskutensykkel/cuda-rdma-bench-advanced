#include <stdexcept>
#include <signal.h>
#include "segment.h"
#include "server.h"
#include "log.h"


static bool keepRunning = true;


static void stopServer(int)
{
    keepRunning = false;
}


int runBenchmarkServer(const SegmentList& segments)
{
    // Catch ctrl + c from terminal
    signal(SIGTERM, (sig_t) stopServer);
    signal(SIGINT, (sig_t) stopServer);

    // Run server
    Log::info("Running server...");
    while (keepRunning);

    Log::info("Shutting down server");

    return 0;
}
