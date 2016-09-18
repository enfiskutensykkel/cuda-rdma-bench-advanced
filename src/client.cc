#include <thread>
#include <sisci_types.h>
#include <sisci_api.h>
#include "barrier.h"
#include "segment.h"
#include "transfer.h"
#include "benchmark.h"
#include "util.h"

using std::thread;
using std::vector;


static void transferDma(Barrier barrier)
{
}


int runBenchmarkClient(const TransferList& transfers)
{
    vector<thread> transferThreads(transfers.size());
    Barrier barrier(transfers.size());

    for (TransferPtr transfer : transfers)
    {
        thread(transferDma, barrier);
    }

    return 0;
}
