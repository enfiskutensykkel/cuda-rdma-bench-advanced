#include <thread>
#include <cstdint>
#include <cstdio>
#include <sisci_types.h>
#include <sisci_api.h>
#include "barrier.h"
#include "datachannel.h"
#include "segment.h"
#include "transfer.h"
#include "benchmark.h"
#include "util.h"
#include "log.h"

using std::thread;
using std::vector;


static void transferDma(Barrier barrier, TransferPtr transfer, uint64_t* time, sci_error_t* err)
{
    // Prepare DMA transfer vector
    const DmaVector& dmaVector = transfer->getDmaVector();
    dis_dma_vec_t vector[dmaVector.size()];

    size_t length = 0;
    size_t totalSize = 0;
    for (const dis_dma_vec_t& entry : dmaVector)
    {
        vector[length++] = entry;
        totalSize += entry.size;
    }

    Log::debug("Ready to perform DMA transfer (local segment: %u, remote node: %u, remote segment: %u)",
            transfer->localSegmentId, transfer->remoteNodeId, transfer->remoteSegmentId);

    sci_dma_queue_t queue = transfer->getDmaQueue();
    sci_local_segment_t lseg = transfer->getLocalSegment();
    sci_remote_segment_t rseg = transfer->getRemoteSegment();
    *time = 0;
    *err = SCI_ERR_OK;

    // Wait for all threads to reach this execution point
    barrier.wait();

    // Execute transfer
    uint64_t timeBefore = currentTime();
    SCIStartDmaTransferVec(queue, lseg, rseg, length, vector, nullptr, nullptr, SCI_FLAG_DMA_WAIT | transfer->flags, err);
    uint64_t timeAfter = currentTime();

    *time = timeAfter - timeBefore;

    // Check errors
    if (*err != SCI_ERR_OK)
    {
        Log::error("DMA transfer failed: %s", scierrstr(*err));
    }
}


int runBenchmarkClient(const TransferList& transfers, FILE* reportFile)
{
    // Create interrupts and connect to remote interrupts for data passing

    Barrier barrier(transfers.size() + 1);
    vector<thread> transferThreads;
    sci_error_t errors[transfers.size()];
    uint64_t times[transfers.size()];
    size_t threadIdx = 0;

    // Create transfer threads and start transfers
    for (TransferPtr transfer : transfers)
    {
        transferThreads.push_back(thread(transferDma, barrier, transfer, &times[threadIdx], &errors[threadIdx]));

        // TODO: debug
        DataChannelClient channel(transfer->adapter, 1000);
        SegmentInfo info;
        channel.getRemoteSegmentInfo(transfer->remoteNodeId, transfer->remoteSegmentId, info);
    }

    // Start all transfers
    Log::info("Preparing to start transfers...");
    barrier.wait();
    Log::info("Executing transfers...");

    // Wait for all transfers to complete
    for (auto& transferThread : transferThreads)
    {
        transferThread.join();
    }
    Log::info("All transfers done");

    // Write benchmark summary

    return 0;
}
