#include <thread>
#include <vector>
#include <map>
#include <cstdint>
#include <cstdio>
#include <sisci_types.h>
#include <sisci_api.h>
#include "barrier.h"
#include "rpc.h"
#include "segment.h"
#include "transfer.h"
#include "benchmark.h"
#include "util.h"
#include "log.h"

using std::thread;
using std::vector;
using std::map;
using std::pair;
using std::make_pair;


/* Convenience type for mapping segment info */
typedef map<pair<uint, uint>, SegmentInfo> SegmentInfoMap;


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


static void writeBandwidthReport(FILE* reportFile, const TransferPtr& transfer, uint64_t time, sci_error_t status)
{
    
}


int runBenchmarkClient(const TransferList& transfers, FILE* reportFile)
{
    // Get data about remote segments
    SegmentInfoMap segmentInfoMap;
    for (TransferPtr transfer : transfers)
    {
        auto key = make_pair(transfer->remoteNodeId, transfer->remoteSegmentId);

        SegmentInfoMap::iterator lowerBound = segmentInfoMap.lower_bound(key);
        if (lowerBound == segmentInfoMap.end() || lowerBound->first != key)
        {
            SegmentInfo info;

            RpcClient client(transfer->adapter, transfer->localSegmentId);
            if (client.getRemoteSegmentInfo(transfer->remoteNodeId, transfer->remoteSegmentId, info))
            {
                segmentInfoMap.insert(lowerBound, make_pair(key, info));
            }
        }
    }

    // Create thread barrier to synchronize transfer threads
    const size_t numTransfers = transfers.size();
    Barrier barrier(numTransfers + 1);
    
    // Create transfer thread data
    thread threads[numTransfers];
    sci_error_t errors[numTransfers];
    uint64_t times[numTransfers];

    // Create transfer threads and start transfers
    for (size_t threadIdx = 0; threadIdx < numTransfers; ++threadIdx)
    {
        const TransferPtr& transfer = transfers[threadIdx];

        errors[threadIdx] = SCI_ERR_OK;
        times[threadIdx] = 0;

        threads[threadIdx] = thread(transferDma, barrier, transfer, &times[threadIdx], &errors[threadIdx]);
    }

    // Start all transfers
    Log::info("Preparing to start transfers...");
    barrier.wait();
    Log::info("Executing transfers...");

    // Wait for all transfers to complete
    for (size_t threadIdx = 0; threadIdx < numTransfers; ++threadIdx)
    {
        threads[threadIdx].join();
    }
    Log::info("All transfers done");

    // Write benchmark summary
    fprintf(reportFile, "================  SUMMARY  ================\n");
    fprintf(reportFile, "%3s   %-4s   %-9s   %-9s   %-9s   %-9s   %-9s   %-9s   %-9s   %-9s\n",
            "#", "RN", "RS", "RS global", "RS device", "RS size", "LS", "LS global", "LS device", "LS size"
    );
    for (size_t idx = 0; idx < numTransfers; ++idx)
    {
        const TransferPtr& transfer = transfers[idx];
        const char* rsGlobal = "N/A";
        const char* rsDevice = "N/A";

        auto segmentInfo = segmentInfoMap.find(make_pair(transfer->remoteNodeId, transfer->remoteSegmentId));
        if (segmentInfo != segmentInfoMap.end())
        {
            const SegmentInfo& info = segmentInfo->second;
            rsGlobal = info.isGlobal ? "yes" : "no";
            rsDevice = info.isDeviceMem ? "yes" : "no";
        }

        const SegmentPtr localSegment = transfer->getLocalSegmentPtr();
        const char* lsGlobal = !!(localSegment->flags | SCI_FLAG_DMA_GLOBAL) ? "yes" : "no";
        const char* lsDevice = !!(localSegment->flags | SCI_FLAG_EMPTY) ? "yes" : "no";

        fprintf(reportFile, "%3zu   %4x   %-9x   %-9s   %-9s   %9s   %-9x   %-9s   %-9s   %9s\n",
            idx,
            transfer->remoteNodeId,
            transfer->remoteSegmentId,
            rsGlobal,
            rsDevice,
            humanReadable(transfer->remoteSegmentSize).c_str(),
            transfer->localSegmentId,
            lsGlobal,
            lsDevice,
            humanReadable(transfer->localSegmentSize).c_str()
        );
    }

    // Write benchmark report
    fprintf(reportFile, "================ BANDWIDTH ================\n");

    return 0;
}
