#include <thread>
#include <vector>
#include <map>
#include <string>
#include <limits>
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

    Log::debug("Ready to perform DMA transfer (ls: %u, rn: %u, rs: %u, sysdma: %s, global: %s, read: %s)",
        transfer->localSegmentId, transfer->remoteNodeId, transfer->remoteSegmentId,
        !!(transfer->flags & SCI_FLAG_DMA_SYSDMA) ? "yes" : "no",
        !!(transfer->flags & SCI_FLAG_DMA_GLOBAL) ? "yes" : "no",
        !!(transfer->flags & SCI_FLAG_DMA_READ) ? "yes" : "no"
    );

    sci_dma_queue_t queue = transfer->getDmaQueue();
    sci_local_segment_t lseg = transfer->getLocalSegment();
    sci_remote_segment_t rseg = transfer->getRemoteSegment();

    // Wait for all threads to reach this execution point
    barrier.wait();

    // Execute transfer
    uint64_t timeBefore = currentTime();
    SCIStartDmaTransferVec(queue, lseg, rseg, length, vector, nullptr, nullptr, SCI_FLAG_DMA_WAIT | transfer->flags, err);
    uint64_t timeAfter = currentTime();

    // Wait for all threads
    barrier.wait();

    if (*err == SCI_ERR_OK)
    {
        *time = timeAfter - timeBefore;
    }
}


static void writeTransferResults(FILE* reportFile, const TransferPtr& transfer, const SegmentInfo* info, uint64_t time, sci_error_t status, size_t num)
{
    size_t transferSize = 0;

    const DmaVector& vector = transfer->getDmaVector();
    for (const dis_dma_vec_t& entry : vector)
    {
        transferSize += entry.size;
    }

    const char* sourceType = !!(transfer->getLocalSegmentPtr()->flags & SCI_FLAG_EMPTY) ? "gpu" : "ram";

    const char* destinationType = "???";
    if (info != nullptr)
    {
        destinationType = info->isDeviceMem ? "gpu" : "ram";
    }

    fprintf(reportFile, " %3zu   %4u   %3s   %3s   %13s   %10lu µs   %16s   %4s\n", 
        num, 
        transfer->remoteNodeId,
        sourceType,
        destinationType,
        humanReadable(transferSize).c_str(),
        status != SCI_ERR_OK ? 0 : time,
        humanReadable(transferSize, time).c_str(),
        status != SCI_ERR_OK ? "FAIL" : "OK"
    );
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
    sci_error_t status[numTransfers];
    uint64_t times[numTransfers];

    // Create transfer threads and start transfers
    for (size_t threadIdx = 0; threadIdx < numTransfers; ++threadIdx)
    {
        const TransferPtr& transfer = transfers[threadIdx];

        status[threadIdx] = SCI_ERR_OK;
        times[threadIdx] = std::numeric_limits<uint64_t>::max();

        threads[threadIdx] = thread(transferDma, barrier, transfer, &times[threadIdx], &status[threadIdx]);
    }

    // Start all transfers
    Log::info("Preparing to start transfers...");
    barrier.wait();
    Log::info("Executing transfers...");
    barrier.wait();

    // Wait for all transfers to complete
    for (size_t threadIdx = 0; threadIdx < numTransfers; ++threadIdx)
    {
        threads[threadIdx].join();

        if (status[threadIdx] != SCI_ERR_OK)
        {
            Log::error("Transfer %zu failed: %s", threadIdx, scierrstr(status[threadIdx]));
        }
    }
    Log::info("All transfers done");

    // Write benchmark report
    fprintf(reportFile, " %3s   %-4s   %-3s   %-3s   %-13s   %-13s   %-16s   %4s\n",
            "#", "node", "src", "dst", "transfer size", "transfer time", "throughput", "note");
    for (size_t i = 0; i < 82; ++i)
    {
        fputc('=', reportFile);
    }
    fputc('\n', reportFile);

    for (size_t idx = 0; idx < numTransfers; ++idx)
    {
        const TransferPtr& transfer = transfers[idx];

        const SegmentInfo* info = nullptr;
        const auto segmentInfoIt = segmentInfoMap.find(make_pair(transfer->remoteNodeId, transfer->remoteSegmentId));
        if (segmentInfoIt != segmentInfoMap.end())
        {
            info = &segmentInfoIt->second;
        }

        writeTransferResults(reportFile, transfer, info, times[idx], status[idx], idx);
    }

    return 0;
}

