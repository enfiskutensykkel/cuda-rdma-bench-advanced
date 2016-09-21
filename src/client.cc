#include <thread>
#include <vector>
#include <map>
#include <string>
#include <limits>
#include <stdexcept>
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


/* Convenience type for information service clients */
typedef map<pair<uint, uint>, RpcClient> ServiceClientMap;


/* Create information service clients */
static void createInfoServiceClients(const TransferList& transfers, ServiceClientMap& clients)
{
    for (const TransferPtr& transfer : transfers)
    {
        auto key = make_pair(transfer->adapter, transfer->localSegmentId);

        auto lowerBound = clients.lower_bound(key);
        if (lowerBound == clients.end() || lowerBound->first != key)
        {
            RpcClient client(transfer->adapter, transfer->localSegmentId);
            clients.insert(lowerBound, make_pair(key, client));
        }
    }
}


/* Write transfer results to file in a neat table */
static void writeTransferResults(FILE* reportFile, const TransferList& transfers, uint64_t* completionTimes, sci_error_t* transferStatus, ServiceClientMap& clients)
{
    const char* remoteSegmentKind[transfers.size()];

    // Figure out the segment types of the remote segments
    for (size_t i = 0; i < transfers.size(); ++i)
    {
        const TransferPtr& transfer = transfers[i];
        remoteSegmentKind[i] = "???";

        // Find the information service client
        auto serviceClient = clients.find(make_pair(transfer->adapter, transfer->localSegmentId));
        if (serviceClient == clients.end())
        {
            Log::warn("Couldn't find information service client for segment %u on adapter %u",
                    transfer->localSegmentId, transfer->adapter);
            continue;
        }

        // Look up remote segment
        SegmentInfo segmentInfo;
        if (serviceClient->second.getRemoteSegmentInfo(transfer->remoteNodeId, transfer->remoteSegmentId, segmentInfo))
        {
            remoteSegmentKind[i] = segmentInfo.isDeviceMem ? "gpu" : "ram";
        }

    }

    // Write results headline
    fprintf(reportFile, " %3s   %-4s   %-3s   %-3s   %-3s   %-13s   %-13s   %-16s   %4s\n",
            "#", "node", "src", "dst", "read", "transfer size", "transfer time", "throughput", "note");
    for (size_t i = 0; i < 89; ++i)
    {
        fputc('=', reportFile);
    }
    fputc('\n', reportFile);

    // Write results for each transfer
    for (size_t i = 0; i < transfers.size(); ++i)
    {
        const TransferPtr& transfer = transfers[i];

        // Calculate total transfer size
        size_t transferSize = 0;
        for (const dis_dma_vec_t& entry : transfer->getDmaVector())
        {
            transferSize += entry.size;
        }

        // Which direction did we transfer?
        const char* sourceType = "???";
        const char* destType = "???";
        if (!!(transfer->flags & SCI_FLAG_DMA_READ))
        {
            sourceType = remoteSegmentKind[i];
            destType = !!(transfer->getLocalSegmentPtr()->flags & SCI_FLAG_EMPTY) ? "gpu" : "ram";
        }
        else
        {
            sourceType = !!(transfer->getLocalSegmentPtr()->flags & SCI_FLAG_EMPTY) ? "gpu" : "ram";
            destType = remoteSegmentKind[i];
        }

        // Write report line
        fprintf(reportFile, " %3zu   %4u   %3s   %3s   %4s   %13s   %10lu µs   %16s   %4s\n", 
            i,
            transfer->remoteNodeId,
            sourceType,
            destType,
            !!(transfer->flags & SCI_FLAG_DMA_READ) ? "yes" : "no",
            humanReadable(transferSize).c_str(),
            transferStatus[i] == SCI_ERR_OK ? 0 : completionTimes[i],
            humanReadable(transferSize, completionTimes[i]).c_str(),
            transferStatus[i] != SCI_ERR_OK ? "FAIL" : "OK"
        );
    }
}


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

    Log::debug("Ready to perform DMA transfer (rn: %u, rs: %u, ls: %u, sysdma: %s, global: %s, read: %s)",
        transfer->remoteNodeId, transfer->remoteSegmentId, transfer->localSegmentId,
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


int validateTransfers(const TransferList& transfers, ChecksumCallback calculateCheksum, FILE* reportFile)
{
    // Create info service clients
    ServiceClientMap serviceClients;
    try
    {
        createInfoServiceClients(transfers, serviceClients);
    }
    catch (const std::runtime_error& error)
    {
        Log::error("%s", error.what());
        return 2;
    }

    // Create benchmark data
    uint64_t times[transfers.size()];
    sci_error_t status[transfers.size()];
    
    // Execute transfers
    Log::info("Validating transfers...");
    for (size_t idx = 0; idx < transfers.size(); ++idx)
    {
        Barrier fakeBarrier(1);
        times[idx] = std::numeric_limits<uint64_t>::max();
        status[idx] = SCI_ERR_OK;

        // Get checksum of remote segment
        
        // Calculate checksum of local segment

        // Execute transfer
        transferDma(fakeBarrier, transfers[idx], &times[idx], &status[idx]);

        // Get checksum of remote segment again
        
        // Calculate checksum of local segment
        
        
    }
    Log::info("Done validating transfers");

    // Write results
    writeTransferResults(reportFile, transfers, times, status, serviceClients);

    return 0;
}


void runBenchmarkClient(const TransferList& transfers, FILE* reportFile)
{
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
    ServiceClientMap clients;
    try
    {
        createInfoServiceClients(transfers, clients);
    }
    catch (const std::runtime_error& error)
    {
        Log::error("Failed to create information service clients: %s", error.what());
    }

    writeTransferResults(reportFile, transfers, times, status, clients);
}

