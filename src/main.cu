#include <functional>
#include <vector>
#include <map>
#include <memory>
#include <cstdio>
#include <cuda.h>
#include <sisci_types.h>
#include <sisci_api.h>
#include "datachannel.h"
#include "segment.h"
#include "transfer.h"
#include "benchmark.h"
#include "util.h"
#include "log.h"
#include "args.h"

typedef std::shared_ptr<void> BufferPtr;
typedef std::map<uint, BufferPtr> BufferMap;


/* Iterate over segment infos and create segments accordingly */
static void createSegments(SegmentSpecMap& segmentSpecs, SegmentMap& segments, BufferMap& buffers)
{
    for (auto segmentIt = segmentSpecs.begin(); segmentIt != segmentSpecs.end(); ++segmentIt)
    {
        SegmentSpecPtr& spec = segmentIt->second;
        SegmentPtr segment;

        if (spec->deviceId != NO_DEVICE)
        {
            const int deviceId = spec->deviceId;

            cudaError_t err = cudaSetDevice(deviceId);
            if (err != cudaSuccess)
            {
                Log::error("Failed to initialize GPU %d: %s", deviceId, cudaGetErrorString(err));
                throw std::string(cudaGetErrorString(err));
            }
                
            void* bufferPtr;
            err = cudaMalloc(&bufferPtr, spec->size);
            if (err != cudaSuccess)
            {
                Log::error("Failed to allocate buffer on GPU %d: %s", deviceId, cudaGetErrorString(err));
                throw std::string(cudaGetErrorString(err));
            }

            Log::debug("Allocated buffer on GPU %d (%p)", deviceId, bufferPtr);

            auto release = [deviceId](void* buffer) {
                Log::debug("Freeing GPU buffer on device %d (%p)", deviceId, buffer);
                cudaFree(buffer);
            };

            buffers[spec->segmentId] = BufferPtr(bufferPtr, release);

            void* devicePtr = getDevicePtr(bufferPtr);
            segment = Segment::createWithPhysMem(spec->segmentId, spec->size, spec->adapters, spec->deviceId, devicePtr, spec->flags);
        }
        else
        {
            segment = Segment::create(spec->segmentId, spec->size, spec->adapters, spec->flags);
        }

        segments[segment->id] = segment;
    }
}


/* Iterate over transfer infos and create transfers */
static void createTransfers(const DmaJobList& jobSpecs, TransferList& transfers, const SegmentMap& segments)
{
    for (const auto job: jobSpecs)
    {
        // Find corresponding local segment
        auto segment = segments.find(job->localSegmentId);
        if (segment == segments.end())
        {
            Log::error("Could not match local segment %u", job->localSegmentId);
            throw std::string("Could not find local segment ") + std::to_string(job->localSegmentId);
        }

        const SegmentPtr& localSegment = segment->second;

        // Notify user about potential error condition
        switch ((!!(localSegment->flags | SCI_FLAG_DMA_GLOBAL) << 1) | !!(job->flags | SCI_FLAG_DMA_GLOBAL))
        {
            case 0:
                break;

            case 2:
                Log::warn("Segment %u is created with SCI_FLAG_GLOBAL_DMA but transfer is non-global", localSegment->id);
                break;

            case 1:
                Log::warn("Transfer specifies SCI_FLAG_GLOBAL_DMA but local segment %u is non-global", localSegment->id);
                break;

            case 3:
                Log::debug("Using global DMA for segment %u", localSegment->id);
                break;
        }
        
        // Connect to remote end and create transfer handle
        TransferPtr transfer = Transfer::create(localSegment, job->remoteNodeId, job->remoteSegmentId, job->localAdapterNo, job->flags);

        const size_t remoteSegmentSize = transfer->remoteSegmentSize;

        // Add transfer vector entries
        for (const dis_dma_vec_t& vecEntry: job->vector)
        {
            if (vecEntry.local_offset + vecEntry.size > localSegment->size)
            {
                Log::error("Transfer size exceeds size of local segment %u", localSegment->id);
                throw std::string("Transfer size exceeds size of local segment ") + std::to_string(localSegment->id);
            }
            else if (vecEntry.remote_offset + vecEntry.size > remoteSegmentSize)
            {
                Log::error("Transfer size exceeds size of remote segment %u on node %u", job->remoteSegmentId, job->remoteNodeId);
                throw std::string("Transfer size exceeds size of remote segment ") 
                    + std::to_string(job->remoteSegmentId) + " on node " + std::to_string(job->remoteNodeId);
            }

            transfer->addVectorEntry(vecEntry);
        }

        // TODO: Handle verify

        transfers.push_back(transfer);
    }
}


int main(int argc, char** argv)
{
    SegmentSpecMap segmentSpecs;
    DmaJobList transferSpecs;

    // Parse command line arguments
    try
    {
        Log::Level logLevel = Log::Level::ERROR;
        parseArguments(argc, argv, segmentSpecs, transferSpecs, logLevel);
        Log::init(stderr, logLevel);
    }
    catch (int error)
    {
        return error;
    }
    catch (const std::string& error)
    {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    // Initialize SISCI API
    sci_error_t sciError;
    SCIInitialize(0, &sciError);
    if (sciError != SCI_ERR_OK)
    {
        Log::abort("Failed to initialize SISCI API");
        return 2;
    }

    SegmentMap segments;
    BufferMap buffers;

    // Allocate buffers and create segments
    try
    {
        createSegments(segmentSpecs, segments, buffers);
    }
    catch (const std::string& error)
    {
        Log::abort("Failed to create segments: %s", error.c_str());
        return 1;
    }
    catch (const std::runtime_error& error)
    {
        Log::abort("Failed to create segments: %s", error.what());
        return 2;
    }

    // Create transfers
    TransferList transfers;
    try
    {
        createTransfers(transferSpecs, transfers, segments);
    }
    catch (const std::string& error)
    {
        Log::abort("Failed to create transfers: %s", error.c_str());
        return 1;
    }
    catch (const std::runtime_error& error)
    {
        Log::abort("Failed to create transfers: %s", error.what());
        return 2;
    }

    if (transfers.empty())
    {
        ChecksumCallback calcChecksum = [&buffers, &segments](const Segment& segment, uint32_t& checksum) -> bool
        {
            return false;
        };

        // No transfers specified, run as server
        if (runBenchmarkServer(segments, calcChecksum) != 0)
        {
        }
    }
    else
    {
        // Run benchmark client
        if (runBenchmarkClient(transfers, stdout) != 0)
        {
        }

        // TODO: if validate, validate()
    }

    // Nuke any active SISCI handles
    transfers.clear();
    segments.clear();

    // Free any GPU buffers
    buffers.clear();

    // Terminate SISCI API
    SCITerminate();

    return 0;
}


/* Print a list of local GPUs */
void listGpus()
{
    cudaError_t err;

    // Get number of devices
    int deviceCount = 0;
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        throw std::string(cudaGetErrorString(err));
    }

    // Print header
    fprintf(stderr, "\n %2s   %-20s   %-9s   %8s   %7s   %7s   %8s   %6s   %3s   %15s\n",
            "ID", "Device name", "IO addr", "Comp mod", "Managed", "Unified", "Map hmem", "#Async", "L1", "Global mem size");
    fprintf(stderr, "-----------------------------------------------------------------------------------------------------------------\n");

    // Iterate over devices and print properties
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp prop;

        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess)
        {
            throw std::string(cudaGetErrorString(err));
        }

        fprintf(stderr, " %2d   %-20s   %02x:%02x.%-3x   %5d.%-2d   %7s   %7s   %8s   %6d   %3s   %10.02f MiB\n",
                i, prop.name, prop.pciBusID, prop.pciDeviceID, prop.pciDomainID,
                prop.major, prop.minor, 
                prop.managedMemory ? "yes" : "no", 
                prop.unifiedAddressing ? "yes" : "no",
                prop.canMapHostMemory ? "yes" : "no",
                prop.asyncEngineCount,
                prop.globalL1CacheSupported ? "yes" : "no",
                prop.totalGlobalMem / (double) (1 << 20)
               );
    }
    fprintf(stderr, "\n");
}
