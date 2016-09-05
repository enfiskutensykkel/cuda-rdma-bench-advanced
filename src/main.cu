#include <vector>
#include <map>
#include <cstdio>
#include <cuda.h>
#include <sisci_types.h>
#include <sisci_api.h>
#include "util.h"
#include "segment.h"
#include "transfer.h"
#include "benchmark.h"
#include "log.h"
#include "args.h"


/* Iterate over segment infos and create segments accordingly */
static void createSegments(SegmentInfoMap& segmentInfos, SegmentList& segments)
{
    for (auto segmentIt = segmentInfos.begin(); segmentIt != segmentInfos.end(); ++segmentIt)
    {
        SegmentInfo& info = segmentIt->second;
        SegmentPtr segment;

        if (info.deviceId != NO_DEVICE)
        {
            cudaError_t err = cudaSetDevice(info.deviceId);
            if (err != cudaSuccess)
            {
                Log::error("Failed to initialize GPU %d: %s", info.deviceId, cudaGetErrorString(err));
                throw std::string(cudaGetErrorString(err));
            }
                
            err = cudaMalloc(&info.deviceBuffer, info.size);
            if (err != cudaSuccess)
            {
                Log::error("Failed to allocate buffer on GPU %d: %s", info.deviceId, cudaGetErrorString(err));
                throw std::string(cudaGetErrorString(err));
            }

            Log::debug("Allocated buffer on GPU %d", info.deviceId);

            void* devicePtr = getDevicePointer(info.deviceBuffer);
            segment = Segment::createWithPhysMem(info.segmentId, devicePtr, info.size, info.adapters);
        }
        else
        {
            segment = Segment::create(info.segmentId, info.size, info.adapters);
        }

        segments.push_back(segment);
    }
}


/* Iterate over segment infos and free buffers */
static void freeBufferMemory(SegmentInfoMap& segmentInfos)
{
    for (auto segmentIt = segmentInfos.begin(); segmentIt != segmentInfos.end(); ++segmentIt)
    {
        if (segmentIt->second.deviceBuffer != nullptr)
        {
            Log::debug("Freeing buffer on GPU %d", segmentIt->second.deviceId);
            cudaFree(segmentIt->second.deviceBuffer);
        }
    }
}


int main(int argc, char** argv)
{
    SegmentList segments;
    TransferList transfers;
    SegmentInfoMap segmentInfos;
    TransferInfoList transferInfos;

    // Parse command line arguments
    try
    {
        Log::Level logLevel = Log::Level::ERROR;
        parseArguments(argc, argv, segmentInfos, transferInfos, logLevel);
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
        fprintf(stderr, "Failed to initialize SISCI API\n");
        return 2;
    }

    // Allocate buffers and create segments
    try
    {
        createSegments(segmentInfos, segments);
    }
    catch (const std::string& error)
    {
        fprintf(stderr, "Failed to create segments: %s\n", error.c_str());
        segments.clear();
        freeBufferMemory(segmentInfos);
        return 1;
    }
    catch (const std::runtime_error& error)
    {
        fprintf(stderr, "Failed to create segments: %s\n", error.what());
        segments.clear();
        freeBufferMemory(segmentInfos);
        return 2;
    }

    if (transfers.empty())
    {
        // No transfers specified, run as server
        if (runBenchmarkServer(segments) != 0)
        {
            fprintf(stderr, "SERVER FAILED\n");
        }
    }
    else
    {
        // Run benchmark client
        if (runBenchmarkClient(segments, transfers) != 0)
        {
            fprintf(stderr, "CLIENT FAILED\n");
        }
    }

    // Nuke any active SISCI handles
    transfers.clear();
    segments.clear();

    // Free any GPU buffers
    freeBufferMemory(segmentInfos);

    // Terminate SISCI API
    SCITerminate();

    return 0;
}


/* Print a list of local GPUs */
void listGPUs()
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
