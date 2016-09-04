#include "segment.h"
#include "transfer.h"
#include <memory>
#include <vector>
#include <map>
#include <exception>
#include <stdexcept>
#include <string>
#include <cstring>
#include <getopt.h>
#include <errno.h>
#include <signal.h>
#include <sisci_api.h>
#include "util.h"
#include "log.h"
#include "server.h"
#include "client.h"

using std::string;


static struct option options[] = 
{
    { .name = "segment", .has_arg = true, .flag = nullptr, .val = 's' },
    { .name = "transfer", .has_arg = true, .flag = nullptr, .val = 't' },
    { .name = "verbosity", .has_arg = true, .flag = nullptr, .val = 'v' },
    { .name = "report", .has_arg = true, .flag = nullptr, .val = 'f' },
    { .name = "report-file", .has_arg = true, .flag = nullptr, .val = 'f' },
    { .name = "log", .has_arg = true, .flag = nullptr, .val = 'g' },
    { .name = "logfile", .has_arg = true, .flag = nullptr, .val = 'g' },
    { .name = "log-file", .has_arg = true, .flag = nullptr, .val = 'g' },
    { .name = "list", .has_arg = false, .flag = nullptr, .val = 'l' },
    { .name = "help", .has_arg = false, .flag = nullptr, .val = 'h' },
    { .name = nullptr, .has_arg = false, .flag = nullptr, .val = 0 }
};


/* Show program usage text */
static void giveUsage(const char* programName)
{
    fprintf(stderr, 
            "Usage: %s --segment <segment string>...\n"
            "   or: %s --segment <segment string>... --transfer <transfer string>...\n"
            "\nDescription\n"
            "    Benchmark the performance of GPU to GPU RDMA transfer.\n"
            "\nServer arguments\n"
            "  --segment    <segment string>    create a local segment\n"
            "  --export     [export string]     expose a local segment\n"
            "\nClient arguments\n"
            "  --segment    <segment string>    create a local segment\n"
            "  --transfer   <transfer string>   DMA transfer specification\n"
            "\nString format\n"
            "        key1=value1,key2,key3,key4=value4,key5=value5...\n"
            "\nSegment string\n"
            "    size=<size>                    specify size of the segment (required)\n"
            "    a=<no>                         export segment on specified adapter (required for servers)\n"
            "    ls=<id>                        local segment id [default is 0]\n"
            "    gpu=<gpu>                      specify local GPU to host buffer on [omit to host buffer in RAM]\n"
            "\nTransfer string\n"
            "    rn=<id>                        remote node id (required)\n"
            "    rs=<id>                        remote segment id [default is 0]\n"
            "    a=<no>                         local adapter for segment [default is 0]\n"
            "    ls=<id>                        local segment id [default is 0]\n"
            "    pull                           read data from remote buffer instead of writing\n"
            "    ro=<offset>                    offset into remote segment [default is 0]\n"
            "    lo=<offset>                    offset into local segment [default is 0]\n"
            "    size=<size>                    transfer size [default is the size of segment]\n"
            "    repeat=<count>                 number of times to repeat transfer [default is 1]\n"
            "    verify                         calculate checksum of transfer\n"
            "\nOther options\n"
            "  --verbosity      <level>         specify \"error\", \"warn\", \"info\" or \"debug\" log level\n"
            "  --log            <filename>      use a log file instead of stderr for logging\n"
            "  --report         <filename>      use a report file instead of stdout\n"
            "  --list                           show a list of local GPUs and quit\n"
            "  --help                           show this help and quit\n"
            "\n"
            , programName, programName);
}


static void listGpus()
{
    cudaError_t err;

    int deviceCount = 0;
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        throw string(cudaGetErrorString(err));
    }

    fprintf(stderr, "\n %2s   %-20s   %-9s   %12s   %7s   %7s   %8s   %6s   %3s   %15s\n",
            "ID", "Device name", "IO addr", "Compute mode", "Managed", "Unified", "Map hmem", "#Async", "L1", "Global mem size");
    fprintf(stderr, "---------------------------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp prop;

        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess)
        {
            throw string(cudaGetErrorString(err));
        }

        fprintf(stderr, " %2d   %-20s   %02x:%02x.%-3x   %9d.%-2d   %7s   %7s   %8s   %6d   %3s   %10.02f MiB\n",
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


/* Helper function for retrieving key--value pairs from option string */
static const char* nextToken(const char* str, string& key, string& value)
{
    bool readValue = false;

    while (true)
    {
        switch (*str)
        {
            case '=':
                if (readValue)
                {
                    throw string("Invalid string syntax");
                }
                readValue = true;
                break;

            case ',':
                return str + 1;

            case '\0':
                return str;

            default:
                if (readValue)
                {
                    value += *str;
                }
                else
                {
                    key += *str;
                }
                break;
        }

        ++str;
    }
}


/* Helper function for reading a number from a string */
static size_t parseNumber(const string& key, const string& value)
{
    if (value.empty())
    {
        throw string("String key '") + key + string("' expects a numerical value but got nothing");
    }

    size_t idx;
    size_t v = std::stoul(value, &idx, 0);

    if (idx != value.size())
    {
        throw string("String key '") + key + string("' expects a numerical value but got ``") + value + "''";
    }

    return v;
}


static void parseSegmentString(const char* segmentString, SegmentMap& segments)
{
    SegmentPtr segment(new Segment);

    // Parse segment string
    while (*segmentString != '\0')
    {
        string key, value;
        segmentString = nextToken(segmentString, key, value);

        if (key == "size" || key == "s")
        {
            segment->size = parseNumber(key, value);
        }
        else if (key == "local-segment-id" || key == "local-segment" || key == "ls")
        {
            segment->segmentId = parseNumber(key, value);
        }
        else if (key == "adapter" || key == "adapt" || key == "a")
        {
            segment->exports[parseNumber(key, value)] = false;
        }
        else if (key == "device" || key == "gpu")
        {
            segment->deviceId = parseNumber(key, value);
        }
        else if (!key.empty())
        {
            throw string("Unknown string key: ``") + key + "''";
        }
    }

    // Some sanity checking
    if (segment->size == 0)
    {
        throw string("Local segment size must be specified");
    }

    // Check if segment is already specified
    SegmentMap::iterator i = segments.lower_bound(segment->segmentId);
    if (i == segments.end() || segment->segmentId < i->first)
    {
        segments.insert(i, std::make_pair(segment->segmentId, segment));
    }
    else
    {
        throw string("Local segment ") + std::to_string(segment->segmentId) + " was already specified";
    }
}


static void parseTransferString(const char* transferString, TransferVec& transfers)
{
    TransferPtr transfer(new Transfer);
    transfer->repeat = 1;

    // Parse transfer string
    while (*transferString != '\0')
    {
        string key, value;
        transferString = nextToken(transferString, key, value);

        if (key == "remote-node-id" || key == "remote-node" || key == "rn")
        {
            transfer->remoteNodeId = parseNumber(key, value);   
        }
        else if (key == "remote-segment-id" || key == "remote-segment" || key == "rs")
        {
            transfer->remoteSegmentId = parseNumber(key, value);
        }
        else if (key == "adapter" || key == "adapt" || key == "a")
        {
            transfer->localAdapterNo = parseNumber(key, value);
        }
        else if (key == "local-segment-id" || key == "local-segment" || key == "ls")
        {
            transfer->localSegmentId = parseNumber(key, value);
        }
        else if (key == "pull" || key == "read")
        {
            transfer->pull = true;
        }
        else if (key == "remote-offset" || key == "ro")
        {
            transfer->remoteOffset = parseNumber(key, value);
        }
        else if (key == "local-offset" || key == "lo")
        {
            transfer->localOffset = parseNumber(key, value);
        }
        else if (key == "size" || key == "s")
        {
            transfer->size = parseNumber(key, value);
        }
        else if (key == "repeat" || key == "c" || key == "r" || key == "n")
        {
            transfer->repeat = parseNumber(key, value);
        }
        else if (key == "verify")
        {
            //transfer.verify = true;
        }
    }


    // Some sanity checking
    if (transfer->remoteNodeId == 0)
    {
        throw string("Remote node id must be specified for transfers");
    }

    if (transfer->repeat == 0)
    {
        throw string("Transfers can not be repeated 0 times");
    }

    transfers.push_back(transfer);
}


static Log::Level parseVerbosity(const char* argument, uint level)
{
    if (argument == nullptr)
    {
        // No argument given, increase verbosity level
        return level < Log::Level::DEBUG ? (Log::Level) (level + 1) : Log::Level::DEBUG;
    }
    else if (strcmp(argument, "error") == 0)
    {
        return Log::Level::ERROR;
    }
    else if (strcmp(argument, "warn") == 0)
    {
        return Log::Level::WARN;
    }
    else if (strcmp(argument, "info") == 0)
    {
        return Log::Level::INFO;
    }
    else if (strcmp(argument, "debug") == 0)
    {
        return Log::Level::DEBUG;
    }

    // Try to parse verbosity level as a number
    char* ptr = nullptr;
    level = strtoul(optarg, &ptr, 10);

    if (ptr == nullptr || *ptr != '\0')
    {
        throw "Unknown log level: ``" + string(optarg) + "''";
    }

    return level < Log::Level::DEBUG ? (Log::Level) (level + 1) : Log::Level::DEBUG;
}


/* Parse command line options */
static void parseArguments(int argc, char** argv, SegmentMap& segments, TransferVec& transfers, Log::Level& logLevel, string& logFile, string& reportFile)
{
    int option;
    int index;

    while ((option = getopt_long(argc, argv, "-:s:t:f:g:vlh", options, &index)) != -1)
    {
        switch (option)
        {
            case ':': // Missing value for option
                fprintf(stderr, "Argument %s requires a value\n", argv[optind - 1]);
                giveUsage(argv[0]);
                throw 1;

            case '?': // Unknown option
                fprintf(stderr, "Unknown option: ``%s''\n", argv[optind - 1]);
                giveUsage(argv[0]);
                throw 1;

            case 'h': // Show help
                giveUsage(argv[0]);
                throw 0;

            case 'l': // List GPUs
                listGpus();
                throw 0;

            case 'v': // Increase verbosity level
                logLevel = parseVerbosity(optarg, logLevel);
                break;

            case 'f': // Set report file
                reportFile = optarg;
                break;

            case 'g': // Set log file
                logFile = optarg;
                break;

            case 's': // Parse local segment options
                parseSegmentString(optarg, segments);
                break;

            case 't': // Parse transfer string
                parseTransferString(optarg, transfers);
                break;
        }
    }
}


int main(int argc, char** argv)
{
    SegmentMap segments;
    TransferVec transfers;
    FILE* logFile = stderr;
    FILE* reportFile = stdout;
    string logFilename, reportFilename;
    Log::Level logLevel = Log::Level::ERROR;

    // Parse command line arguments
    try
    {
        parseArguments(argc, argv, segments, transfers, logLevel, logFilename, reportFilename);
    }
    catch (int error)
    {
        return error;
    }
    catch (const string& error)
    {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    // Do some sanity checking
    if (segments.empty())
    {
        fprintf(stderr, "No segments specified\n");
        return 1;
    }
    else if (transfers.empty())
    {
        for (SegmentMap::const_iterator it = segments.begin(); it != segments.end(); ++it)
        {
            if (it->second->exports.empty())
            {
                fprintf(stderr, "Server mode specified, but no segment %u has no exports\n", it->first);
                return 1;
            }
        }
    }

    // Initialize SISCI API
    sci_error_t err = SCI_ERR_OK;
    SCIInitialize(0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to initialize SISCI API\n");
        return 2;
    }

    // Open log file
    if (!logFilename.empty())
    {
        logFile = fopen(logFilename.c_str(), "a");
        if (logFile == nullptr)
        {
            SCITerminate();
            fprintf(stderr, "Failed to open log file: %s\n", strerror(errno));
            return 1;
        }
    }
    Log::init(logFile, logLevel);
    Log::info("New run started...");

    if (transfers.empty())
    {
        // Catch ctrl+c from terminal
        auto stopServer = [](int) {
            stopBenchmarkServer();
        };

        signal(SIGTERM, (sig_t) stopServer);
        signal(SIGINT, (sig_t) stopServer);

        // No transfers specified, run as server
        if (runBenchmarkServer(segments) != 0)
        {
            fprintf(stderr, "SERVER FAILED\n");
        }
    }
    else
    {
        // Open report file
        if (!reportFilename.empty())
        {
            reportFile = fopen(reportFilename.c_str(), "a");
            if (reportFile == nullptr)
            {
                fprintf(stderr, "Failed to open report file: %s\n", strerror(errno));
                SCITerminate();
                fclose(logFile);
                return 1;
            }
        }

        // Run benchmark client
        if (runBenchmarkClient(segments, transfers) != 0)
        {
            fprintf(stderr, "CLIENT FAILED\n");
        }

        fflush(reportFile);
        fclose(reportFile);
    }

    // Destroy any active SISCI handles
    transfers.clear();
    segments.clear();

    // Terminate SISCI API
    SCITerminate();
    fclose(logFile);

    return 0;
}
