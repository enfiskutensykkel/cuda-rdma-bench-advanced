#include <string>
#include <getopt.h>
#include <cstring>
#include "log.h"
#include "transfer.h"
#include "segment.h"
#include "args.h"

using std::string;


/* Print a list of enabled CUDA devices */
extern void listGpus();


/* Valid command line options */
static struct option options[] = 
{
    { .name = "segment", .has_arg = true, .flag = nullptr, .val = 's' },
    { .name = "transfer", .has_arg = true, .flag = nullptr, .val = 't' },
    { .name = "verbosity", .has_arg = true, .flag = nullptr, .val = 'v' },
    { .name = "list", .has_arg = false, .flag = nullptr, .val = 'l' },
    { .name = "help", .has_arg = false, .flag = nullptr, .val = 'h' },
    { .name = nullptr, .has_arg = false, .flag = nullptr, .val = 0 }
};

//ls=0,rn=4,rs=0,pull,ro=0:lo=0:sz=10


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
            "    ls=<id>                        local segment id (required)\n"
            "    size=<size>                    specify size of the segment [default is 30 MiB]\n"
            "    gpu=<gpu>                      specify local GPU to host buffer on (omit to host buffer in RAM)\n"
            "    a=<no>                         export segment on specified adapter (required for servers)\n"
            "\nTransfer string\n"
            "    ls=<id>                        local segment id (required)\n"
            "    rn=<id>                        remote node id (required)\n"
            "    rs=<id>                        remote segment id (required)]\n"
            "    transfer=<vector entry>        specify a DMA vector entry (required)\n"
            "    a=<no>                         local adapter for remote node [default is 0]\n"
            "    pull                           read data from remote buffer instead of writing\n"
            "    repeat=<count>                 number of times to repeat transfer [default is 1]\n"
            "    verify                         calculate checksum of segments after transfer\n"
            "\nVector entry format\n"
            "        lo=<offset>:ro=<offset>:size=<size>\n\n"
            "    lo=<offset>                    offset into local segment\n"
            "    ro=<offset>                    offset into remote segment\n"
            "    size=<size>                    transfer size\n"
            "\nOther options\n"
            "  --verbosity      <level>         specify \"error\", \"warn\", \"info\" or \"debug\" log level\n"
            "  --log            <filename>      use a log file instead of stderr for logging\n"
            "  --report         <filename>      use a report file instead of stdout\n"
            "  --list                           show a list of local GPUs and quit\n"
            "  --help                           show this help and quit\n"
            "\n"
            , programName, programName);
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


static void parseSegmentString(const char* segmentString, SegmentSpecMap& segments)
{
    SegmentSpec segment;
    segment.segmentId = 0;
    segment.deviceId = NO_DEVICE;
    segment.size = 0;

    bool suppliedId = false;

    // Parse segment string
    while (*segmentString != '\0')
    {
        string key, value;
        segmentString = nextToken(segmentString, key, value);

        if (key == "local-segment-id" || key == "local-segment" || key == "segment-id" || key == "ls" || key == "id")
        {
            segment.segmentId = parseNumber(key, value);
            suppliedId = true;
        }
        else if (key == "length" || key == "len" || key == "l" || key == "size" || key == "sz" || key == "s")
        {
            segment.size = parseNumber(key, value);
        }
        else if (key == "adapter" || key == "adapt" || key == "a" || key == "export")
        {
            segment.adapters.insert(parseNumber(key, value));
        }
        else if (key == "device" || key == "dev" || key == "gpu" || key == "g")
        {
            segment.deviceId = parseNumber(key, value);
        }
        else if (!key.empty())
        {
            throw string("Unknown string key: ``") + key + "''";
        }
    }

    // Some sanity checking
    if (!suppliedId)
    {
        throw string("Local segment id must be specified");
    }

    if (segment.size == 0)
    {
        throw string("Local segment size must be specified");
    }

    // Check if segment is already specified
    SegmentSpecMap::iterator i = segments.lower_bound(segment.segmentId);
    if (i == segments.end() || segment.segmentId < i->first)
    {
        segments.insert(i, std::make_pair(segment.segmentId, segment));
    }
    else
    {
        throw string("Local segment ") + std::to_string(segment.segmentId) + " was already specified";
    }
}


static void parseTransferString(const char* transferString, TransferSpecList& transfers)
{
    TransferSpecPtr transfer(new TransferSpec);
    transfer->localSegmentId = 0;
    transfer->remoteNodeId = 8;
    transfer->remoteSegmentId = 0;
    transfer->localAdapterNo = 0;
    transfers.push_back(transfer);
//    TransferPtr transfer(new Transfer);
//    transfer->repeat = 1;
//
//    // Parse transfer string
//    while (*transferString != '\0')
//    {
//        string key, value;
//        transferString = nextToken(transferString, key, value);
//
//        if (key == "local-segment-id" || key == "local-segment" || key == "ls" || key == "lid")
//        {
//            transfer->localSegmentId = parseNumber(key, value);
//        }
//        else if (key == "remote-node-id" || key == "remote-node" || key == "rn")
//        {
//            transfer->remoteNodeId = parseNumber(key, value);   
//        }
//        else if (key == "remote-segment-id" || key == "remote-id" || key == "remote-segment" || key == "rs" || key == "rid")
//        {
//            transfer->remoteSegmentId = parseNumber(key, value);
//        }
//        else if (key == "adapter" || key == "adapt" || key == "a")
//        {
//            transfer->localAdapterNo = parseNumber(key, value);
//        }
//        else if (key == "pull" || key == "read")
//        {
//            transfer->pull = true;
//        }
//        else if (key == "remote-offset" || key == "ro")
//        {
//            transfer->remoteOffset = parseNumber(key, value);
//        }
//        else if (key == "local-offset" || key == "lo")
//        {
//            transfer->localOffset = parseNumber(key, value);
//        }
//        else if (key == "length" || key == "len" || key == "l" || key == "size" || key == "sz" || key == "s")
//        {
//            transfer->size = parseNumber(key, value);
//        }
//        else if (key == "repeat" || key == "c" || key == "r" || key == "n")
//        {
//            transfer->repeat = parseNumber(key, value);
//        }
//        else if (key == "verify")
//        {
//            //transfer.verify = true;
//        }
//    }
//
//    // Some sanity checking
//    if (transfer->remoteNodeId == 0)
//    {
//        throw string("Remote node id must be specified for transfers");
//    }
//
//    if (transfer->repeat == 0)
//    {
//        throw string("Transfers can not be repeated 0 times");
//    }
//
//    transfers.push_back(transfer);
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
void parseArguments(int argc, char** argv, SegmentSpecMap& segments, TransferSpecList& transfers, Log::Level& logLevel)
{
    int option;
    int index;

    // Parse arguments
    while ((option = getopt_long(argc, argv, "-:s:t:vlh", options, &index)) != -1)
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

            case 's': // Parse local segment options
                parseSegmentString(optarg, segments);
                break;

            case 't': // Parse transfer string
                parseTransferString(optarg, transfers);
                break;
        }
    }

    // Do some sanity checking
    if (segments.empty())
    {
        throw string("At least one local segment must be specified");
    }
    
    if (transfers.empty())
    {
        for (SegmentSpecMap::const_iterator it = segments.begin(); it != segments.end(); ++it)
        {
            if (it->second.adapters.empty())
            {
                throw string("Server mode specified but segment ") + std::to_string(it->first) + " has no exports";
            }
        }
    }
    
    // TODO check that transfer sizes doesn't exceed local segment size
}
