#include <map>
#include <vector>
#include <string>
#include <cstring>
#include <getopt.h>
#include <boost/regex.hpp>
#include <sisci_types.h>
#include <sisci_api.h>
#include "log.h"
#include "util.h"
#include "transfer.h"
#include "segment.h"
#include "args.h"

using std::string;
using std::to_string;
using std::vector;


/* Convenience type for extracting key--value pairs from specification strings */
typedef std::map<std::string, std::vector<std::string> > KVMap;


/* Print a list of enabled CUDA devices */
extern void listGpus();


/* Valid command line options */
static struct option options[] = 
{
    { .name = "segment", .has_arg = true, .flag = nullptr, .val = 's' },
    { .name = "transfer", .has_arg = true, .flag = nullptr, .val = 't' },
    { .name = "verbosity", .has_arg = true, .flag = nullptr, .val = 'v' },
    { .name = "check", .has_arg = false, .flag = nullptr, .val = 'c' },
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
            "    ls=<id>                        local segment id (required)\n"
            "    size=<size>                    specify size of the segment [default is 30 MiB]\n"
            "    gpu=<gpu>                      specify local GPU to host buffer on (omit to host buffer in RAM)\n"
            "    a=<no>                         export segment on specified adapter (required for servers)\n"
            "    global                         create segment with SCI_FLAG_DMA_GLOBAL\n"
            "\nTransfer string\n"
            "    ls=<id>                        local segment id (required)\n"
            "    rn=<id>                        remote node id (required)\n"
            "    rs=<id>                        remote segment id (required)]\n"
            "    transfer=<vector entry>        specify a DMA vector entry\n"
            "    a=<no>                         local adapter for remote node [default is 0]\n"
            "    pull                           read data from remote buffer instead of writing (SCI_FLAG_DMA_READ)\n"
            "    global                         set SCI_FLAG_DMA_GLOBAL\n"
            "    sysdma                         set SCI_FLAG_DMA_SYSDMA\n"
            "\nVector entry format\n"
            "        <local offset>:<remote offset>:<size>:<repeat>\n\n"
            "    <local offset>                 offset into local segment\n"
            "    <remote offset>                offset into remote segment\n"
            "    <size>                         transfer size\n"
            "    <repeat>                       number of times to repeat vector entry\n"
            "\nOther options\n"
            "  --check                          for each remote segment, verify that data changes\n"
            "  --verbosity      <level>         specify \"error\", \"warn\", \"info\" or \"debug\" log level\n"
            "  --log            <filename>      use a log file instead of stderr for logging\n"
            "  --report         <filename>      use a report file instead of stdout\n"
            "  --list                           show a list of local GPUs and quit\n"
            "  --help                           show this help and quit\n"
            "\n"
            , programName, programName);
}


static void extractKeyValuePairs(const string& str, KVMap& kvMap)
{
    static boost::regex expr(",?(?<key>[^=,]+)(=(?<value>[^,]+))?");
    boost::smatch match;

    auto start = str.begin();
    auto end = str.end();

    while (boost::regex_search(start, end, match, expr))
    {
        kvMap[match["key"].str()].push_back(match["value"].str());
        start = match[4].second;
    }
}


static inline size_t parseNumber(const string& key, const string& value)
{
    if (value.empty())
    {
        throw "String key `" + key + "' expects a numerical argument but got nothing";
    }

    try
    {
        return std::stoul(value, nullptr, 0);
    }
    catch (const std::invalid_argument& e)
    {
        throw "String key `" + key + "' expects a numerical argument: ``" + value + "''";
    }
}


static void parseSegmentString(const string& segmentString, SegmentSpecMap& segments)
{
    SegmentSpecPtr segment(new SegmentSpec);
    segment->segmentId = 0;
    segment->deviceId = NO_DEVICE;
    segment->size = 0;
    segment->flags = 0;

    bool providedId = false;

    KVMap kvMap;
    extractKeyValuePairs(segmentString, kvMap);

    for (auto it = kvMap.begin(); it != kvMap.end(); ++it)
    {
        const string& key = it->first;
    
        if (key == "local-segment-id" || key == "local-segment" || key == "local-id" || key == "lid" || key == "id" || key == "ls")
        {
            segment->segmentId = parseNumber(key, it->second.back());
            providedId = true;
        }
        else if (key == "length" || key == "len" || key == "n" || key == "size" || key == "sz" || key == "s")
        {
            segment->size = parseNumber(key, it->second.back());
        }
        else if (key == "adapter" || key == "adapt" || key == "a" || key == "export")
        {
            for (const string& adapter : it->second)
            {
                segment->adapters.insert(parseNumber(key, adapter));
            }
        }
        else if (key == "device" || key == "dev" || key == "gpu" || key == "g")
        {
            segment->deviceId = parseNumber(key, it->second.back());
        }
        else if (key == "global")
        {
            segment->flags |= SCI_FLAG_DMA_GLOBAL;
        }
        else 
        {
            throw string("Unknown segment string key: ``") + key + "''";
        }
    }

    // Check if segment id was specified
    if (!providedId)
    {
        throw string("Local segment id must be specified");
    }

    // Check if segment size is specified
    if (segment->size == 0)
    {
        throw string("Local segment size must be specified");
    }

    // Check if segment is already specified
    SegmentSpecMap::iterator i = segments.lower_bound(segment->segmentId);
    if (i == segments.end() || segment->segmentId < i->first)
    {
        segments.insert(i, std::make_pair(segment->segmentId, segment));
    }
    else
    {
        throw string("Local segment ") + to_string(segment->segmentId) + " was already specified";
    }
}


static void parseTransferVectorEntryString(const vector<string>& entryStrings, DmaVector& vector)
{
    static boost::regex expr("^(?<loff>[^:]+):(?<roff>[^:]+):(?<size>[^:]+):(?<repeat>.+)$");
    boost::smatch match;

    for (const string& entryString : entryStrings)
    {
        dis_dma_vec_t entry;

        if (!boost::regex_match(entryString, match, expr))
        {
            throw "Invalid vector entry string: ``" + entryString + "''";
        }

        string nullstr;
        entry.local_offset = parseNumber(nullstr, match["loff"].str());
        entry.remote_offset = parseNumber(nullstr, match["roff"].str());
        entry.size = parseNumber(nullstr, match["size"].str());
        entry.flags = 0;

        size_t n = parseNumber(nullstr, match["repeat"].str());
        if (n >= DIS_DMA_MAX_VECLEN)
        {
            throw "DMA vector length can't exceed " + to_string(DIS_DMA_MAX_VECLEN) + " elements";
        }


        for (size_t i = 0; i < n; ++i)
        {
            vector.push_back(entry);
        }
    }
}


static void parseTransferString(const string& transferString, DmaJobList& transfers)
{
    DmaJobPtr transfer(new DmaJob);
    transfer->localSegmentId = 0;
    transfer->remoteNodeId = 0;
    transfer->remoteSegmentId = 0;
    transfer->localAdapterNo = 0;
    transfer->flags = 0;

    bool suppliedLocalId = false, suppliedRemoteId = false;

    KVMap kvMap;
    extractKeyValuePairs(transferString, kvMap);

    // Parse transfer string
    for (auto it = kvMap.begin(); it != kvMap.end(); ++it)
    {
        const string& key = it->first;

        if (key == "local-segment-id" || key == "local-segment" || key == "ls" || key == "lid")
        {
            transfer->localSegmentId = parseNumber(key, it->second.back());
            suppliedLocalId = true;
        }
        else if (key == "remote-node-id" || key == "remote-node" || key == "rn")
        {
            transfer->remoteNodeId = parseNumber(key, it->second.back());
        }
        else if (key == "remote-segment-id" || key == "remote-id" || key == "remote-segment" || key == "rs" || key == "rid")
        {
            transfer->remoteSegmentId = parseNumber(key, it->second.back());
            suppliedRemoteId = true;
        }
        else if (key == "adapter" || key == "adapt" || key == "a")
        {
            transfer->localAdapterNo = parseNumber(key, it->second.back());
        }
        else if (key == "pull" || key == "read")
        {
            transfer->flags |= SCI_FLAG_DMA_READ;
        }
        else if (key == "global")
        {
            transfer->flags |= SCI_FLAG_DMA_GLOBAL;
        }
        else if (key == "system-dma" || key == "system" || key == "sysdma")
        {
            transfer->flags |= SCI_FLAG_DMA_SYSDMA;
        }
        else if (key == "transfer" || key == "vector" || key == "vec" || key == "v" || key == "t")
        {
            parseTransferVectorEntryString(it->second, transfer->vector);
        }
        else if (!key.empty())
        {
            throw string("Unknown transfer string key: ``") + key + "''";
        }
    }

    // Check if local segment id was specified
    if (!suppliedLocalId)
    {
        throw string("Local segment id must be specified");
    }

    // Check if remote segment id was specified
    if (!suppliedRemoteId)
    {
        throw string("Local segment id must be specified");
    }

    // Check that remote node id was specified
    if (transfer->remoteNodeId == 0)
    {
        throw string("Remote node id must be specified for transfers");
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
    else if (strcmp(argument, "abort") == 0 || strcmp(argument, "none") == 0)
    {
        return Log::Level::ABORT;
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
void parseArguments(int argc, char** argv, SegmentSpecMap& segments, DmaJobList& transfers, Log::Level& logLevel/*, bool& validate*/)
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

            case 'c': // Verify all remote segments
                // TODO: loop through all transfers and identify remote segments, skip local global
                break;
        }
    }

    // Are there any segments specified?
    if (segments.empty())
    {
        throw string("At least one local segment must be specified");
    }
    
    // Check that segments are exported in server mode
    if (transfers.empty())
    {
        for (SegmentSpecMap::const_iterator it = segments.begin(); it != segments.end(); ++it)
        {
            if (it->second->adapters.empty())
            {
                throw string("Server mode specified but segment ") + to_string(it->first) + " has no exports";
            }
        }
    }

    // Check that DMA transfers correspond with local segments
    for (DmaJobPtr transfer : transfers)
    {
        SegmentSpecMap::const_iterator localSegment = segments.find(transfer->localSegmentId);
        if (localSegment == segments.end())
        {
            throw string("Local segment ") + to_string(transfer->localSegmentId) + " is used in transfer but is not specified";
        }

        for (const dis_dma_vec_t& vecEntry : transfer->vector)
        {
            if (vecEntry.local_offset + vecEntry.size > localSegment->second->size)
            {
                throw string("Transfer vector entry size (") 
                    + to_string(vecEntry.local_offset) +" + " + std::to_string(vecEntry.size)
                    + ") exceeds local segment size (" + std::to_string(localSegment->second->size) + ")";
            }
        }

        // Add default transfer if entry vector is empty
        if (transfer->vector.empty())
        {
            dis_dma_vec_t entry;
            entry.local_offset = 0;
            entry.remote_offset = 0;
            entry.size = localSegment->second->size;
            entry.flags = 0;

            transfer->vector.push_back(entry);
         }
    }
}
