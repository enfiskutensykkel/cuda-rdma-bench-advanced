#include <vector>
#include <exception>
#include <stdexcept>
#include <getopt.h>
#include <sisci_types.h>
#include <sisci_api.h>
#include "task.h"
#include "util.h"

using std::vector;


static const char* logFilename = nullptr;


static struct option options[] = 
{
    // Server options
    { .name = "size", .has_arg = true, .flag = nullptr, .val = 's' },
    { .name = "port", .has_arg = true, .flag = nullptr, .val = 's' },

    // Client options
    { .name = "remote-node", .has_arg = true, .flag = nullptr, .val = 'c' },
    { .name = "remote-segment", .has_arg = true, .flag = nullptr, .val = 'c' },
    { .name = "remote-port", .has_arg = true, .flag = nullptr, .val = 'c' },

    // Local options
    { .name = "adapter", .has_arg = true, .flag = nullptr, .val = 'l' },
    { .name = "segment", .has_arg = true, .flag = nullptr, .val = 'l' },
    { .name = "device", .has_arg = true, .flag = nullptr, .val = 'l' },

    // Other options
    { .name = "si", .has_arg = false, .flag = nullptr, .val = 'i' },
    { .name = "log-file", .has_arg = true, .flag = nullptr, .val = 'w' },
    //{ .name = "report-file
    { .name = "help", .has_arg = false, .flag = nullptr, .val = 'h' },
    { .name = nullptr, .has_arg = false, .flag = nullptr, .val = 0 }
};


static void giveUsage(const char* programName)
{
}


static int parseLocalOption(int index, const char* argument)
{
    fprintf(stderr, "option=%s argument=%s\n", options[index].name, argument);
    return 0;
}


static int parseArguments(int argc, char** argv, vector<Task>& tasks)
{
    int option;
    int index;
    int error;

    while ((option = getopt_long(argc, argv, "-:vh", options, &index)) != -1)
    {
        switch (option)
        {
            case ':': // Missing value for option
                fprintf(stderr, "Option %s requires a value\n", argv[optind - 1]);
                giveUsage(argv[0]);
                return ':';

            case '?': // Unknown option
                fprintf(stderr, "Unknown option: %s\n", argv[optind - 1]);
                giveUsage(argv[0]);
                return '?';

            case 'h': // Show help
                giveUsage(argv[0]);
                return 'h';

            case 'v': // Increase verbosity level
                ++logger.level;
                break;

            case 'w': // Set log file
                logFilename = optarg;
                break;

            case 'l': // Local options
                if ((error = parseLocalOption(index, optarg)) != 0)
                {
                    return error;
                }
                break;
        }
    }

    
    return 0;
}


int main(int argc, char** argv)
{
    vector<Task> tasks;

    // Parse command line arguments
    if (parseArguments(argc, argv, tasks) != 0)
    {
    }

    // Initialize logging
    try
    {
        logger.setLogFile(logFilename);
    }
    catch (const std::runtime_error& error)
    {
        fprintf(stderr, "Failed to open log file: %s\n", error.what());
        return 1;
    }

    // Initialize SISCI API
    sci_error_t err;
    SCIInitialize(0, &err);
    if (err != SCI_ERR_OK)
    {
        logger.error("Failed to initialise SISCI: %s", scierrstr(err));
        return 2;
    }

    //std::set_unexpected(&handle_unexpected);

    //
    try
    {
    }
    catch (sci_error_t error)
    {
    }

    // Terminate SISCI API
    SCITerminate();

    return 0;
}
