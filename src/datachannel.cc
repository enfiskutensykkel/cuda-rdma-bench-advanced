#include <functional>
#include <mutex>
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <arpa/inet.h>
#include <sisci_types.h>
#include <sisci_api.h>
#include "datachannel.h"
#include "interrupt.h"
#include "util.h"
#include "log.h"

using std::runtime_error;



enum class RequestType : uint8_t
{
    GET_SEGMENT_INFO = 0x00
};


struct Request
{
    uint interrupt;
    RequestType type;
};


static sci_error_t sendRequest(sci_remote_data_interrupt_t interrupt, uint adapter, const Request& request)
{
    sci_error_t err;
    
    // Prepend data message with local node identifier
    uint8_t buffer[sizeof(uint32_t) + sizeof(Request)];
    *((uint32_t*) buffer) = ntohl(getLocalNodeId(adapter));
    *((Request*) (buffer + sizeof(uint32_t))) = request;

    // Trigger remote interrupt
    SCITriggerDataInterrupt(interrupt, buffer, sizeof(buffer), 0, &err);
    if (err != SCI_ERR_OK)
    {
        Log::error("Failed to trigger remote interrupt: %s", scierrstr(err));
    }

    return err;
}


static sci_error_t sendSegmentInfo(sci_remote_data_interrupt_t interrupt, uint adapter, const SegmentInfo& info)
{
    sci_error_t err;

    // Prepend data message with local node identifier
    uint8_t buffer[sizeof(uint32_t) + sizeof(SegmentInfo)];
    *((uint32_t*) buffer) = ntohl(getLocalNodeId(adapter));
    *((SegmentInfo*) (buffer + sizeof(uint32_t))) = info;

    // Trigger remote interrupt
    SCITriggerDataInterrupt(interrupt, buffer, sizeof(buffer), 0, &err);
    if (err != SCI_ERR_OK)
    {
        Log::error("Failed to trigger remote interrupt: %s", scierrstr(err));
    }

    return err;
}


static void handleClientInterrupt(const InterruptEvent& event, const void*, size_t)
{
    Log::error("Got interrupt %u from remote node %u on adapter %u but no callback in action",
            event.interruptNo, event.remoteNodeId, event.localAdapterNo);
}


static void handleServerInterrupt(const InterruptEvent& event, const void* data, size_t length)
{
    sci_error_t err;

    if (length >= sizeof(Request))
    {
        Request* request = (Request*) data;

        sci_desc_t sd = nullptr;
        sci_remote_data_interrupt_t interrupt = nullptr;

        try
        {
            if ((err = openDescriptor(sd)) != SCI_ERR_OK)
            {
                throw err;
            }

            SCIConnectDataInterrupt(sd, &interrupt, event.remoteNodeId, event.localAdapterNo, request->interrupt, 0, 0, &err);
            if (err != SCI_ERR_OK)
            {
                interrupt = nullptr;
                Log::error("Failed to connect to remote interrupt: %s", scierrstr(err));
                throw err;
            }

            SegmentInfo info;
            info.isGlobal = true;
            info.isDeviceMemory = true;
            if ((err = sendSegmentInfo(interrupt, event.localAdapterNo, info)) != SCI_ERR_OK)
            {
                throw err;
            }

        }
        catch (sci_error_t err)
        {
        }

        if (interrupt != nullptr)
        {
            do
            {
                SCIDisconnectDataInterrupt(interrupt, 0, &err);
            }
            while (err == SCI_ERR_BUSY);
        }

        if (sd != nullptr)
        {
            closeDescriptor(sd);
        }
    }
}


DataChannelServer::DataChannelServer(const SegmentPtr& segment, ChecksumCallback cb)
    :
    localSegment(segment),
    calculateChecksum(cb)
{
    // Create interrupt on each adapter
    for (uint adapter : segment->adapters)
    {
        Log::debug("Creating data channel server for segment %u on adapter %u...", segment->id, adapter);
        interrupts.push_back(InterruptPtr(new Interrupt(adapter, segment->id, handleServerInterrupt))); // TODO pass segment to callback
    }

    Log::info("Data channel server for segment %u created", segment->id); 
}


DataChannelClient::DataChannelClient(uint adapter, uint id)
    :
    channelLock(new std::mutex)
{
    // Create callback that can be replaced
    auto handleCallback = [this](const InterruptEvent& event, const void* data, size_t length) 
    {
        callback(event, data, length);
    };

    // Set default callback
    callback = handleClientInterrupt;

    // Create local interrupt
    interrupt = InterruptPtr(new Interrupt(adapter, id, handleCallback));
}


bool DataChannelClient::getRemoteSegmentInfo(uint nodeId, uint segmentId, SegmentInfo& info)
{
    // TODO write as try catch and throw sci_error_t

    sci_error_t err;

    // Create SISCI descriptor
    sci_desc_t sd;
    if ((err = openDescriptor(sd)) != SCI_ERR_OK)
    {
        throw runtime_error(scierrstr(err));
    }

    // Connect to remote data interrupt
    Log::debug("Connecting to remote interrupt %u on node %u...", segmentId, nodeId);
    sci_remote_data_interrupt_t remoteInterrupt;
    SCIConnectDataInterrupt(sd, &remoteInterrupt, nodeId, interrupt->adapter, segmentId, 5000, 0, &err);
    if (err != SCI_ERR_OK)
    {
        Log::error("Failed to connect to remote interrupt: %s", scierrstr(err));
        closeDescriptor(sd);
        throw runtime_error(scierrstr(err));
    }

    bool response = false;

    auto callback = [&response, &info](const InterruptEvent& event, const void* data, size_t length)
    {
        if (length >= sizeof(SegmentInfo))
        {
            
        }

        Log::info("got response");
        response = true;
    };

    {
        std::lock_guard<std::mutex> lock(*channelLock);
        this->callback = callback;

        // Fire request
        Request request = {
            .interrupt = interrupt->no,
            .type = RequestType::GET_SEGMENT_INFO
        };

        if ((err = sendRequest(remoteInterrupt, interrupt->adapter, request)) != SCI_ERR_OK)
        {
            this->callback = handleClientInterrupt;
            Log::error("Failed to trigger interrupt: %s", scierrstr(err));
            SCIDisconnectDataInterrupt(remoteInterrupt, 0, &err);
            closeDescriptor(sd);
            throw runtime_error(scierrstr(err));
        }

        // Hang until remote call completes
        Log::debug("Waiting for response...");
        while (!response);

        // Restore default callback
        this->callback = handleClientInterrupt;
    }

    // Disconnect remote interrupt
    Log::debug("Disconnecting from remote interrupt %u on node %u...", segmentId, nodeId);
    do
    {
        SCIDisconnectDataInterrupt(remoteInterrupt, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    if (err != SCI_ERR_OK)
    {
        Log::error("Failed to disconnect remote interrupt: %s", scierrstr(err));
    }

    closeDescriptor(sd);
    return true;
}
