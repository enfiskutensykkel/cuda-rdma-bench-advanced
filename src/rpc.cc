#include <functional>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <arpa/inet.h>
#include <sisci_types.h>
#include <sisci_api.h>
#include "rpc.h"
#include "interrupt.h"
#include "util.h"
#include "log.h"

using std::runtime_error;

/* Convenience type for client callbacks */
typedef std::function<void (const void*, size_t)> Callback;


/* RPC client data */
struct RpcClientImpl
{
    bool            callInProgress; // caller blocks until this is false
    std::mutex      callLock;       // avoid race conditions
    Callback        callback;       // current callback
    InterruptPtr    interrupt;      // local interrupt
};


/* Request types */
enum class Type : uint8_t
{
    GetSegmentInfo  = 0x01,         // get info about a remote segment
    GetDeviceInfo   = 0x02,         // get info about a remote GPU
    GetChecksum     = 0x03,         // calculate checksum for a remote segment
};


/* Message format */
struct Message 
{
    uint32_t        origNodeId;     // node id of the request originator
    uint32_t        origInterrupt;  // interrupt number to connect to backwards
    uint8_t         payload[1];     // data payload
};


/* Default client callback */
static void defaultCallback(const void*, size_t)
{
    Log::warn("Got unexpected interrupt");
}


/* Helper function to send a message */
static bool sendMessage(uint adapter, uint nodeId, uint remoteIntrNo, uint localIntrNo, const void* data, uint8_t length)
{
    sci_error_t status, err;
    sci_desc_t desc = nullptr;
    sci_remote_data_interrupt_t interrupt = nullptr;

    status = err = SCI_ERR_OK;

    try
    {
        // Create SISCI descriptor
        if ((err = openDescriptor(desc)) != SCI_ERR_OK)
        {
            throw err;
        }

        // Connect to remote data interrupt
        SCIConnectDataInterrupt(desc, &interrupt, nodeId, adapter, remoteIntrNo, 0, 0, &err);
        if (err != SCI_ERR_OK)
        {
            interrupt = nullptr;
            Log::error("Failed to connect to interrupt %u on node %u: %s", remoteIntrNo, nodeId, scierrstr(err));
            throw err;
        }

        // Create message and copy payload
        const size_t totalSize = sizeof(Message) + length - 1;
        Message* message = (Message*) malloc(totalSize);
        message->origNodeId = htonl(getLocalNodeId(adapter));
        message->origInterrupt = htonl(localIntrNo);
        memcpy(&message->payload[0], data, length);
                
        // Trigger remote interrupt
        SCITriggerDataInterrupt(interrupt, (void*) message, totalSize, 0, &err);
        if (err != SCI_ERR_OK)
        {
            free(message);
            Log::error("Failed to trigger remote interrupt %u on node %u: %s", remoteIntrNo, nodeId, scierrstr(err));
            throw err;
        }

        Log::debug("Message sent to interrupt %u on node %u", remoteIntrNo, nodeId);
        free(message);

    }
    catch (sci_error_t error)
    {
        Log::debug("Aborting sending data");
        status = error;
    }

    // Disconnect from remote interrupt
    if (interrupt != nullptr)
    {
        do
        {
            SCIDisconnectDataInterrupt(interrupt, 0, &err);
        }
        while (err != SCI_ERR_BUSY);

        if (err != SCI_ERR_OK)
        {
            Log::warn("Failed to disconnect from remote interrupt %u on node %u: %s", remoteIntrNo, nodeId, scierrstr(err));
        }
    }

    if (desc != nullptr)
    {
        closeDescriptor(desc);
    }

    return status == SCI_ERR_OK;
}


/* Helper function to send segment info back to requester */
static bool sendSegmentInfo(const SegmentPtr& segment, uint adapter, uint nodeId, uint remoteInterrupt, uint localInterrupt)
{
    SegmentInfo info;
    info.id = segment->id;
    info.size = segment->size;
    info.isGlobal = !!(segment->flags & SCI_FLAG_DMA_GLOBAL);
    info.isDeviceMem = !!(segment->flags & SCI_FLAG_EMPTY);

    return sendMessage(adapter, nodeId, remoteInterrupt, localInterrupt, &info, sizeof(info));
}


RpcServer::RpcServer(uint adapter, const SegmentPtr& segment, ChecksumCallback callback)
    :
    segment(segment),
    callback(callback)
{
    IntrCallback handleInterrupt = [this](const InterruptEvent& event, const void* data, size_t length)
    {
        handleRequest(event, data, length);
    };

    interrupt = InterruptPtr(new Interrupt(adapter, segment->id, handleInterrupt));

    Log::debug("RPC server for segment %u on adapter %u", segment->id, adapter);
}


/* Handle a RPC request from a client */
void RpcServer::handleRequest(const InterruptEvent& event, const void* data, size_t length)
{
    // Extract originator node id
    if (length != sizeof(Message) - sizeof(uint32_t))
    {
        Log::warn("Expected request in interrupt handler but got garbage");
        return;
    }

    uint32_t remoteInterrupt = ntohl(*((uint32_t*) data));
    
    // What kind of request is it?
    Type request = Type(*(((uint8_t*) data) + sizeof(uint32_t)));

    switch (request)
    {
        case Type::GetSegmentInfo:
            sendSegmentInfo(segment, event.localAdapterNo, event.remoteNodeId, remoteInterrupt, interrupt->no);
            break;

        // TODO: checksum and device info

        default:
            Log::warn("Got unknown request type from node %u on adapter %u: 0x%02x", 
                    event.remoteNodeId, event.localAdapterNo, request);
            break;
    }
}


RpcClient::RpcClient(uint adapter, uint id)
    :
    impl(new RpcClientImpl)
{
    impl->callInProgress = nullptr;
    impl->callback = defaultCallback;
    
    auto handleInterrupt = [this](const InterruptEvent&, const void* data, size_t length)
    {
        if (length >= sizeof(uint32_t))
        {
            impl->callback(((uint8_t*) data) + sizeof(uint32_t), length - sizeof(uint32_t));
        }

        impl->callInProgress = false;
        impl->callback = defaultCallback;
    };

    impl->interrupt = InterruptPtr(new Interrupt(adapter, id, handleInterrupt));

    Log::debug("RPC client created on adapter %u", adapter);
}


bool RpcClient::getRemoteSegmentInfo(uint nodeId, uint segmentId, SegmentInfo& info)
{
    std::lock_guard<std::mutex> lock(impl->callLock);

    // Prepare to send request
    bool success = false;
    Type request = Type::GetSegmentInfo;

    impl->callInProgress = true;
    impl->callback = [this, &success, &info](const void* data, size_t length)
    {
        if (length < sizeof(SegmentInfo))
        {
            Log::error("Expected segment info response but got garbage");
            success = false;
            return;
        }

        info = *((SegmentInfo*) data);
        success = true;
    };

    // Send request
    if (!sendMessage(impl->interrupt->adapter, nodeId, segmentId, impl->interrupt->no, &request, sizeof(request)))
    {
        impl->callback = defaultCallback;
        impl->callInProgress = false;
        throw runtime_error("Failed to send RPC request");
    }

    // Wait for response
    while (impl->callInProgress);

    return success;
}
