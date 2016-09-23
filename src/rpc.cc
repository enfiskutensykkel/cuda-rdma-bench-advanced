#include <functional>
#include <mutex>
#include <memory>
#include <map>
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
using std::map;
using std::make_pair;
using std::pair;


/* Convenience type for client callbacks */
typedef std::function<void (const void*, size_t)> Callback;


/* RPC client data */
struct RpcClientImpl
{
    bool                                callInProgress; // caller blocks until this is false
    uint                                remoteNodeId;   // expect response from this node
    std::mutex                          callLock;       // avoid race conditions
    Callback                            callback;       // current callback
    InterruptPtr                        interrupt;      // local interrupt
    map<pair<uint, uint>, SegmentInfo>  infoCache;      // reduce number of times going to server
};


/* RPC server data */
struct RpcServerImpl
{
    SegmentPtr                          segment;        // local segment
    InterruptPtr                        interrupt;      // local interrupt
    ChecksumCallback                    callback;       // checksum calculation callback
};


/* Request types */
enum class Type : uint8_t
{
    GetSegmentInfo                      = 0x01,         // get info about a remote segment
    GetDeviceInfo                       = 0x02,         // get info about a remote GPU
    CalculateChecksum                   = 0x03,         // calculate checksum for a remote segment
};


/* Message format */
struct Message 
{
    uint32_t                            origNodeId;     // node id of the request originator
    uint32_t                            origInterrupt;  // interrupt number to connect to backwards
    uint8_t                             payload[1];     // data payload
};


struct ChecksumRequest
{
    size_t                              offset;         // offset into segment
    size_t                              size;           // size
};


struct ChecksumResponse
{
    uint32_t                            checksum;
    bool                                success;
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
            Log::error("Failed to connect to interrupt %u on node %u: %s", 
                    remoteIntrNo, nodeId, scierrstr(err));
            throw err;
        }

        // Create message and copy payload
        const size_t totalSize = sizeof(Message) + length - 1;
        Message* message = (Message*) malloc(totalSize);
        message->origNodeId = htonl(getLocalNodeId(adapter));
        message->origInterrupt = htonl(localIntrNo);
        memcpy(&message->payload[0], data, length);
                
        // Trigger remote interrupt
        SCITriggerDataInterrupt(interrupt, message, totalSize, 0, &err);
        if (err != SCI_ERR_OK)
        {
            free(message);
            Log::error("Failed to trigger remote interrupt %u on node %u: %s", 
                    remoteIntrNo, nodeId, scierrstr(err));
            throw err;
        }

        Log::debug("Message sent to interrupt %u on node %u", remoteIntrNo, nodeId);
        free(message);
    }
    catch (sci_error_t error)
    {
        status = error;
    }

    // Disconnect from remote interrupt
    if (interrupt != nullptr)
    {
        do
        {
            SCIDisconnectDataInterrupt(interrupt, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        if (err != SCI_ERR_OK)
        {
            Log::warn("Failed to disconnect from remote interrupt %u on node %u: %s", 
                    remoteIntrNo, nodeId, scierrstr(err));
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


/* Helper function to send checksum respones back to requester */
static bool handleChecksumRequest(const RpcServerImpl* impl, const InterruptEvent& event, uint32_t interruptNo, ChecksumRequest* request)
{
    uint32_t checksum = 0;
    bool status = impl->callback(*impl->segment, request->offset, request->size, checksum);

    ChecksumResponse response;
    response.checksum = checksum;
    response.success = status;

    return sendMessage(event.adapter, event.remoteNodeId, interruptNo, event.interruptNo, &response, sizeof(response));
}


/* Handle a RPC request from a client */
static void handleRequest(const RpcServerImpl* impl, const InterruptEvent& event, const void* data, size_t length)
{
    // Extract originator node id
    if (length != sizeof(Message) - sizeof(uint32_t))
    {
        Log::warn("Expected request in interrupt handler but got unknown interrupt data");
        return;
    }

    uint32_t remoteInterrupt = ntohl(*((uint32_t*) data));

    // What kind of request is it?
    Type request = Type(*(((uint8_t*) data) + sizeof(uint32_t)));

    switch (request)
    {
        case Type::GetSegmentInfo:
            sendSegmentInfo(impl->segment, event.adapter, event.remoteNodeId, remoteInterrupt, impl->interrupt->no);
            break;

        case Type::CalculateChecksum:
            if (length != 1 + sizeof(ChecksumRequest))
            {
                Log::error("Expected checksum request but got unknown data");
                break;
            }

            handleChecksumRequest(impl, event, remoteInterrupt, (ChecksumRequest*) (((uint8_t*) data) + sizeof(uint32_t) + 1));
            break;

        // TODO: Get device info

        default:
            Log::warn("Got unknown request type from node %u on adapter %u: 0x%02x", 
                    event.remoteNodeId, event.adapter, request);
            break;
    }
}


RpcServer::RpcServer(uint adapter, const SegmentPtr& segment, ChecksumCallback callback)
    :
    impl(new RpcServerImpl)
{
    auto capture = impl.get(); // Beware!!
    auto handleInterrupt = [capture](const InterruptEvent& event, const void* data, size_t length)
    {
        handleRequest(capture, event, data, length);
    };

    impl->callback = callback;
    impl->segment = segment;
    impl->interrupt.reset(new Interrupt(adapter, segment->id, handleInterrupt));

    Log::debug("Running information service for segment %u on adapter %u", segment->id, adapter);
}


RpcClient::RpcClient(uint adapter, uint id)
    :
    impl(new RpcClientImpl)
{
    auto capture = impl.get(); // Beware!!
    auto handleInterrupt = [capture](const InterruptEvent& event, const void* data, size_t length)
    {
        if (event.remoteNodeId != capture->remoteNodeId)
        {
            Log::error("Expected response from node %u but got response from %u",
                    capture->remoteNodeId, event.remoteNodeId);
        }
        else if (length < sizeof(uint32_t))
        {
            Log::error("Got garbled response from node %u",
                    event.remoteNodeId);
        }
        else
        {
            capture->callback(((uint8_t*) data) + sizeof(uint32_t), length - sizeof(uint32_t));
        }

        capture->remoteNodeId = 0;
        capture->callInProgress = false;
        capture->callback = defaultCallback;
    };

    impl->remoteNodeId = 0;
    impl->callInProgress = nullptr;
    impl->callback = defaultCallback;
    impl->interrupt.reset(new Interrupt(adapter, id, handleInterrupt));

    Log::debug("Information service client %u running on adapter %u", id, adapter);
}


bool RpcClient::calculateChecksum(uint nodeId, uint segmentId, size_t offset, size_t size, uint32_t& checksum)
{
    Log::debug("Querying node %u about checksum for segment %u...", nodeId, segmentId);


    bool success = false;
    unsigned char request[sizeof(Type) + sizeof(ChecksumRequest)];
    *((Type*) request) = Type::CalculateChecksum;
    ChecksumRequest* ptr = (ChecksumRequest*) (request + sizeof(Type));
    ptr->offset = offset;
    ptr->size = size;

    std::unique_lock<std::mutex> lock(impl->callLock);
    impl->callInProgress = true;
    impl->remoteNodeId = nodeId;

    impl->callback = [this, nodeId, segmentId, &success, &checksum](const void* data, size_t length)
    {
        if (length < sizeof(ChecksumResponse))
        {
            Log::error("Expected checksum response but got something else from node %u", nodeId);
            return;
        }

        ChecksumResponse* response = (ChecksumResponse*) data;
        checksum = response->checksum;
        success = response->success;
    };

    while (impl->callInProgress);
    lock.release();

    if (!success)
    {
        Log::warn("Querying remote node %u about checksum for segment %u failed", nodeId, segmentId);
    }

    return success;
}


bool RpcClient::getRemoteSegmentInfo(uint nodeId, uint segmentId, SegmentInfo& info)
{
    auto remoteSegmentKey = make_pair(nodeId, segmentId);

    // Try to find segment info in cache
    auto lowerBound = impl->infoCache.lower_bound(remoteSegmentKey);
    if (lowerBound != impl->infoCache.end() && lowerBound->first == remoteSegmentKey)
    {
        info = lowerBound->second;
        return true;
    }

    Log::debug("Querying node %u about segment %u...", nodeId, segmentId);
    std::unique_lock<std::mutex> lock(impl->callLock);

    // Prepare to send request
    bool success = false;
    Type request = Type::GetSegmentInfo;

    impl->callInProgress = true;
    impl->remoteNodeId = nodeId;
    impl->callback = [this, nodeId, segmentId, &success, &info](const void* data, size_t length)
    {
        if (length < sizeof(SegmentInfo))
        {
            Log::error("Expected information response but got something else from node %u", nodeId);
            return;
        }

        info = *((SegmentInfo*) data);
        if (info.id != segmentId)
        {
            Log::error("Expected information response about segment %u from node %u but got for segment %u",
                    segmentId, nodeId, info.id);
        }

        success = true;
    };

    // Send request
    if (!sendMessage(impl->interrupt->adapter, nodeId, segmentId, impl->interrupt->no, &request, sizeof(request)))
    {
        impl->callback = defaultCallback;
        impl->callInProgress = false;
        throw runtime_error("Failed to send information service request");
    }

    // Wait for response
    while (impl->callInProgress);
    lock.release();

    if (!success)
    {
        Log::warn("Querying remote node %u about segment %u failed", nodeId, segmentId);
        return false;
    }

    impl->infoCache.insert(lowerBound, make_pair(remoteSegmentKey, info));
    return true;
}

