#include <functional>
#include <memory>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <arpa/inet.h>
#include <sisci_types.h>
#include <sisci_api.h>
#include "interrupt.h"
#include "log.h"
#include "util.h"

using std::runtime_error;
using std::shared_ptr;


/* Definition of interrupt implementation class */
struct InterruptImpl
{
    uint                        intno;      // interrupt number
    uint                        adapter;    // local adapter number
    IntrCallback                callback;   // user-supplied callback function
    void*                       userData;   // user-supplied callback data
    sci_desc_t                  sd;         // SISCI descriptor
    sci_local_data_interrupt_t  intr;       // SISCI data interrupt descriptor
};


static sci_callback_action_t
interruptCallback(InterruptImpl* intrData, sci_local_data_interrupt_t, void* data, uint length, sci_error_t err)
{
    if (err != SCI_ERR_OK)
    {
        Log::warn("Error condition in interrupt callback: %s", scierrstr(err));
    }

    if (length > sizeof(uint32_t))
    {
        uint32_t remoteNodeId = ntohl(*((uint32_t*) data));
        Log::debug("Got interrupt from remote node %u", remoteNodeId);
    
        const void* dataPtr = (const void*) (((uint8_t*) data) + sizeof(uint32_t));

        const InterruptEvent event = {
            .interruptNo = intrData->intno,
            .localAdapterNo = intrData->adapter,
            .remoteNodeId = remoteNodeId,
            .timestamp = currentTime()
        };

        //intrData->callback(event, intrData->userData, dataPtr, (size_t) length - sizeof(uint32_t));
        intrData->callback(event, dataPtr, (size_t) length - sizeof(uint32_t));
    }

    return SCI_CALLBACK_CONTINUE;
}


static void releaseInterrupt(InterruptImpl* ptr)
{
    // Release data interrupt handle
    if (ptr->intr != nullptr)
    {
        sci_error_t err;

        Log::debug("Removing data interrupt %u on adapter %u...", ptr->intno, ptr->adapter);
        do
        {
            SCIRemoveDataInterrupt(ptr->intr, 0, &err);
        } 
        while (err == SCI_ERR_BUSY);

        if (err != SCI_ERR_OK)
        {
            Log::error("Failed to remove data interrupt %u on adapter %u", ptr->intno, ptr->adapter);
        }
    }

    // Close SISCI descriptor
    if (ptr->sd != nullptr)
    {
        closeDescriptor(ptr->sd);
    }

    delete ptr;
}


static shared_ptr<InterruptImpl> createInterruptImpl(uint adapterNo, uint interruptNo, IntrCallback cb)
{
    shared_ptr<InterruptImpl> ptr(new InterruptImpl, &releaseInterrupt);
    ptr->intno = interruptNo;
    ptr->adapter = adapterNo;
    ptr->callback = cb;
    ptr->userData = nullptr;
    ptr->sd = nullptr;
    ptr->intr = nullptr;
    return ptr;
}


Interrupt::Interrupt(uint adapter, uint interruptNo, IntrCallback cb)
    :
    no(interruptNo),
    adapter(adapter),
    impl(createInterruptImpl(adapter, interruptNo, cb))
{
    sci_error_t err;

    // Open SISCI descriptor
    if  ((err = openDescriptor(impl->sd)) != SCI_ERR_OK)
    {
        throw runtime_error(scierrstr(err));
    }

    // Create data interrupt
    SCICreateDataInterrupt(impl->sd, &impl->intr, adapter, &impl->intno, (sci_cb_data_interrupt_t) &interruptCallback, impl.get(), SCI_FLAG_USE_CALLBACK | SCI_FLAG_FIXED_INTNO, &err);
    if (err != SCI_ERR_OK)
    {
        impl->intr = nullptr;
        Log::error("Failed to create interrupt on adapter %u: %s", adapter, scierrstr(err));
        throw runtime_error(scierrstr(err));
    }

    Log::info("Created interrupt %u on adapter %u", interruptNo, adapter);
}
