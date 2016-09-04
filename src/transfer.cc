#include <cstddef>
#include <memory>
#include <map>
#include <sisci_api.h>
#include <sisci_types.h>
#include "transfer.h"


Transfer::Transfer() :
    remoteNodeId(0),
    remoteSegmentId(0),
    localAdapterNo(0),
    localSegmentId(0),
    size(0),
    localOffset(0),
    remoteOffset(0),
    pull(false),
    repeat(0),
    sci_desc(nullptr),
    remoteSegment(nullptr),
    dmaQueue(nullptr),
    dmaVector(nullptr),
    vectorLength(0)
{
}


Transfer::~Transfer()
{
}
