#ifndef __TASK_H__
#define __TASK_H__

#include <cstddef>

#define NO_DEVICE -1


struct Segment
{
    uint                    adapterNo;
    uint                    segmentId;
    int                     deviceId;
    size_t                  size;
    sci_desc_t              sci_desc;
    sci_local_segment_t     segment;
};


struct Transfer
{
    uint                    remoteNodeId;
    uint                    remoteSegmentId;
    uint                    localAdapterNo;
    uint                    localSegmentId;
    size_t                  size;
    size_t                  localOffset;
    size_t                  remoteOffset;
    size_t                  repeat;
    bool                    verify;
    bool                    pull;
    bool                    global;
    sci_desc_t              sci_desc;
    sci_remote_segment_t    remoteSegment;
};

#endif
