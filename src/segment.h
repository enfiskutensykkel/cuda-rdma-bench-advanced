#ifndef __RDMA_BENCH_SEGMENT_H__
#define __RDMA_BENCH_SEGMENT_H__

#include <cstddef>
#include <memory>
#include <map>
#include <sisci_types.h>


#define NO_DEVICE   -1


/* Convenience type for exported segments */
typedef std::map<uint, bool> ExportMap;


/* Local segment descriptor */
struct Segment
{
    uint                segmentId;  // segment identifier
    int                 deviceId;   // CUDA device the buffer is allocated on
    size_t              size;       // size of the buffer
    sci_desc_t          sci_desc;   // SISCI descriptor associated with the segment
    sci_local_segment_t segment;    // SISCI segment descriptor
    void*               buffer;     // pointer to buffer memory
    ExportMap           exports;    // map over exports

    // Constructor
    Segment();

    // Destructor
    ~Segment();

    // Create the local segment
    void createSegment(bool fill);

    // Export segment on all adapters
    void exportSegment();
};


/* Convenience type for a segment pointer */
typedef std::shared_ptr<Segment> SegmentPtr;


/* Convenience type for a segment map */
typedef std::map<uint, SegmentPtr> SegmentMap;


#endif
