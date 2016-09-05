#ifndef __RDMA_BENCH_SEGMENT_H__
#define __RDMA_BENCH_SEGMENT_H__

#include <cstddef>
#include <memory>
#include <vector>
#include <set>


#define NO_DEVICE   -1


/* Describe a local segment and how to create it */
struct SegmentInfo
{
    uint            segmentId;      // segment identifier
    int             deviceId;       // CUDA device the buffer is allocated on
    void*           deviceBuffer;   // buffer memory pointer
    size_t          size;           // segment size
    std::set<uint>  adapters;       // list of local adapters to export the segment on
};


/* Forward declaration of implementation class */
struct SegmentImpl;


/* Actual local segment representation */
class Segment
{
    public:
        const uint id;
        const size_t size;
        const std::set<uint> adapters;

        // Create the local segment
        Segment(const SegmentInfo& segmentInfo);

        // Set segment available on the specified adapter
        void setAvailable(uint adapter);

        // Get a pointer to segment memory
        void* getPointer() const;

    private:
        mutable std::shared_ptr<SegmentImpl> impl;
};


/* Convenience type for a segment pointer */
typedef std::shared_ptr<Segment> SegmentPtr;


/* Convenience type for a collection of segments */
typedef std::vector<Segment> SegmentList;

#endif
