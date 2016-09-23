#ifndef __RDMA_BENCH_SEGMENT_H__
#define __RDMA_BENCH_SEGMENT_H__

#include <cstddef>
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <sisci_types.h>

class Segment;
struct SegmentImpl;


/* Convenience type for a segment pointer */
typedef std::shared_ptr<Segment> SegmentPtr;


/* Convenience type for a collection of segments */
typedef std::map<uint, SegmentPtr> SegmentMap;


/* Local segment wrapper class */
class Segment
{
    public:
        const uint id;
        const size_t size;
        const std::set<uint> adapters;
        const uint flags;

        // Create local segment
        static SegmentPtr create(uint id, size_t size, const std::set<uint>& adapters, uint flags);

        // Create local segment and attach physical memory
        static SegmentPtr createWithPhysMem(uint id, size_t size, const std::set<uint>& adapters, int devId, void* devMem, uint flags);

        // Create local segment and register virtual memory
        static SegmentPtr createWithVirtMem(uint id, void* ptr, size_t size, const std::set<uint>& adapters); 

        // Set segment available on the specified adapter
        void setAvailable(uint adapter);

        // Set segment unavailable on the specified adapter
        void setUnavailable(uint adapter);

        // Get SISCI handle to local segment
        sci_local_segment_t getSegment() const;

        //void getConnections(std::vector<uint>& connections) const;

    private:
        std::shared_ptr<SegmentImpl> impl;
        Segment(std::shared_ptr<SegmentImpl> impl, const std::set<uint>& adapters, uint flags);
};

#endif
