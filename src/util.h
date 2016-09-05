#ifndef __RDMA_BENCH_UTIL_H__
#define __RDMA_BENCH_UTIL_H__

#include <cstdint>
#include <sisci_types.h>


/* Get current timestamp (microseconds) */
uint64_t currentTime();


/* Helper function for opening a SISCI descriptor */
sci_error_t openDescriptor(sci_desc_t& descriptor);


/* Helper function for closing a SISCI descriptor */
sci_error_t closeDescriptor(sci_desc_t descriptor);


/* Get the IO address of a local segment */
uint64_t IOAddress(sci_local_segment_t segment);


/* Get the IO address of a remote segment */
uint64_t IOAddress(sci_remote_segment_t segment);


/* Get the physicl address of a segment */
uint64_t physicalAddress(sci_local_segment_t segment);


#endif
