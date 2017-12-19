#ifndef __DADA_IPCBUF_CUDA_H
#define __DADA_IPCBUF_CUDA_H

/* ************************************************************************


   ************************************************************************ */

#include "ipcbuf.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! zero the next buffer in a IPCBUF using CUDA */
ssize_t ipcbuf_zero_next_block_cuda (ipcbuf_t* id, char * dev_ptr, size_t dev_bytes, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
