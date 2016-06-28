#ifndef __DADA_IPCIO_CUDA_H
#define __DADA_IPCIO_CUDA_H

/* ************************************************************************


   ************************************************************************ */

#include "ipcio.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! read bytes from ipcbuf directly to cuda device memory */
ssize_t ipcio_read_cuda (ipcio_t* ipc, char* ptr, size_t bytes, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
