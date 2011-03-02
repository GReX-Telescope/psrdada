/***************************************************************************
 *  
 *    Copyright (C) 2009 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/


#ifndef __CASPSR_RDMA_H
#define __CASPSR_RDMA_H

#define CASPSR_RDMA_NBUFS   8
#define CASPSR_RDMA_PKTSZ   8192
#define CASPSR_RDMA_BUFSZ   81920
#define CASPSR_RDMA_CM_PORT 20079
#define CASPSR_RDMA_NXFERS  10

#define CASPSR_IB_MESSAGE_SIZE 8192
#define CASPSR_IB_RX_DEPTH 10
#define CASPSR_IB_USE_CQ 1 

enum {
  RESOLVE_TIMEOUT_MS      = 5000,
};

#endif // __CASPSR_RDMA_H
