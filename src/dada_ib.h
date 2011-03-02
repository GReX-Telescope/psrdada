/***************************************************************************
 *  
 *    Copyright (C) 2010 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/


#ifndef __DADA_IB_H
#define __DADA_IB_H

/*
 * DADA Infiniband library functions
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <inttypes.h>

#include <infiniband/arch.h>
#include <rdma/rdma_cma.h>

#include "multilog.h"

#define DADA_IB_DEFAULT_CHUNK_SIZE 65536

#ifndef RESOLVE_TIMEOUT_MS
#define RESOLVE_TIMEOUT_MS 5000
#endif

#define KERNEL_BUFFER_SIZE_MAX     67108864
#define KERNEL_BUFFER_SIZE_DEFAULT 131071
#define STATS_INIT = {0, 0, 0, 0}

typedef struct dada_ib_mb
{
  void          * buffer;
  uint64_t        size;
  struct ibv_mr * mr;
  unsigned        wr_id;
} dada_ib_mb_t;

typedef struct dada_ib_shm_block 
{
  //uint64_t  buf_va;
  uintptr_t  buf_va;
  uint32_t  buf_rkey;
  uint32_t  buf_lkey;
} dada_ib_shm_block_t;

typedef struct dada_ib_rdma_cm 
{
  struct rdma_event_channel     * cm_channel;
  struct rdma_cm_id             * cm_id;
  struct rdma_cm_event          * event;
  struct rdma_conn_param          conn_param;
  struct ibv_comp_channel       * comp_chan;
  struct ibv_pd                 * pd;
  struct ibv_cq                 * cq;
  unsigned                        port;
  uint64_t                        rbuf;
  unsigned                        verbose;
  uint64_t                        nbufs;
  char **                         db_buffers;
  uint64_t                        bufs_size;
  dada_ib_mb_t                 ** bufs;
  uint64_t                        header_size;
  char                          * header;
  dada_ib_mb_t                  * header_mb;
  unsigned                        depth;
  size_t                          sync_size;
  dada_ib_mb_t                  * sync_to;
  uint64_t                      * sync_to_val;
  dada_ib_mb_t                  * sync_from;
  uint64_t                      * sync_from_val;
  dada_ib_shm_block_t           * local_blocks;
  int                             cm_connected;
  int                             ib_connected;
  multilog_t                    * log;
} dada_ib_cm_t;

struct rdma_cm_id * dada_rdma_resolve_addr (const char * host, const char * port);

dada_ib_cm_t * dada_ib_create_cm (unsigned nbufs, multilog_t * log);

int dada_ib_listen_cm (dada_ib_cm_t * ctx, int port);

int dada_ib_connect_cm (dada_ib_cm_t * ctx, const char *host, unsigned port);

int dada_ib_create_verbs (dada_ib_cm_t * ctx, unsigned depth);

dada_ib_mb_t * dada_ib_reg_buffer (dada_ib_cm_t * ctx, void * buffer,
                                   uint64_t bufsz, int access_flags);

int dada_ib_reg_buffers(dada_ib_cm_t * ctx, char ** buffers, uint64_t bufsz, 
                        int access_flags);

int dada_ib_create_qp (dada_ib_cm_t * ctx, unsigned send_depth, unsigned recv_depth);

int dada_ib_accept (dada_ib_cm_t * ctx);

int dada_ib_connect(dada_ib_cm_t * ctx);

int dada_ib_post_send (dada_ib_cm_t * ctx, dada_ib_mb_t *mb);

int dada_ib_post_recv (dada_ib_cm_t * ctx, dada_ib_mb_t *mb);

int dada_ib_wait_recv (dada_ib_cm_t * ctx, dada_ib_mb_t *mb);

int dada_ib_post_sends (dada_ib_cm_t * ctx, void * buffer, uint64_t bytes, 
                        uint64_t chunk_size, uint32_t lkey, uint32_t rkey, 
                        uintptr_t raddr);

int dada_ib_post_sends_gap (dada_ib_cm_t * ctx, void * buffer, uint64_t bytes, 
                            uint64_t chunk_size, uint32_t lkey, uint32_t rkey, 
                            uintptr_t raddr, uint64_t roffset, uint64_t rgap);

int dada_ib_client_destroy (dada_ib_cm_t * ctx);

int dada_ib_destroy (dada_ib_cm_t * ctx);

int dada_ib_disconnect (dada_ib_cm_t * ctx);


#endif // __DADA_IB_H */
