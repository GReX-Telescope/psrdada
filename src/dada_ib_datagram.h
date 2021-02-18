/***************************************************************************
 *  
 *    Copyright (C) 2010 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/


#ifndef __DADA_IB_DATAGRAM_H
#define __DADA_IB_DATAGRAM_H

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

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#include "multilog.h"
#include "dada_ib.h"

#define DADA_IB_DG_IB_PORT 1
#define DADA_IB_DB_TCP_PORT 54321

#define IB_DATAGRAM 2048
#define IB_HEADER   40
#define IB_PAYLOAD  2008
#define IB_DEFAULT_PSN 12853282

typedef struct dada_ib_datagram
{
  struct ibv_context      * context;
  struct ibv_comp_channel * channel;
  struct ibv_pd           * pd;
  struct ibv_mr           * mr;
  struct ibv_cq           * cq;
  struct ibv_qp           * qp;
  struct ibv_ah           * ah;
  dada_ib_mb_t           ** bufs;
  unsigned int              nbufs;
  int                       buf_size;
  int                       queue_depth;
  struct ibv_port_attr      portinfo;
  struct ibv_device      ** dev_list;
  multilog_t              * log;
  unsigned int              verbose;
  int                       port;
  int                       ib_port;
  unsigned                  buffered_cqe;
  unsigned                  buffered_cqe_wr_id;
} dada_ib_datagram_t;

typedef struct dada_ib_datagram_dest
{
  int lid;
  int qpn;
  int psn;
  union ibv_gid gid;
}
dada_ib_datagram_dest_t;

dada_ib_datagram_t * dada_ib_dg_create (unsigned int nbufs, multilog_t * log);

struct ibv_device * dada_ib_dg_get_device (dada_ib_datagram_t * ctx, char * ib_devname);

int dada_ib_dg_init (dada_ib_datagram_t * ctx);
int dada_ib_dg_activate (dada_ib_datagram_t * ctx, dada_ib_datagram_dest_t * local, dada_ib_datagram_dest_t * remote, int sgid_idx, int sl);

dada_ib_mb_t * dada_ib_dg_alloc_buffer (dada_ib_datagram_t * ctx, uint64_t bufsz, int access_flags);

int dada_ib_dg_disconnect(dada_ib_datagram_t * ctx);

int dada_ib_dg_destroy (dada_ib_datagram_t * ctx);

dada_ib_datagram_dest_t * dada_ib_dg_get_local_port (dada_ib_datagram_t * ctx);

int dada_ib_dg_post_recvs (dada_ib_datagram_t * ctx, dada_ib_mb_t ** mb, int n_to_post);
int dada_ib_dg_post_recv (dada_ib_datagram_t * ctx, dada_ib_mb_t * mb);
int dada_ib_dg_wait_recv (dada_ib_datagram_t * ctx, dada_ib_mb_t * mb);

int dada_ib_dg_post_sends (dada_ib_datagram_t * ctx, dada_ib_mb_t ** mbs, int nmbs, int remote_qpn);
int dada_ib_dg_post_send (dada_ib_datagram_t * ctx, dada_ib_mb_t * mb, int remote_qpn);
int dada_ib_dg_wait_send (dada_ib_datagram_t * ctx, dada_ib_mb_t * mb);

int dada_ib_dg_wait_cq (dada_ib_datagram_t * ctx, dada_ib_mb_t * mb, struct ibv_comp_channel * comp_chan, struct ibv_cq * cq);

dada_ib_datagram_dest_t * dada_ib_dg_server_exch_dest(dada_ib_datagram_t * ctx, int ib_port, int port, int sl, dada_ib_datagram_dest_t * my_dest, int sgid_idx);
dada_ib_datagram_dest_t * dada_ib_dg_client_exch_dest (const char *servername, int port, const dada_ib_datagram_dest_t * my_dest);

void wire_gid_to_gid(const char *wgid, union ibv_gid *gid);
void gid_to_wire_gid(const union ibv_gid *gid, char wgid[]);

void dada_ib_dg_encode_header (char *b, uint64_t seq_no);
void dada_ib_dg_decode_header (unsigned char * b, uint64_t *seq_no);

#endif // __DADA_IB_DATAGRAM_H */
