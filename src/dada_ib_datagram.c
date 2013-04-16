/***************************************************************************
 *  
 *    Copyright (C) 2012 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <malloc.h>

#include "dada_ib_datagram.h"

dada_ib_datagram_t * dada_ib_dg_create (unsigned int nbufs, multilog_t * log)
{

  dada_ib_datagram_t * ctx = (dada_ib_datagram_t *) malloc(sizeof(dada_ib_datagram_t));
  if (!ctx)
  {
    multilog(log, LOG_ERR, "dada_ib_create_cm: malloc failed\n");
    return 0;
  }

  ctx->context = 0;
  ctx->channel = 0;
  ctx->pd = 0;
  ctx->cq = 0;
  ctx->qp = 0;
  ctx->ah = 0;
  ctx->buf_size = IB_DATAGRAM;  // 40 bytes + payload
  ctx->log = log;
  ctx->nbufs = nbufs;
  ctx->verbose = 0;
  //ctx->dest_lid = 0;
  //ctx->dest_qpn = 0;
  ctx->port = 0;

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "dada_ib_dg_create allocating %d buffers\n", nbufs);
  ctx->bufs = (dada_ib_mb_t **) malloc(sizeof(dada_ib_mb_t *) * nbufs);
  multilog(ctx->log, LOG_INFO, "dada_ib_dg_create ctx->bufs=%p\n", ctx->bufs);
  if (!ctx->bufs)
  {
    multilog(log, LOG_ERR, "dada_ib_create_cm: could not allocate memory for ctx->bufs\n");
    return 0;
  }

  return ctx;
}

struct ibv_device * dada_ib_dg_get_device (dada_ib_datagram_t * ctx, char * ib_devname)
{
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "dada_ib_dg_get_device()\n"); 

  multilog(ctx->log, LOG_INFO, "get_device: ibv_get_device_list(NULL)\n");
  ctx->dev_list = ibv_get_device_list(NULL);
  if (!ctx->dev_list) 
  {
    multilog(ctx->log, LOG_ERR, "dada_ib_dg_get_device: ibv_get_device_list failed: %s\n",
             strerror(errno));
    return 0;
  }

  struct ibv_device * ib_dev = 0;

  if (!ib_devname)
  {
    multilog(ctx->log, LOG_INFO, "get_device: setting to first value\n");
    ib_dev = *(ctx->dev_list);
    if (!ib_dev) 
    {
      multilog(ctx->log, LOG_ERR, "dada_ib_dg_get_device: no IB devices found\n");
      return 0;
    }
  }
  else
  {
    int i;
    for (i = 0; ctx->dev_list[i]; ++i)
      if (!strcmp(ibv_get_device_name(ctx->dev_list[i]), ib_devname))
        break;
    ib_dev = ctx->dev_list[i];
    if (!ib_dev) 
    {
      multilog (ctx->log, LOG_ERR, "dada_ib_dg_get_device: IB device %s not found\n", ib_devname);
      return 0;
    }
  }
  return ib_dev;
}

/*
 *  
 */
int dada_ib_dg_init (dada_ib_datagram_t * ctx)
{
  assert(ctx);
  multilog_t * log = ctx->log;

  if (!ctx->queue_depth)
  {
    multilog (ctx->log, LOG_ERR, "dada_ib_dg_init: no queue depth specified\n");
    return -1;
  }

  struct ibv_device * ib_dev = dada_ib_dg_get_device (ctx, 0);
  if (!ib_dev)
  {
    multilog (ctx->log, LOG_ERR, "dada_ib_dg_init: dada_ib_dg_get_device failed\n");
    return -1;
  }

  // open the RDMA device context
  ctx->context = ibv_open_device(ib_dev);
  if (!ctx->context) 
  {
    multilog (ctx->log, LOG_ERR, "dada_ib_dg_init: ibv_open_device failed\n");
    return -1;
  }

  // create the completion channel
  ctx->channel = ibv_create_comp_channel (ctx->context);
  if (!ctx->channel)
  {
    multilog (ctx->log, LOG_ERR, "dada_ib_dg_init: couldn't create completion channel\n");
    return -1;
  }

  // allocate the protection domain
  ctx->pd = ibv_alloc_pd(ctx->context);
  if (!ctx->pd)
  {
    multilog (ctx->log, LOG_ERR, "dada_ib_dg_init: couldn't allocate PD\n");
    return -1;
  }

  // register memory regions for "sockets" 
  unsigned ibuf;
  uint64_t bufsz = IB_DATAGRAM;
  int flags = IBV_ACCESS_LOCAL_WRITE;
  for (ibuf=0; ibuf < ctx->nbufs; ibuf++)
  {
    multilog (ctx->log, LOG_INFO, "init: dada_ib_dg_alloc_buffer (ctx=%p, bufsz=%"PRIu64", flags=%d)\n", ctx, bufsz, flags);
    ctx->bufs[ibuf] = dada_ib_dg_alloc_buffer (ctx, bufsz, flags);
    ctx->bufs[ibuf]->wr_id = 100 + ibuf;
    multilog (ctx->log, LOG_INFO, "init: ctx->bufs[%d]=%p [MR] wr_id=%d\n", ibuf, ctx->bufs[ibuf], ctx->bufs[ibuf]->wr_id);
  }

  // create the completion queue
  multilog (ctx->log, LOG_INFO, "init: ibv_create_cq (%p, %d, NULL, %p, 0)\n", ctx->context, ctx->queue_depth + 1, ctx->channel);
  ctx->cq = ibv_create_cq(ctx->context, ctx->queue_depth + 1, NULL, ctx->channel, 0);
  if (!ctx->cq) 
  {
    multilog (ctx->log, LOG_ERR, "dada_ib_dg_init: couldn't create CQ\n");
    return -1;
  }

  // enable the notification mechanism for the CQ
  multilog (ctx->log, LOG_INFO, "init: ibv_req_notify_cq()\n");
  if (ibv_req_notify_cq(ctx->cq, 0))
  {
    multilog(log, LOG_ERR, "init: ibv_req_notify_cq ctx->cq failed\n");
    return -1;
  }

  // create the QP
  {
    struct ibv_qp_init_attr attr = {
        .send_cq = ctx->cq,
        .recv_cq = ctx->cq,
        .cap     = {
            .max_send_wr  = ctx->queue_depth,
            .max_recv_wr  = ctx->queue_depth,
            .max_send_sge = 1,
            .max_recv_sge = 1
        },
        .qp_type = IBV_QPT_UD,
    };

    multilog (ctx->log, LOG_INFO, "dada_ib_dg_init: ibv_create_qp(%p, %p)\n", ctx->pd, &attr);
    ctx->qp = ibv_create_qp(ctx->pd, &attr);
    if (!ctx->qp)  
    {
      multilog (ctx->log, LOG_ERR, "dada_ib_dg_init: couldn't create QP\n");
      return -1;
    }
  }

  // modify the QP to init
  {
    struct ibv_qp_attr attr = {
        .qp_state        = IBV_QPS_INIT,
        .pkey_index      = 0,
        .port_num        = ctx->ib_port,
        .qkey            = 0x11111111
    };

    multilog (ctx->log, LOG_INFO, "dada_ib_dg_init: ibv_modify_qp(%p, %p) port=%d\n", ctx->qp, &attr, ctx->ib_port);
    if (ibv_modify_qp(ctx->qp, &attr,
              IBV_QP_STATE              |
              IBV_QP_PKEY_INDEX         |
              IBV_QP_PORT               |
              IBV_QP_QKEY)) 
    {
      multilog (ctx->log, LOG_ERR, "dada_ib_dg_init: failed to modify QP to INIT: %s\n", strerror(errno));
      return -1;
    }
  }

  return 0;
}

int dada_ib_dg_activate (dada_ib_datagram_t * ctx, dada_ib_datagram_dest_t * local, dada_ib_datagram_dest_t * remote, int sgid_idx, int sl)
{
  // create the address handler
  multilog (ctx->log, LOG_INFO, "dada_ib_dg_activate: configuring AH remote->lid=%"PRIu16", sl=%d, port_num=%d\n", remote->lid, sl, ctx->ib_port);
  struct ibv_ah_attr ah_attr = {
    .is_global     = 0,
    .dlid          = remote->lid,
    .sl            = sl,
    .src_path_bits = 0,
    .port_num      = ctx->ib_port
  };

  struct ibv_qp_attr attr = {
    .qp_state       = IBV_QPS_RTR
  };

  if (ibv_modify_qp (ctx->qp, &attr, IBV_QP_STATE)) 
  {
    multilog (ctx->log, LOG_ERR, "dada_ib_dg_activate: failed to modify QP to RTR: %s\n", strerror(errno));
    return -1;
  }

  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn   = local->psn;

  if (ibv_modify_qp (ctx->qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) 
  {
    multilog (ctx->log, LOG_ERR, "dada_ib_dg_activate: failed to modify QP to RTS: %s\n", strerror(errno));
    return -1;
  }

/*
  if (remote->gid.global.interface_id)
  {
    multilog (ctx->log, LOG_INFO, "dada_ib_dg_activate: remote->gid.global.interface_id == TRUE\n");
    ah_attr.is_global = 1;
    ah_attr.grh.hop_limit = 1;
    ah_attr.grh.dgid = remote->gid;
    ah_attr.grh.sgid_index = sgid_idx;
  }
*/

  ctx->ah = ibv_create_ah (ctx->pd, &ah_attr);
  if (!ctx->ah)
  {
    multilog (ctx->log, LOG_ERR, "dada_ib_dg_activate: failed to create AH: %s\n", strerror(errno));
    return -1;
  }

  multilog (ctx->log, LOG_INFO, "dada_ib_dg_activate: initialized\n");

  return 0;
}

// allocate a buffer of the specified size registering the MR in the PD
dada_ib_mb_t * dada_ib_dg_alloc_buffer (dada_ib_datagram_t * ctx, uint64_t bufsz, int access_flags)
{

  assert(ctx);
  multilog_t * log = ctx->log;

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "dada_ib_dg_alloc_buffer()\n");

  dada_ib_mb_t * mr = (dada_ib_mb_t *) malloc(sizeof(dada_ib_mb_t));
  if (!mr)
  {
    multilog (log, LOG_ERR, "dada_ib_dg_alloc_buffer: failed to allocate memory for dada_ib_datagram_t\n");
    return 0;
  }

  size_t page_size = sysconf(_SC_PAGESIZE);
  size_t buffer_size = (size_t) bufsz + 40;
  void * buffer = memalign(page_size, buffer_size);
  mr->buffer = buffer;
  if (! mr->buffer)
  {
    multilog (log, LOG_ERR, "dada_ib_dg_alloc_buffer: failed to allocate aligned memory for buffer\n");
    return 0;
  }

  mr->mr = ibv_reg_mr(ctx->pd, mr->buffer, buffer_size, access_flags);
  if (!mr->mr)
  {
    multilog(log, LOG_ERR, "dada_ib_dg_alloc_buffer: ibv_reg_mr failed buffer=%p, "
                           "buf_size=%"PRIu64"\n", mr->buffer, bufsz);
    free(mr);
    return 0;
  }

  mr->size   = bufsz;
  mr->wr_id  = 0;

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "alloc_buffer: buffer=%p, size=%"PRIu64"\n", mr->buffer, mr->size);

  return mr;
}

dada_ib_datagram_dest_t * dada_ib_dg_get_local_port (dada_ib_datagram_t * ctx)
{
  dada_ib_datagram_dest_t * dest = malloc(sizeof(dada_ib_datagram_dest_t));

  struct ibv_port_attr attr;
  if (ibv_query_port(ctx->context, ctx->ib_port, &attr))
  {
    multilog(ctx->log, LOG_ERR, "dada_ib_dg_get_local_port: ibv_query_port failed\n");
    return 0;
  }

  srand48(getpid() * time(NULL));

  dest->lid = attr.lid;
  dest->qpn = ctx->qp->qp_num;
  dest->psn = lrand48() & 0xffffff;

  int gidx = -1;
  if (gidx >= 0) 
  {
    multilog (ctx->log, LOG_INFO, "dada_ib_dg_get_local_port: ibv_query_gid()\n");
    if (ibv_query_gid (ctx->context, ctx->ib_port, gidx, &(dest->gid))) 
    {
      multilog(ctx->log, LOG_ERR, "dada_ib_dg_get_local_port: could not get local gid for gid index %d\n", gidx);
      return 0;
    }
  }

  multilog (ctx->log, LOG_INFO, "dada_ib_dg_get_local_port: lid=%"PRIu16"\n", dest->lid);
  multilog (ctx->log, LOG_INFO, "dada_ib_dg_get_local_port: qpn=%"PRIu32"\n", dest->qpn);
  multilog (ctx->log, LOG_INFO, "dada_ib_dg_get_local_port: psn=%d\n", dest->psn);

  return dest;
}


int dada_ib_dg_free_buffer (dada_ib_mb_t * mr)
{
  if (mr->mr)
  {
    ibv_dereg_mr(mr->mr);
    mr->mr = 0; 
  }
  if (mr->buffer)
    free(mr->buffer);
  mr->buffer = 0;
  mr->size = 0;
  mr->wr_id = 0;
  free(mr);

  return 0;
}


int dada_ib_dg_post_recvs (dada_ib_datagram_t * ctx, dada_ib_mb_t ** mb, int n_to_post)
{
  multilog_t * log = ctx->log;

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "dada_ib_dg_post_recvs()\n");

  struct ibv_sge      sge;
  struct ibv_recv_wr  rcv_wr = { };
  struct ibv_recv_wr *bad_wr;

  int i;
  for (i = 0; i < n_to_post; ++i)
  {
    sge.addr       = (uintptr_t) mb[i]->buffer;
    sge.length     = mb[i]->size + 40;
    sge.lkey       = mb[i]->mr->lkey;

    rcv_wr.wr_id   = mb[i]->wr_id;
    rcv_wr.sg_list = &sge;
    rcv_wr.num_sge = 1;

    if (ctx->verbose > 1)
      multilog(log, LOG_INFO, "dada_ib_dg_post_recvs: posting [%d] wr_id=%d\n", i, mb[i]->wr_id);
    if (ibv_post_recv(ctx->qp, &rcv_wr, &bad_wr))
    {
      multilog(log, LOG_ERR, "dada_ib_post_recv: ibv_post_recv i=%d failed\n", i);
      break;
    }
  }

  return i;
}


/*
 *  post recv on the specified Memory Buffer (mb) and connection 
 */
int dada_ib_dg_post_recv (dada_ib_datagram_t * ctx, dada_ib_mb_t *mb)
{
  assert(ctx);
  multilog_t * log = ctx->log;

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "dada_ib_dg_post_recv()\n");

  struct ibv_sge      sge;
  struct ibv_recv_wr  rcv_wr = { };
  struct ibv_recv_wr *bad_wr;

  sge.addr   = (uintptr_t) mb->buffer;
  sge.length = mb->size + 40;
  sge.lkey   = mb->mr->lkey;

  rcv_wr.wr_id   = mb->wr_id;
  rcv_wr.sg_list = &sge;
  rcv_wr.num_sge = 1;

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "dada_ib_dg_post_recv: addr=%p, length=%d, lkey=%"PRIu32", wr_id=%d\n",
                            sge.addr, sge.length, sge.lkey, mb->wr_id);

  if (ibv_post_recv(ctx->qp, &rcv_wr, &bad_wr))
  {
    multilog(log, LOG_ERR, "dada_ib_post_recv: ibv_post_recv failed\n");
    return -1;
  }

  return 0;
}

/*
 * Wait for a CQE on the send CC/CQ with IBV_SUCCESS and matching wr_id
 */
int dada_ib_dg_wait_send (dada_ib_datagram_t * ctx, dada_ib_mb_t * mb)
{
  return dada_ib_dg_wait_cq (ctx, mb, ctx->channel, ctx->cq);
}

/*
 *  Waits for a CQE on the CC/CQ with IBV_SUCCESS and a matching wr_id
 */
int dada_ib_dg_wait_recv(dada_ib_datagram_t * ctx, dada_ib_mb_t * mb)
{
  return dada_ib_dg_wait_cq (ctx, mb, ctx->channel, ctx->cq);
}

int dada_ib_dg_wait_cq (dada_ib_datagram_t * ctx, dada_ib_mb_t * mb, 
                     struct ibv_comp_channel * comp_chan, struct ibv_cq * cq)
{ 
  assert(ctx);
  multilog_t * log = ctx->log;

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "dada_ib_dg_wait_cq()\n");

  struct ibv_cq  * evt_cq;
  void           * cq_context;
  struct ibv_wc    wc;

  int data_received = 0;

  assert(comp_chan != 0);
  assert(cq != 0);

  // check if this CQE is buffered
  if ((ctx->buffered_cqe) && (ctx->buffered_cqe_wr_id == mb->wr_id))
  {
    multilog(log, LOG_INFO, "wait_cq: restoring buffered event for %"PRIu64"\n", mb->wr_id);
    ctx->buffered_cqe_wr_id = wc.wr_id;
    ctx->buffered_cqe = 0;
    ctx->buffered_cqe_wr_id = 0;
    data_received = 1;
  }

  while (!data_received) {
  
    if (ctx->verbose > 1)
      multilog(log, LOG_INFO, "wait_cq: ibv_get_cq_event\n");
    if (ibv_get_cq_event(comp_chan, &evt_cq, &cq_context))
    {
      multilog(log, LOG_ERR, "wait_cq: ibv_get_cq_event failed\n");
      return -1;
    }

    // request notification for CQE's
    if (ctx->verbose > 1)
      multilog(log, LOG_INFO, "wait_cq: ibv_req_notify_cq\n");
    if (ibv_req_notify_cq(cq, 0))
    {
      multilog(log, LOG_ERR, "dada_ib_dg_wait_cq: ibv_req_notify_cq() failed\n");
      return -1;
    } 

    if (ctx->verbose > 1)
      multilog(log, LOG_INFO, "wait_cq: ibv_poll_cq\n");
    int ne = ibv_poll_cq(cq, 1, &wc);
    if (ne < 0) {
      multilog(log, LOG_ERR, "wait_cq: ibv_poll_cq() failed: ne=%d\n", ne);
      return -1;
    }

    if (wc.status != IBV_WC_SUCCESS) 
    {
      multilog(log, LOG_WARNING, "wait_cq: wc.status != IBV_WC_SUCCESS "
               "[wc.status=%s, wc.wr_id=%"PRIu64", mb->wr_id=%"PRIu64"]\n", 
               ibv_wc_status_str(wc.status), wc.wr_id, mb->wr_id);
      return -1;
    }

    if (wc.wr_id != mb->wr_id ) 
    {
      multilog(log, LOG_WARNING, "wait_cq: wr_id=%"PRIu64" != %"PRIu64"\n", wc.wr_id, mb->wr_id);
      uint64_t * tmpptr = (uint64_t *) mb->buffer;
      multilog(log, LOG_WARNING, "wait_cq: wr_id=%"PRIu64" key=%"PRIu64" val=%"PRIu64"\n", wc.wr_id, tmpptr[0], tmpptr[1]);

      ctx->verbose = 2;

      // give up if we have to buffer more than 1 CQE
      if (ctx->buffered_cqe)
      {
        multilog(log, LOG_ERR, "wait_cq: will not buffer more than 1 CQE\n");
        multilog(log, LOG_INFO,  "wait_cq: ibv_ack_cq_events\n"); 
        ibv_ack_cq_events(cq, 1);
        return -1;
      } 
      // save this CQE and wr_id so that it can be retrieved later
      else 
      {
        multilog(log, LOG_WARNING, "dada_ib_dg_wait_cq: buffering wr_id=%"PRIu64"\n", wc.wr_id);
        ctx->buffered_cqe = 1;
        ctx->buffered_cqe_wr_id = wc.wr_id;
      }
    }

    if ((wc.wr_id == mb->wr_id) && (wc.status == IBV_WC_SUCCESS))
    {
      if (ctx->verbose > 1)
        multilog(log, LOG_INFO, "wait_cq: wr correct\n");
      data_received = 1;
    }

    if (ctx->verbose > 1)
      multilog(log, LOG_INFO, "wait_cq: ibv_ack_cq_events\n"); 
    ibv_ack_cq_events(cq, 1);
  }

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "dada_ib_dg_wait_cq: returned\n");

  return 0;

}

/*
 *  post_send the specified memory buffer (ib) on the specified connection
 *
 *  Will send the data in the buffer in one transaction via the WR_SEND 
 *  mode which will require a post_recv on the remote end
 */
int dada_ib_dg_post_send (dada_ib_datagram_t * ctx, dada_ib_mb_t * mb, int remote_qpn)
{
  assert(ctx);
  multilog_t * log = ctx->log;

  struct ibv_sge       sge;
  struct ibv_send_wr * bad_send_wr;
  struct ibv_send_wr   wr = { };
  int err = 0;

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "dada_ib_dg_post_send: buffer=%p, bytes=%"PRIu64", wr_id=%d\n", mb->buffer, mb->size, mb->wr_id);

  sge.addr   = (uintptr_t) mb->buffer + 40;
  sge.length = mb->size;
  sge.lkey   = mb->mr->lkey;

  wr.wr_id      = mb->wr_id;
  wr.sg_list    = &sge;
  wr.num_sge    = 1;
  wr.opcode     = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  wr.wr.ud.ah          = ctx->ah;
  wr.wr.ud.remote_qpn  = remote_qpn;
  wr.wr.ud.remote_qkey = 0x11111111;

  /*
  struct ibv_send_wr wr = {
    .wr_id      = mb->wr_id,
    .sg_list    = &sge,
    .num_sge    = 1,
    .opcode     = IBV_WR_SEND,
    .send_flags = IBV_SEND_SIGNALED,
  }

    .wr = {
      .ud = {
        .ah          = ctx->ah,
        //.remote_qpn  = remote_qpn,
        .remote_qkey = 0x11111111
      }
    }
  };*/

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "dada_ib_dg_post_send: ibv_post_send\n");
  err = ibv_post_send(ctx->qp, &wr, &bad_send_wr);
  if (err)
  {
    multilog(log, LOG_ERR, "dada_ib_dg_post_send: ibv_post_send failed errno=%d strerror=%s\n", errno, strerror(errno));
    multilog(log, LOG_INFO, "dada_ib_dg_post_send: bad_send_wr: lkey=%p wr_id=%d\n", bad_send_wr->sg_list->lkey, bad_send_wr->wr_id);
    return -1;
  }

  return 0;

}

int dada_ib_dg_post_sends (dada_ib_datagram_t * ctx, dada_ib_mb_t ** mbs, int nmbs, int remote_qpn)
{
  multilog_t * log = ctx->log;

  struct ibv_sge       sge;
  struct ibv_send_wr * bad_send_wr;
  struct ibv_send_wr   wr = { };
  int err = 0;

  wr.sg_list    = &sge;
  wr.num_sge    = 1;
  wr.opcode     = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  wr.wr.ud.ah      = ctx->ah;
  wr.wr.ud.remote_qpn  = remote_qpn;
  wr.wr.ud.remote_qkey = 0x11111111;

  unsigned int i = 0;
  for (i=0; i < nmbs; i++)
  {
    sge.addr   = (uintptr_t) mbs[i]->buffer + 40;
    sge.length = mbs[i]->size;
    sge.lkey   = mbs[i]->mr->lkey;

    wr.wr_id   = mbs[i]->wr_id;
    wr.sg_list    = &sge;
    wr.num_sge    = 1;
    wr.opcode     = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;

    wr.wr.ud.ah      = ctx->ah;
    wr.wr.ud.remote_qpn  = remote_qpn;
    wr.wr.ud.remote_qkey = 0x11111111;

    if (ctx->verbose > 1)
      multilog(log, LOG_INFO, "dada_ib_dg_post_sends: ibv_post_send [%d]\n", i);
    err = ibv_post_send(ctx->qp, &wr, &bad_send_wr);
    if (err)
    {
      multilog(log, LOG_ERR, "dada_ib_dg_post_send: ibv_post_send failed errno=%d strerror=%s\n", errno, strerror(errno));
      multilog(log, LOG_INFO, "dada_ib_dg_post_send: bad_send_wr: lkey=%p wr_id=%d\n", bad_send_wr->sg_list->lkey, bad_send_wr->wr_id);
      return -1;
    }
  }
  return 0;
}



#if 0
// check if the UD message size is validm
int dada_ib_dg_check_message_size (dada_ib_datagram_t * ctx, int message_size)
{
  struct ibv_port_attr port_attr; 
  int ret;
  ret = ibv_query_port(ctx->context->verbs, ctx->port, &port_attr);
  if (ret)
    return ret;
  if (message_size > (1 << (port_attr.active_mtu + 7))) 
  {
    multilog(ctx->log, LOG_ERR, "dada_ib_check_message_size: size %d is larger "
             "than active mtu %d\n", message_size, 1 << (port_attr.active_mtu + 7));
    return -EINVAL;
  }
  return 0;
}
#endif


/*
 * Disconnect the IB connection  
 */
int dada_ib_dg_disconnect(dada_ib_datagram_t * ctx)
{

  assert(ctx);
  multilog_t * log = ctx->log;

  int err;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "disconnect: ibv_destroy_qp\n"); 
  if (ctx->qp)
  {
    if (ibv_destroy_qp(ctx->qp)) {
      multilog(log, LOG_ERR, "disconnect: failed to destroy QP\n");
      err = 1;
    }
    ctx->qp = 0;
  }

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "disconnect: ibv_destroy_cq\n"); 
  if (ctx->cq) {
    if (ibv_destroy_cq(ctx->cq)) {
      multilog(log, LOG_ERR, "disconnect: failed to destroy CQ\n");
      err = 1;
    }
    ctx->cq = 0;
  }

  if (ctx->bufs) 
  {
    unsigned i=0;
    for (i=0; i<ctx->nbufs; i++)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "disconnect: dada_ib_dg_buffer bufs[%d]\n", i);
      if (dada_ib_dg_free_buffer(ctx->bufs[i]))
      {
        multilog(log, LOG_ERR, "disconnect: failed to free MR[%d]\n", i);
        err = 1;
      }
    }
  }

  if (ctx->channel) 
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "disconnect: ibv_destroy_comp_channel()\n");
    if (ibv_destroy_comp_channel(ctx->channel)) 
    {
      multilog(log, LOG_ERR, "disconnect: failed to destroy completion channel\n");
      err = 1;
    }
    ctx->channel = 0;
  }

  if (ctx->pd) 
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "disconnect: ibv_dealloc_pd()\n");
    if (ibv_dealloc_pd(ctx->pd)) 
    {
      multilog(log, LOG_ERR, "disconnect: failed to deallocate PD\n");
      err = 1;
    }
    ctx->pd = 0;
  }

  if (ctx->context)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "disconnect: ibv_close_device()\n");
    if (ibv_close_device (ctx->context))
    {
      multilog(log, LOG_ERR, "disconnect: failed to close IBV device\n");
      err = 1;
    }
    ctx->context = 0;
  }

  return err;
}

/*
 * clean up IB resources 
 */
int dada_ib_dg_destroy (dada_ib_datagram_t * ctx)
{

  int err;

  assert(ctx);
  multilog_t * log = ctx->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "dada_ib_destroy()\n");

  if (dada_ib_dg_disconnect(ctx) < 0)
  {
    multilog(log, LOG_ERR, "destroy: dada_ib_disconnect failed()\n");    
    err = 1;    
  }

  if (ctx->bufs)
    free(ctx->bufs);
  ctx->bufs = 0;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "destroy: success\n");

  ctx->log = 0;

  if (ctx->dev_list)
    free (ctx->dev_list);
  ctx->dev_list = 0;

  free(ctx);

  return err;
}


dada_ib_datagram_dest_t * dada_ib_dg_client_exch_dest ( const char *servername, int port, const dada_ib_datagram_dest_t * my_dest)
{
  struct addrinfo *res, *t;
  struct addrinfo hints = {
      .ai_family   = AF_UNSPEC,
      .ai_socktype = SOCK_STREAM
  };
  char *service;
  char msg[sizeof "0000:000000:000000:00000000000000000000000000000000"];
  int n;
  int sockfd = -1;
  dada_ib_datagram_dest_t *rem_dest = NULL;
  char gid[33];

  if (asprintf(&service, "%d", port) < 0)
    return NULL;

  n = getaddrinfo(servername, service, &hints, &res);

  if (n < 0) {
    fprintf(stderr, "%s for %s:%d\n", gai_strerror(n), servername, port);
    free(service);
    return NULL;
  }

  for (t = res; t; t = t->ai_next) {
    sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
    if (sockfd >= 0) {
      if (!connect(sockfd, t->ai_addr, t->ai_addrlen))
        break;
      close(sockfd);
      sockfd = -1;
    }
  }

  freeaddrinfo(res);
  free(service);

  if (sockfd < 0) {
    fprintf(stderr, "Couldn't connect to %s:%d\n", servername, port);
    return NULL;
  }

  gid_to_wire_gid(&my_dest->gid, gid);
  sprintf(msg, "%04x:%06x:%06x:%s", my_dest->lid, my_dest->qpn, my_dest->psn, gid);
  if (write(sockfd, msg, sizeof msg) != sizeof msg) {
    fprintf(stderr, "Couldn't send local address\n");
    goto out;
  }

  if (read(sockfd, msg, sizeof msg) != sizeof msg) {
    perror("client read");
    fprintf(stderr, "Couldn't read remote address\n");
    goto out;
  }

  write(sockfd, "done", sizeof "done");

  rem_dest = malloc(sizeof *rem_dest);
  if (!rem_dest)
    goto out;

  sscanf(msg, "%x:%x:%x:%s", &rem_dest->lid, &rem_dest->qpn, &rem_dest->psn, gid);
  wire_gid_to_gid(gid, &rem_dest->gid);

out:
  close(sockfd);
  return rem_dest;
}

dada_ib_datagram_dest_t * dada_ib_dg_server_exch_dest(dada_ib_datagram_t * ctx,
    int ib_port, int port, int sl, dada_ib_datagram_dest_t * my_dest, int sgid_idx)
{
    struct addrinfo *res, *t;
    struct addrinfo hints = {
        .ai_flags    = AI_PASSIVE,
        .ai_family   = AF_UNSPEC,
        .ai_socktype = SOCK_STREAM
    };
    char *service;
    char msg[sizeof "0000:000000:000000:00000000000000000000000000000000"];
    int n;
    int sockfd = -1, connfd;
    dada_ib_datagram_dest_t * rem_dest = NULL;
    char gid[33];

    if (asprintf(&service, "%d", port) < 0)
        return NULL;

    n = getaddrinfo(NULL, service, &hints, &res);

    if (n < 0) {
        fprintf(stderr, "%s for port %d\n", gai_strerror(n), port);
        free(service);
        return NULL;
    }

    for (t = res; t; t = t->ai_next) {
        sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
        if (sockfd >= 0) {
            n = 1;

            setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &n, sizeof n);

            if (!bind(sockfd, t->ai_addr, t->ai_addrlen))
                break;
            close(sockfd);
            sockfd = -1;
        }
    }

    freeaddrinfo(res);
    free(service);

    if (sockfd < 0) {
      fprintf(stderr, "Couldn't listen to port %d\n", port);
      return NULL;
    }

    multilog (ctx->log, LOG_INFO, "dada_ib_dg_server_exch_dest: waiting for connection\n");

    listen(sockfd, 1);
    connfd = accept(sockfd, NULL, 0);
    close(sockfd);
    if (connfd < 0) {
      fprintf(stderr, "accept() failed\n");
      return NULL;
    }

    multilog (ctx->log, LOG_INFO, "dada_ib_dg_server_exch_dest: accepted connection\n");

    n = read(connfd, msg, sizeof msg);
    if (n != sizeof msg) {
      perror("server read");
      fprintf(stderr, "%d/%d: Couldn't read remote address\n", n, (int) sizeof msg);
      goto out;
    }

    rem_dest = malloc(sizeof *rem_dest);
    if (!rem_dest)
      goto out;

    sscanf(msg, "%x:%x:%x:%s", &rem_dest->lid, &rem_dest->qpn, &rem_dest->psn, gid);
    wire_gid_to_gid(gid, &rem_dest->gid);

    multilog (ctx->log, LOG_INFO, "dada_ib_dg_server_exch_dest: dada_ib_dg_activate() sl=%d\n", sl);
    if (dada_ib_dg_activate (ctx, my_dest, rem_dest, sgid_idx, sl) < 0)
    {
      fprintf(stderr, "Couldn't connect to remote QP\n");
      free(rem_dest);
      rem_dest = NULL;
      goto out;
    }

    gid_to_wire_gid(&my_dest->gid, gid);
    sprintf(msg, "%04x:%06x:%06x:%s", my_dest->lid, my_dest->qpn, my_dest->psn, gid);
    if (write(connfd, msg, sizeof msg) != sizeof msg) {
      fprintf(stderr, "Couldn't send local address\n");
      free(rem_dest);
      rem_dest = NULL;
      goto out;
    }

    read(connfd, msg, sizeof msg);

out:
    close(connfd);
    return rem_dest;
}

void wire_gid_to_gid(const char *wgid, union ibv_gid *gid)
{
  char tmp[9];
  uint32_t v32;
  int i;

  for (tmp[8] = 0, i = 0; i < 4; ++i) {
    memcpy(tmp, wgid + i * 8, 8);
    sscanf(tmp, "%x", &v32);
    *(uint32_t *)(&gid->raw[i * 4]) = ntohl(v32);
  }
}

void gid_to_wire_gid(const union ibv_gid *gid, char wgid[])
{
  int i;

  for (i = 0; i < 4; ++i)
    sprintf(&wgid[i * 8], "%08x", htonl(*(uint32_t *)(gid->raw + i * 4)));
}

void dada_ib_dg_encode_header (char *b, uint64_t seq_no)
{
  b[0] = (uint8_t) (seq_no>>56);
  b[1] = (uint8_t) (seq_no>>48);
  b[2] = (uint8_t) (seq_no>>40);
  b[3] = (uint8_t) (seq_no>>32);
  b[4] = (uint8_t) (seq_no>>24);
  b[5] = (uint8_t) (seq_no>>16);
  b[6] = (uint8_t) (seq_no>>8);
  b[7] = (uint8_t) (seq_no);
}

void dada_ib_dg_decode_header (unsigned char * b, uint64_t *seq_no)
{
  uint64_t tmp = 0;
  unsigned i = 0;
  *seq_no = UINT64_C (0);
  for (i = 0; i < 8; i++ )
  {
    tmp = UINT64_C (0);
    tmp = b[8 - i - 1];
    *seq_no |= (tmp << ((i & 7) << 3));
  }
}
