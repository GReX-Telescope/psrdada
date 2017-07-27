/***************************************************************************
 *  
 *    Copyright (C) 2017 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "string.h"
#include "bpsr_ct.h"

int bpsr_setup_cornerturn (char * config, const char * config_file, bpsr_ct_t * ctx)
{
  if (fileread (config_file, config, 65536) < 0)
  {
    fprintf (stderr, "ERROR: could not read ASCII configuration from %s\n", config_file);
    return -1;
  }

  if (ascii_header_get (config , "NSEND", "%d", &(ctx->nsend)) != 1)
  {
    fprintf (stderr, "ERROR: setup_cornerturn: config with no NSEND\n");
    return -1;
  }

  if (ascii_header_get (config , "NRECV", "%d", &(ctx->nrecv)) != 1)
  {
    fprintf (stderr, "ERROR: setup_cornerturn: config with no NRECV\n");
    return -1;
  }

  if (ascii_header_get (config , "SEND_BLOCK_SIZE", "%lu", &(ctx->send_block_size)) != 1)
  {
    fprintf (stderr, "ERROR: setup_cornerturn: config with no SEND_BLOCK_SIZE\n");
    return -1;
  }

  if (ascii_header_get (config , "SEND_RESOLUTION", "%lu", &(ctx->send_resolution)) != 1)
  {
    fprintf (stderr, "ERROR: setup_cornerturn: config with no SEND_RESOLUTION\n");
    return -1;
  }

  if (ascii_header_get (config , "ATOMIC_SIZE", "%lu", &(ctx->atomic_size)) != 1)
  {
    fprintf (stderr, "ERROR: setup_cornerturn: config with no ATOMIC_SIZE\n");
    return -1;
  }

  if (ascii_header_get (config , "RECV_BLOCK_SIZE", "%lu", &(ctx->recv_block_size)) != 1)
  {
    fprintf (stderr, "ERROR: setup_cornerturn: config with no RECV_BLOCK_SIZE\n");
    return -1;
  }
  
  if (ascii_header_get (config , "RECV_RESOLUTION", "%lu", &(ctx->recv_resolution)) != 1)
  {
    fprintf (stderr, "ERROR: setup_cornerturn: config with no RECV_RESOLUTION\n");
    return -1;
  }

  if (ascii_header_get (config , "BASEPORT", "%d", &(ctx->baseport)) != 1)
  {
    fprintf (stderr, "ERROR: setup_cornerturn: config with no CHAN_BASEPORT\n");
    return 0;
  }

  return 0;
}

bpsr_conn_t * bpsr_setup_send (const char * config_file, bpsr_ct_t * ctx, unsigned send_id)
{
  char config[65536];
  char key[32];
  char host[64];

  // read generic cornerturn info
  if (bpsr_setup_cornerturn(config, config_file, ctx) < 0)
  {
    fprintf (stderr, "ERROR: could not read general cornerturn configuration\n");
    return 0;
  }

  // senders will maintain nrecv connections
  ctx->nconn = ctx->nrecv;
  bpsr_conn_t * conns = (bpsr_conn_t *) malloc (sizeof(bpsr_conn_t) * ctx->nconn);

  unsigned int irecv, ichan;
  for (irecv=0; irecv<ctx->nrecv; irecv++)
  {
    sprintf (key, "RECV_%d", irecv);
    if (ascii_header_get (config , key, "%s", host) != 1)
    {
      fprintf (stderr, "ERROR: setup_cornerturn_send: config with no %s\n", key);
      return 0;
    }

    strcpy (conns[irecv].host, host);
    conns[irecv].port   = ctx->baseport + (send_id * ctx->nrecv) + irecv;

    conns[irecv].isend  = send_id;
    conns[irecv].irecv  = irecv;

    // compute the send/receive starting offsets
    conns[irecv].send_offset = send_id * ctx->atomic_size;
    conns[irecv].recv_offset = irecv * ctx->atomic_size;

    conns[irecv].atomic_size = ctx->atomic_size;
  }

  return conns;
}

/*
 *  
 */
bpsr_conn_t * bpsr_setup_recv (const char * config_file, bpsr_ct_t * ctx, unsigned recv_id)
{
  char config[65536];
  char key[32];
  char host[64];

  // read generic cornerturn info
  if (bpsr_setup_cornerturn(config, config_file, ctx) < 0)
  {
    fprintf (stderr, "ERROR: could not read general cornerturn configuration\n");
    return 0;
  }

  // receivers will maintain NSEND connections
  ctx->nconn = ctx->nsend;
  bpsr_conn_t * conns = (bpsr_conn_t *) malloc (sizeof(bpsr_conn_t) * ctx->nconn);

  unsigned isend;
  for (isend=0; isend<ctx->nsend; isend++)
  {
    sprintf (key, "SEND_%d", isend);
    if (ascii_header_get (config , key, "%s", host) != 1)
    {
      fprintf (stderr, "ERROR: setup_dirty_recv: config with no %s\n", key);
      return 0;
    }

    strcpy (conns[isend].host, host);
    conns[isend].port   = ctx->baseport + (isend * ctx->nrecv) + recv_id;

    conns[isend].isend  = isend;
    conns[isend].irecv  = recv_id;

    // compute the send/receive starting offsets
    conns[isend].send_offset = (unsigned) isend * ctx->atomic_size;
    conns[isend].recv_offset = (unsigned) recv_id * ctx->atomic_size;
    conns[isend].atomic_size = (unsigned) ctx->atomic_size;

  }

  return conns;
}
