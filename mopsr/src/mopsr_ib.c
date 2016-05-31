/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "mopsr_ib.h"

mopsr_bf_conn_t * mopsr_setup_cornerturn_send (const char * config_file, mopsr_bf_ib_t * ctx, unsigned int send_id)
{
  char config[65536];
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "setup_cornerturn_send: fileread (%s)\n", config_file);
  if (fileread (config_file, config, 65536) < 0)
  {
    fprintf (stderr, "ERROR: could not read ASCII configuration from %s\n", config_file);
    return 0;
  }

  if (ascii_header_get (config , "NCHAN", "%d", &(ctx->nchan)) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: config with no NCHAN\n");
    return 0;
  }

  if (ascii_header_get (config , "NANT", "%d", &(ctx->nant)) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: config with no NANT\n");
    return 0;
  }

  unsigned int nsend;
  if (ascii_header_get (config , "NSEND", "%d", &nsend) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: config with no NSEND\n");
    return 0;
  }
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "setup_cornerturn_send: NSEND=%d\n", nsend);

  unsigned int nrecv;
  if (ascii_header_get (config , "NRECV", "%d", &nrecv) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: config with no NRECV\n");
    return 0;             
  }
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "setup_cornerturn_send: NRECV=%d\n", nrecv);

  unsigned int chan_baseport;
  if (ascii_header_get (config , "CHAN_BASEPORT", "%d", &chan_baseport) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: config with no CHAN_BASEPORT\n");
    return 0;
  }

  char key[32];
  unsigned int ant_first;
  unsigned int ant_last;
  sprintf (key, "ANT_FIRST_SEND_%d", send_id);
  if (ascii_header_get (config , key, "%d", &ant_first) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: config with no %s\n", key);
    return 0;
  }

  sprintf (key, "ANT_LAST_SEND_%d", send_id);
  if (ascii_header_get (config , key, "%d", &ant_last) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: config with no %s\n", key);
    return 0;
  }

  // senders will maintain nrecv connections
  ctx->nconn = nrecv;
  mopsr_bf_conn_t * conns = (mopsr_bf_conn_t *) malloc (sizeof(mopsr_bf_conn_t) * ctx->nconn);

  char host[64];
  unsigned int irecv, ichan;
  for (irecv=0; irecv<nrecv; irecv++)
  {
    sprintf (key, "RECV_%d", irecv);
    if (ascii_header_get (config , key, "%s", host) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: config with no %s\n", key);
      return 0;
    }

    // get the channel to be sent to this receiver
    sprintf (key, "RECV_CHAN_%d", irecv);
    if (ascii_header_get (config , key, "%u", &ichan) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: config with no %s\n", key);
      return 0;
    }

    // set destination host for opening IB connection
    strcpy (conns[irecv].host, host);

    // now get the IB address for this hostname
    sprintf (key, "%s_IB", host);
    if (ascii_header_get (config, key, "%s", host) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: config with no %s\n", key);
      return 0;
    }
    strcpy (conns[irecv].ib_host, host);

    // set destination port for opening IB connection
    conns[irecv].pfb       = send_id;
    conns[irecv].port      = chan_baseport + (send_id * nrecv) + irecv;
    conns[irecv].chan      = ichan;
    conns[irecv].npfb      = nsend;
    conns[irecv].ant_first = ant_first;
    conns[irecv].ant_last  = ant_last;
    conns[irecv].ib_cm     = 0;
  }

#if 0
  unsigned irecv;
  char host[64];
  unsigned int chan_first;
  unsigned int chan_last;
  unsigned int prev_chan = -1;
  for (irecv=0; irecv<nrecv; irecv++)
  {
    sprintf (key, "RECV_%d", irecv);
    if (ascii_header_get (config , key, "%s", host) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: config with no %s\n", key);
      return 0;
    }

    sprintf (key, "CHAN_FIRST_RECV_%d", irecv);
    if (ascii_header_get (config , key, "%d", &chan_first) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: config with no %s\n", key);
      return 0;
    }

    sprintf (key, "CHAN_LAST_RECV_%d", irecv);
    if (ascii_header_get (config , key, "%d", &chan_last) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: config with no %s\n", key);
      return 0;
    }

    unsigned int ichan;
    for (ichan=chan_first; ichan<=chan_last; ichan++)
    {
      if (ichan != prev_chan + 1)
      {
        multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: non-continuous channels for %s\n", host);
        return 0;
      }
        
      // set destination host for opening IB connection
      strcpy (conns[ichan].host, host);

      // set destination port for opening IB connection
      conns[ichan].port      = chan_baseport + (send_id * ctx->nchan) + ichan;
      conns[ichan].chan      = ichan;
      conns[ichan].ant_first = ant_first;
      conns[ichan].ant_last  = ant_last;
      conns[ichan].ib_cm     = 0;
      prev_chan = ichan;
    }
  }
#endif
  return conns;
}

/*
 *  
 */
mopsr_bf_conn_t * mopsr_setup_cornerturn_recv (const char * config_file, mopsr_bf_ib_t * ctx, unsigned int channel)
{
  char config[65536];

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "setup_cornerturn_recv: fileread (%s)\n", config_file);
  if (fileread (config_file, config, 65536) < 0)
  {
    fprintf (stderr, "ERROR: could not read ASCII configuration from %s\n", config_file);
    return 0;
  }

  // number of input channels to cornerturn 
  if (ascii_header_get (config , "NCHAN", "%d", &(ctx->nchan)) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_cornerturn_recv: config with no NCHAN\n");
    return 0;
  }

  // numner of output channels from cornerturn
  unsigned int nrecv;
  if (ascii_header_get (config , "NRECV", "%d", &nrecv) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_read_config: config with no NRECV\n");
    return 0;
  }

  if (ascii_header_get (config , "NANT", "%d", &(ctx->nant)) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_cornerturn_recv: config with no NANT\n");
    return 0;
  }

  unsigned int nsend;
  if (ascii_header_get (config , "NSEND", "%d", &nsend) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_cornerturn_recv: config with no NSEND\n");
    return 0;
  }
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "setup_cornerturn_recv: NSEND=%d\n", nsend);

  unsigned int chan_baseport;
  if (ascii_header_get (config , "CHAN_BASEPORT", "%d", &chan_baseport) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_cornerturn_recv: config with no CHAN_BASEPORT\n");
    return 0;
  }

  // senders will maintain nchan connections
  ctx->nconn = nsend;
  mopsr_bf_conn_t * conns = (mopsr_bf_conn_t *) malloc (sizeof(mopsr_bf_conn_t) * ctx->nconn);

  char key[32];
  char host[64];
  unsigned int ant_first;
  unsigned int ant_last;
  unsigned int isend;
  for (isend=0; isend<nsend; isend++)
  {
    sprintf (key, "SEND_%d", isend);
    if (ascii_header_get (config , key, "%s", host) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_cornerturn_recv: config with no %s\n", key);
      return 0;
    }
    strcpy (conns[isend].host, host);

    // now get the IB address for this hostname
    sprintf (key, "%s_IB", host);
    if (ascii_header_get (config, key, "%s", host) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_cornerturn_recv: config with no %s\n", key);
      return 0;
    } 
    strcpy (conns[isend].ib_host, host);

    sprintf (key, "ANT_FIRST_SEND_%d", isend);
    if (ascii_header_get (config , key, "%d", &ant_first) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_cornerturn_recv: config with no %s\n", key);
      return 0;
    }

    sprintf (key, "ANT_LAST_SEND_%d", isend);
    if (ascii_header_get (config , key, "%d", &ant_last) != 1)
    { 
      multilog (ctx->log, LOG_ERR, "setup_cornerturn_recv: config with no %s\n", key);
      return 0;
    } 

    conns[isend].pfb       = isend;
    conns[isend].port      = chan_baseport + (isend * nrecv) + channel;
    conns[isend].chan      = channel;
    conns[isend].npfb      = nsend;
    conns[isend].ant_first = ant_first;
    conns[isend].ant_last  = ant_last;

    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "setup_cornerturn_recv: conns[%d].port=%d "
                "ctx->nchan=%d nrecv=%u channel=%d\n", isend, conns[isend].port, ctx->nchan, nrecv, channel);
  }
  return conns;
}

int mopsr_setup_bp_read_config (mopsr_bp_ib_t * ctx, const char * config_file, char * config)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "setup_bp_read_config: fileread (%s)\n", config_file);

  if (fileread (config_file, config, 65536) < 0)
  {
    fprintf (stderr, "ERROR: could not read ASCII configuration from %s\n", config_file);
    return -1;
  }

  if (ascii_header_get (config , "NDIM", "%d", &(ctx->ndim)) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_bp_read_config: config with no NDIM\n");
    return -1;
  }

  int nbit;
  if (ascii_header_get (config , "NBIT", "%d", &nbit) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_bp_read_config: config with no NBIT\n");
    return -1;
  }
  ctx->nbyte = nbit * 8;

  // get the number of coarse and fine channels
  int nchan_coarse, nchan_fine;
  if (ascii_header_get (config , "NCHAN_COARSE", "%d", &nchan_coarse) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_bp_read_config: config with no NCHAN_COARSE\n");
    return -1;
  }
  if (ascii_header_get (config , "NCHAN_FINE", "%d", &nchan_fine) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_bp_read_config: config with no NCHAN_FINE\n");
    return -1;
  }

  ctx->nchan_recv = nchan_coarse * nchan_fine;

  // total number of backend beams
  if (ascii_header_get (config , "NBEAM", "%d", &(ctx->nbeam_send)) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_bp_read_config: config with no NBEAM\n");
    return -1;
  }

  if (ascii_header_get (config , "NSEND", "%d", &(ctx->nsend)) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_bp_read_config: config with no NSEND\n");
    return -1;
  }
  unsigned int nrecv;
  if (ascii_header_get (config , "NRECV", "%d", &nrecv) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_bp_read_config: config with no NRECV\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "setup_bp_read_config: NRECV=%d\n", nrecv);

  ctx->nbeam_recv = ctx->nbeam_send / nrecv;
  ctx->nchan_send = ctx->nchan_recv / ctx->nsend;

  return 0;
}
//
// send_id is a proxy for ichan
//
mopsr_bp_conn_t * mopsr_setup_bp_cornerturn_send (const char * config_file, mopsr_bp_ib_t * ctx, unsigned int send_id)
{
  char config[65536];

  if (mopsr_setup_bp_read_config (ctx, config_file, config) < 0)
  {
    multilog (ctx->log, LOG_INFO, "setup_bp_cornerturn_send: failed to read configuration\n");
    return 0;
  }

  unsigned int chan_baseport;
  if (ascii_header_get (config , "CHAN_BASEPORT", "%d", &chan_baseport) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_bp_cornerturn_send: config with no CHAN_BASEPORT\n");
    return 0;
  }
  char key[32];
  // read in the first and last channels for this sender
  unsigned chan_first;
  unsigned chan_last;
  sprintf (key, "CHAN_FIRST_SEND_%u", send_id);
  if (ascii_header_get (config , key, "%u", &chan_first) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_bp_cornerturn_send: config with no %s\n", key);
    return 0;
  }
  sprintf (key, "CHAN_LAST_SEND_%u", send_id);
  if (ascii_header_get (config , key, "%u", &chan_last) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_bp_cornerturn_send: config with no %s\n", key);
    return 0;
  }

  // get the chan for this sender
  unsigned beam_first;
  unsigned beam_last;

  // senders will open nrecv connections
  if (ascii_header_get (config , "NRECV", "%d", &(ctx->nconn)) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_bp_read_config: config with no NRECV\n");
    return 0;
  }
  mopsr_bp_conn_t * conns = (mopsr_bp_conn_t *) malloc (sizeof(mopsr_bp_conn_t) * ctx->nconn);

  char host[64];
  unsigned int iconn;
  for (iconn=0; iconn<ctx->nconn; iconn++)
  {
    sprintf (key, "RECV_%d", iconn);
    if (ascii_header_get (config , key, "%s", host) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_bp_cornerturn_send: config with no %s\n", key);
      return 0;
    }

    // set destination host for opening IB connection
    strcpy (conns[iconn].host, host);

    // now get the IB address for this hostname
    sprintf (key, "%s_IB", host);
    if (ascii_header_get (config, key, "%s", host) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_bp_cornerturn_send: config with no %s\n", key);
      return 0;
    }
    strcpy (conns[iconn].ib_host, host);

    sprintf (key, "BEAM_FIRST_RECV_%u", iconn);
    if (ascii_header_get (config , key, "%u", &beam_first) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_bp_cornerturn_send: config with no %s\n", key);
      return 0;
    }

    sprintf (key, "BEAM_LAST_RECV_%u", iconn);
    if (ascii_header_get (config , key, "%u", &beam_last) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_bp_cornerturn_send: config with no %s\n", key);
      return 0;
    }

    // set destination port for opening IB connection
    conns[iconn].port       = chan_baseport + (send_id * ctx->nsend) + iconn;
    conns[iconn].chan_first = chan_first;
    conns[iconn].chan_last  = chan_last;
    conns[iconn].beam_first = beam_first;
    conns[iconn].beam_last  = beam_last;
    conns[iconn].ib_cm      = 0;
    conns[iconn].isend      = send_id;
    conns[iconn].nsend      = ctx->nsend;
    conns[iconn].irecv      = iconn;
  }

  return conns;
}

/*
 *
 */
mopsr_bp_conn_t * mopsr_setup_bp_cornerturn_recv (const char * config_file, mopsr_bp_ib_t * ctx, unsigned int recv_id)
{
  char config[65536];

  if (mopsr_setup_bp_read_config (ctx, config_file, config) < 0)
  {
    multilog (ctx->log, LOG_INFO, "setup_bp_cornerturn_recv: failed to read configuration\n");
    return 0;
  }

  unsigned int chan_baseport;
  if (ascii_header_get (config , "CHAN_BASEPORT", "%d", &chan_baseport) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_bp_cornerturn_recv: config with no CHAN_BASEPORT\n");
    return 0;
  }

  // receivers wil maintain NSEND connections
  ctx->nconn = ctx->nsend;
  mopsr_bp_conn_t * conns = (mopsr_bp_conn_t *) malloc (sizeof(mopsr_bp_conn_t) * ctx->nconn);

  char key[32];
  char host[64];
  unsigned int beam_first;
  unsigned int beam_last;

  sprintf (key, "BEAM_FIRST_RECV_%d", recv_id);
  if (ascii_header_get (config , key, "%d", &beam_first) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_bp_cornerturn_recv: config with no %s\n", key);
    return 0;
  }

  sprintf (key, "BEAM_LAST_RECV_%d", recv_id);
  if (ascii_header_get (config , key, "%d", &beam_last) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_bp_cornerturn_recv: config with no %s\n", key);
    return 0;
  }

  unsigned chan_first, chan_last, iconn;
  
  for (iconn=0; iconn<ctx->nconn; iconn++)
  {
    sprintf (key, "SEND_%d", iconn);
    if (ascii_header_get (config , key, "%s", host) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_bp_cornerturn_recv: config with no %s\n", key);
      return 0;
    }
    strcpy (conns[iconn].host, host);

    // now get the IB address for this hostname
    sprintf (key, "%s_IB", host);
    if (ascii_header_get (config, key, "%s", host) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_bp_cornerturn_recv: config with no %s\n", key);
      return 0;
    }
    strcpy (conns[iconn].ib_host, host);

    sprintf (key, "CHAN_FIRST_SEND_%u", iconn);
    if (ascii_header_get (config , key, "%u", &chan_first) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_bp_cornerturn_recv: config with no %s\n", key);
      return 0;
    }

    sprintf (key, "CHAN_LAST_SEND_%u", iconn);
    if (ascii_header_get (config , key, "%u", &chan_last) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_bp_cornerturn_recv: config with no %s\n", key);
      return 0;
    }

    conns[iconn].port       = chan_baseport + (iconn * ctx->nconn) + recv_id;
    conns[iconn].chan_first = chan_first;
    conns[iconn].chan_last  = chan_last;
    conns[iconn].beam_first = beam_first;
    conns[iconn].beam_last  = beam_last;
    conns[iconn].isend      = iconn;
    conns[iconn].nsend      = ctx->nsend;
    conns[iconn].irecv      = recv_id;

    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "setup_bp_cornerturn_recv: conns[%d].port=%d "
                "ctx->nchan_send=%d recv_id=%d\n", iconn, conns[iconn].port, ctx->nchan_send, recv_id);

  }
  return conns;
}

