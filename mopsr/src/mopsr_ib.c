/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "mopsr_ib.h"

mopsr_conn_t * mopsr_setup_cornerturn_send (const char * config_file, mopsr_ib_t * ctx, unsigned int send_id)
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

  // senders will maintain nchan connections
  ctx->nconn = ctx->nchan;
  mopsr_conn_t * conns = (mopsr_conn_t *) malloc (sizeof(mopsr_conn_t) * ctx->nconn);

  char host[64];
  unsigned int ichan;
  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    sprintf (key, "RECV_%d", ichan);
    if (ascii_header_get (config , key, "%s", host) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: config with no %s\n", key);
      return 0;
    }

    // set destination host for opening IB connection
    strcpy (conns[ichan].host, host);

    // now get the IB address for this hostname
    sprintf (key, "%s_IB", host);
    if (ascii_header_get (config, key, "%s", host) != 1)
    {
      multilog (ctx->log, LOG_ERR, "setup_cornerturn_send: config with no %s\n", key);
      return 0;
    }
    strcpy (conns[ichan].ib_host, host);

    // set destination port for opening IB connection
    conns[ichan].port      = chan_baseport + (send_id * ctx->nchan) + ichan;
    conns[ichan].chan      = ichan;
    conns[ichan].ant_first = ant_first;
    conns[ichan].ant_last  = ant_last;
    conns[ichan].ib_cm     = 0;
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
mopsr_conn_t * mopsr_setup_cornerturn_recv (const char * config_file, mopsr_ib_t * ctx, unsigned int channel)
{
  char config[65536];

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "setup_cornerturn_recv: fileread (%s)\n", config_file);
  if (fileread (config_file, config, 65536) < 0)
  {
    fprintf (stderr, "ERROR: could not read ASCII configuration from %s\n", config_file);
    return 0;
  }

  if (ascii_header_get (config , "NCHAN", "%d", &(ctx->nchan)) != 1)
  {
    multilog (ctx->log, LOG_ERR, "setup_cornerturn_recv: config with no NCHAN\n");
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
  mopsr_conn_t * conns = (mopsr_conn_t *) malloc (sizeof(mopsr_conn_t) * ctx->nconn);

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

    conns[isend].port      = chan_baseport + (isend * ctx->nchan) + channel;
    conns[isend].chan      = channel;
    conns[isend].ant_first = ant_first;
    conns[isend].ant_last  = ant_last;

    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "setup_cornerturn_recv: conns[%d].port=%d "
                "ctx->nchan=%d channel=%d\n", isend, conns[isend].port, ctx->nchan, channel);

  }
  return conns;
}

