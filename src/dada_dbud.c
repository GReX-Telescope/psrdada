#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_msg.h"
#include "dada_ib_datagram.h"
#include "dada_affinity.h"

#include "ascii_header.h"
#include "daemon.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/socket.h>


void usage()
{
  fprintf (stdout,
     "Usage: dada_dbud [options] dest_host\n"
     " -b <core>     bind process to CPU core\n"
     " -g <guid>     GUID of destination IB port\n"
     " -h            print this usage help\n"
     " -k <key>      hexadecimal shared memory key  [default: %x]\n"
     " -l <lid>      LID of destination IB port\n"
     " -q <qpn>      QPN of destination IB port\n"
     " -p <port>     port to transmit to [default: %d]\n"
     " -s            single transfer only\n",
     DADA_DEFAULT_BLOCK_KEY,
     DADA_DEFAULT_IBDB_PORT);
}


typedef struct dada_dbud {
  
  // port to listen on for connections 
  unsigned port;

  // verbose messages
  char verbose;

  // flag for active RDMA connection
  unsigned connected;

  void * send_buf;

  size_t send_bufsz;

  char * header;
  
  /* current observation id, as defined by OBS_ID attribute */
  char obs_id [DADA_OBS_ID_MAXLEN];

  // Infiniband Connection Manager
  dada_ib_datagram_t * ib_dg;

  // Remote IB dest ptr
  dada_ib_datagram_dest_t * remote;

} dada_dbud_t;

#define DADA_IBDB_INIT { 0, 0, 0, 0, 0, "", "", 0 }

/*! transfer function for transfer of just header */
int64_t dada_dbud_send (dada_client_t* client, void* data, uint64_t data_size)
{
  dada_dbud_t * dbud = (dada_dbud_t *) client->context;

  dada_ib_datagram_t * ib_dg = dbud->ib_dg;

  multilog_t* log = client->log;

  if (dbud->verbose)
    multilog (log, LOG_INFO, "dada_dbud_send()\n");

  if (dbud->verbose)
    multilog(log, LOG_INFO, "send: returning %"PRIu64" bytes\n", data_size);

  return (int64_t) data_size;
}

/*
 * transfer function write data directly to the specified memory
 * block buffer with the specified block_id and size
 */
int64_t dada_dbud_send_block (dada_client_t* client, void* data, 
                         uint64_t data_size, uint64_t block_id)
{

  dada_dbud_t * dbud = (dada_dbud_t *) client->context;
  dada_ib_datagram_t * ib_dg = dbud->ib_dg;

  // receive the requisite number of packets, ignoring sequence number
  uint64_t ipkt = 0;
  uint64_t npkt = data_size / IB_PAYLOAD;

  multilog (client->log, LOG_INFO, "send_block: will transmit %"PRIu64" x %d byte packets\n", npkt, IB_PAYLOAD);

  for (ipkt=0; ipkt < npkt; ipkt++)
  {
    // register the "send"
    if (dbud->verbose > 1)
      multilog (client->log, LOG_INFO, "send_block: dada_ib_dg_post_send (%p, %p [MR])\n", ib_dg, ib_dg->bufs[0], dbud->remote->qpn);
    dada_ib_dg_post_send (ib_dg, ib_dg->bufs[0], dbud->remote->qpn);

    // now wait for the CQE on the CC/CQ
    if (dbud->verbose > 1)
      multilog (client->log, LOG_INFO, "send_block: dada_ib_dg_wait_send (%p, bufs[%d])\n", ib_dg, 0);
    dada_ib_dg_wait_send (ib_dg, ib_dg->bufs[0]);
  }

  return (int64_t) data_size;
}


/*! Function that closes the data file */
int dada_dbud_close (dada_client_t* client, uint64_t bytes_written)
{

  dada_dbud_t * dbud = (dada_dbud_t *) client->context;

  dada_ib_datagram_t * ib_dg = dbud->ib_dg;

  if (dbud->verbose)
    multilog (client->log, LOG_INFO, "dada_dbud_close()\n");

  return 0;
}


/*! Function that opens the data transfer target */
int dada_dbud_open (dada_client_t* client)
{

  assert (client != 0);

  dada_dbud_t * dbud = (dada_dbud_t *) client->context;

  dada_ib_datagram_t * ib_dg = dbud->ib_dg;

  if (dbud->verbose)
    multilog(client->log, LOG_INFO, "dada_dbud_open()\n");

  if (dbud->verbose)
    multilog(client->log, LOG_INFO, "dada_dbud_open() returns\n");

  return 0;
}

int main (int argc, char **argv)
{

  /* IB DB configuration */
  dada_dbud_t dbud = DADA_IBDB_INIT;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Secondary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* port on which to listen for incoming connections */
  int port = DADA_DEFAULT_IBDB_PORT;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  /* Quit flag */
  char quit = 0;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int cpu_core = -1;

  // LID IB for remote port
  //uint16_t dest_lid = 0;

  // GID for remote IB port
  ///char * gid;

  //int qpn = 0;

  int arg = 0;

  int sl = 0;

  while ((arg=getopt(argc,argv,"b:hk:p:sv")) != -1)
  {
    switch (arg) {

    case 'b':
      cpu_core = atoi(optarg);
      break;

    //case 'g':
    //  gid = strdup(optarg);
    //  break;

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr,"dada_dbud: could not parse key from %s\n",optarg);
        return EXIT_FAILURE;
      }
      break;

    //case 'l':
    //  if (sscanf (optarg, "%"PRIu16, &dest_lid) != 1)
    //  {
    //    fprintf (stderr, "dada_dbud: could not parse dest_lid from %s\n", optarg);
    //    return EXIT_FAILURE;
    //  }
    //  break;
      
    //case 'q':
    //  qpn = atoi (optarg);
    //  break;

    case 'p':
      port = atoi (optarg);
      break;

    case 's':
      quit = 1;
      break;

    case 'v':
      verbose++;
      break;
      
    default:
      usage ();
      return 0;
      
    }
  }

  if (argc - optind != 1)
  {
    fprintf (stderr, "ERROR: expected 1 argument\n");
    usage ();
    return EXIT_FAILURE;
  }

  char * dest_host = argv[optind];

  if (cpu_core >= 0)
    dada_bind_thread_to_core(cpu_core);

  log = multilog_open ("dada_dbud", daemon);

  if (daemon) {
    be_a_daemon ();
    multilog_serve (log, DADA_DEFAULT_IBDB_LOG);
  }
  else
    multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_read (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  client->log = log;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function     = dada_dbud_open;
  client->io_function       = dada_dbud_send;
  client->io_block_function = dada_dbud_send_block;
  client->close_function    = dada_dbud_close;
  client->direction         = dada_client_reader;

  client->context = &dbud;

  // initialize IB resources
  dbud.verbose = verbose;
  dbud.port = port;

  // connection information for remote sender
  dada_ib_datagram_dest_t * local;

  char gid[30];

  // create the Infiniband verbs struct
  dbud.ib_dg = dada_ib_dg_create (1, log);
  if (dbud.ib_dg)
  {
    dbud.ib_dg->port = 54321;
    dbud.ib_dg->ib_port = 1;
    //dbud.ib_dg->dest_lid = dest_lid;
    //dbud.ib_dg->dest_qpn = qpn;
    dbud.ib_dg->queue_depth = 100;

    // intialize the IB resources
    multilog(log, LOG_INFO, "main: dada_ib_dg_init()\n");
    if (dada_ib_dg_init (dbud.ib_dg) < 0)
    {
      multilog(log, LOG_ERR, "main: dada_ib_dg_init failed\n");
      quit = 1;
    }
    else
    {
      multilog(log, LOG_INFO, "main: post_recv for no reason!\n");
      dada_ib_dg_post_recv (dbud.ib_dg, dbud.ib_dg->bufs[0]);

      multilog (log, LOG_INFO, "main: ibv_req_notify_cq()\n");
      if (ibv_req_notify_cq(dbud.ib_dg->cq, 0))
      {
        multilog(log, LOG_ERR, "main: ibv_req_notify_cq dbud.ib_dg->cq failed\n");
        return EXIT_FAILURE;
      }

      // get the local infiniband port information
      multilog(log, LOG_INFO, "main: dada_ib_dg_get_local_port()\n");
      local = dada_ib_dg_get_local_port (dbud.ib_dg);

      inet_ntop(AF_INET6, &local->gid, gid, sizeof gid);
      multilog(log, LOG_INFO, "local address: LID 0x%04x, QPN 0x%06x, PSN 0x%06x, GID %s\n",
                local->lid, local->qpn, local->psn, gid);

      // exchange local information with server/receiver to get remote info
      multilog(log, LOG_INFO, "main: dada_ib_dg_client_exch_dest()\n");
      dbud.remote = dada_ib_dg_client_exch_dest (dest_host, dbud.ib_dg->port, local);

      inet_ntop(AF_INET6, &(dbud.remote->gid), gid, sizeof gid);
      multilog(log, LOG_INFO, "remote address: LID 0x%04x, QPN 0x%06x, PSN 0x%06x, GID %s\n",
               dbud.remote->lid, dbud.remote->qpn, dbud.remote->psn, gid);


      // intialize the IB resources
      multilog(log, LOG_INFO, "main: dada_ib_dg_activate()\n");
      if (dada_ib_dg_activate (dbud.ib_dg, local, dbud.remote, -1, sl) < 0)
      {
        multilog(log, LOG_ERR, "main: dada_ib_dg_init failed\n");
        quit = 1;
      }
    }
  }
  else
  {
    multilog(log, LOG_ERR, "main: dada_ib_dg_create failed()\n");
    client->quit = 1;
  }

  while (!client->quit) 
  {
    if (dada_client_read (client) < 0)
    {
      multilog (log, LOG_ERR, "Error during transfer\n");
      quit = 1;
    }

    if (verbose)
      multilog (client->log, LOG_INFO, "main: dada_ib_disconnect()\n");
    if (dada_ib_dg_disconnect(dbud.ib_dg) < 0)
    {
      multilog(client->log, LOG_ERR, "dada_ib_disconnect failed\n");
    }

    if (verbose)
      multilog (log, LOG_INFO, "main: dada_hdu_unlock_read()\n");
    if (dada_hdu_unlock_read(hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      quit = 1;
    }

    if (quit)
      client->quit = 1;
    else
    {
      if (dada_hdu_lock_read(hdu) < 0)
      {
        multilog (log, LOG_ERR, "could not lock read on hdu\n");
        return EXIT_FAILURE;
      }
    }
  }

  if (dbud.ib_dg)
    dada_ib_dg_destroy (dbud.ib_dg);

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;


  return EXIT_SUCCESS;
}

