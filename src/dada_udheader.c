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
     "Usage: dada_udheader [options]\n"
     " -b <core>     bind process to CPU core\n"
     " -h            print this usage help\n"
     " -i <port>     IB port to use\n"
     " -s <level>    IB service level [default 0, range 0-15]\n"
     " -t <port>     TCP port to negotiate IB handshake\n");
}


typedef struct dada_udheader {
  
  // port to listen on for connections 
  unsigned port;

  // verbose messages
  char verbose;

  // flag for active RDMA connection
  unsigned connected;

  void * recv_buf;

  size_t recv_bufsz;
    
  char * header;
  
  /* current observation id, as defined by OBS_ID attribute */
  char obs_id [DADA_OBS_ID_MAXLEN];

  // Infiniband Connection Manager
  dada_ib_datagram_t * ib_dg;

} dada_udheader_t;

#define DADA_IBDB_INIT { 0, 0, 0, 0, 0, "", "", 0 }

int main (int argc, char **argv)
{

  /* IB DB configuration */
  dada_udheader_t udheader = DADA_IBDB_INIT;

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

  int cpu_core = -1;

  int service_level = 0;

  int ib_port = 1;

  int tcp_port = 54321;

  int arg = 0;

  while ((arg=getopt(argc,argv,"b:hp:s:v")) != -1)
  {
    switch (arg) {

    case 'b':
      cpu_core = atoi(optarg);
      break;

    case 'i':
      ib_port = atoi(optarg);
      break;

    case 'p':
      port = atoi (optarg);
      break;

    case 's':
      service_level = atoi (optarg);
      break;

    case 't':
      tcp_port = atoi(optarg);
      break;

    case 'v':
      verbose++;
      break;
      
    default:
      usage ();
      return 0;
      
    }
  }

  if (cpu_core >= 0)
    dada_bind_thread_to_core(cpu_core);

  log = multilog_open ("dada_udheader", daemon);

  if (daemon) {
    be_a_daemon ();
    multilog_serve (log, DADA_DEFAULT_IBDB_LOG);
  }
  else
    multilog_add (log, stderr);

  // initialize IB resources
  udheader.verbose = verbose;
  udheader.port = port;

  // create the Infiniband verbs struct
  int nbufs = 75;
  udheader.ib_dg = dada_ib_dg_create (nbufs, log);

  // connection information for remote sender
  dada_ib_datagram_dest_t * remote;
  dada_ib_datagram_dest_t * local;

  // used to select something...
  int gidx = 0;
  char gid[30];

  if (udheader.ib_dg)
  {
    udheader.ib_dg->port = tcp_port;
    udheader.ib_dg->ib_port = ib_port;
    udheader.ib_dg->queue_depth = 1000;

    // intialize the IB resources
    multilog(log, LOG_INFO, "main: dada_ib_dg_init()\n");
    if (dada_ib_dg_init (udheader.ib_dg) < 0)
    {
      multilog(log, LOG_ERR, "main: dada_ib_dg_init failed\n");
      quit = 1;
    }
    else
    {
      // post receive before doing anything
      multilog(log, LOG_INFO, "main: dada_ib_dg_post_recvs(%p, %p, %d)\n", udheader.ib_dg, udheader.ib_dg->bufs, nbufs);
      dada_ib_dg_post_recvs (udheader.ib_dg, udheader.ib_dg->bufs, nbufs);

      //multilog(log, LOG_INFO, "main: dada_ib_dg_post_recv(%p, %p)\n", udheader.ib_dg, udheader.ib_dg->bufs[0]);
      //dada_ib_dg_post_recv (udheader.ib_dg, udheader.ib_dg->bufs[0]);

      // get the local infiniband port information
      multilog(log, LOG_INFO, "main: dada_ib_dg_get_local_port()\n");
      local = dada_ib_dg_get_local_port (udheader.ib_dg);

      inet_ntop(AF_INET6, &local->gid, gid, sizeof gid);
      multilog(log, LOG_INFO, "local address: LID 0x%04x, QPN 0x%06x, PSN 0x%06x, GID %s\n", local->lid, local->qpn, local->psn, gid);

      // exchange local information with client/sender to get remote info
      multilog(log, LOG_INFO, "main: dada_ib_dg_server_exch_dest()\n");
      remote = dada_ib_dg_server_exch_dest (udheader.ib_dg, udheader.ib_dg->ib_port, udheader.ib_dg->port, service_level, local, gidx);

      inet_ntop(AF_INET6, &remote->gid, gid, sizeof gid);
      multilog(log, LOG_INFO, "remote address: LID 0x%04x, QPN 0x%06x, PSN 0x%06x, GID %s\n", remote->lid, remote->qpn, remote->psn, gid);
    }
  }
  else
  {
    multilog(log, LOG_ERR, "main: dada_ib_dg_create failed()\n");
    quit = 1;
  }

  unsigned char * buffer;
  uint64_t seq_no = 0;
  uint64_t prev_seq_no = 0;

  struct ibv_cq *ev_cq;
  void          *ev_ctx;
  struct ibv_wc  wcs[nbufs];
  unsigned int num_cq_received = 0;

  int i, j;

  while (!quit)
  {
    // get a completion event
    if (verbose > 1)
      multilog(log, LOG_INFO, "main: ibv_get_cq_event()\n");
    if (ibv_get_cq_event(udheader.ib_dg->channel, &ev_cq, &ev_ctx))
    {
      multilog(log, LOG_ERR, "main: ibv_get_cq_event failed: %s\n", strerror(errno));
      quit = 1;
      break;
    }

    num_cq_received++;

    // confirm the ev_cq matches the expected cq
    if (verbose > 1)
      multilog(log, LOG_INFO, "main: testing completion queue matches\n");
    if (ev_cq != udheader.ib_dg->cq)
    {
      multilog(log, LOG_ERR, "main: event cq did not match expected cq\n");
      quit = 1;
      break;
    }

    // reset notification for CQEs
    if (verbose > 1)
      multilog(log, LOG_INFO, "main: ibv_req_notify_cq()\n");
    if (ibv_req_notify_cq(udheader.ib_dg->cq, 0))
    {
      multilog(log, LOG_ERR, "main: ibv_req_notify_cq() failed\n");
      quit = 1;
      break;
    }

    // see how many completion events are waiting for us
    if (verbose > 1)
      multilog(log, LOG_INFO, "main: ibv_poll_cq on %d bufs\n", nbufs);
    int ne = ibv_poll_cq(udheader.ib_dg->cq, nbufs, wcs);
    if (ne < 0)
    {
      multilog(log, LOG_ERR, "main: ibv_poll_cq() failed: ne=%d\n", ne);
      quit = 1;
      break;
    }

    // for each completion event, check the status
    for (i = 0; i < ne; i++) 
    {
      if (wcs[i].status != IBV_WC_SUCCESS)
      {
        multilog(log, LOG_WARNING, "main: wcs[%d].status != IBV_WC_SUCCESS "
                                   "[wc.status=%s, wc.wr_id=%"PRIu64"\n",
                                    i, ibv_wc_status_str(wcs[i].status));
      }

      // work request ID
      j = wcs[i].wr_id - 100;

      if (verbose > 1)
        multilog(log, LOG_INFO, "main: buffer %d recieved data\n", j);

      // setup pointer to data (inc 40 bytes due to some IB header thing...
      buffer = udheader.ib_dg->bufs[j]->buffer + 40;

      // decode seq number of this packet
      dada_ib_dg_decode_header (buffer, &seq_no);

      if (verbose > 1)
        multilog(log, LOG_INFO, "main: seq_no=%"PRIu64"\n", seq_no);

      //if (prev_seq_no && (prev_seq_no + 1 != seq_no))
      //  multilog(log, LOG_ERR, "main: seq_no mismatch: seq=%"PRIu64" prev=%"PRIu64", diff=%"PRIu64"\n", seq_no, prev_seq_no, (seq_no - prev_seq_no));

      prev_seq_no = seq_no;

      // HERE is where we would do something with the received data

      // post receive on this buffer
      if (verbose > 1)
        multilog(log, LOG_INFO, "main: dada_ib_dg_post_recv(%p, %p) [%d]\n", udheader.ib_dg, udheader.ib_dg->bufs[j], j);
      dada_ib_dg_post_recv (udheader.ib_dg, udheader.ib_dg->bufs[j]);
    }

    if (num_cq_received)
    {
      if (verbose > 1)
        multilog(log, LOG_INFO, "main: ibv_ack_cq_events() for %d events\n", num_cq_received);
      ibv_ack_cq_events(udheader.ib_dg->cq, num_cq_received);
    }
    num_cq_received = 0;
  }

  if (verbose)
    multilog (log, LOG_INFO, "main: dada_ib_disconnect()\n");
  if (dada_ib_dg_disconnect(udheader.ib_dg) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_disconnect failed\n");
  }

  if (udheader.ib_dg)
    dada_ib_dg_destroy (udheader.ib_dg);

  return EXIT_SUCCESS;
}

