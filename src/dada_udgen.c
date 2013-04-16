/***************************************************************************
 *  
 *    Copyright (C) 2012 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

/*
 * Infiniband UD packet generator
 * 
 */

#include <stdlib.h> 
#include <stdio.h> 
#include <unistd.h>
#include <errno.h> 
#include <string.h> 
#include <sys/types.h> 
#include <netinet/in.h> 
#include <netdb.h> 
#include <sys/socket.h> 
#include <sys/wait.h> 
#include <sys/timeb.h> 
#include <math.h>
#include <pthread.h>
#include <assert.h>

#include "multilog.h"
#include "dada_ib_datagram.h"

#include "arch.h"
#include "Statistics.h"
#include "RealTime.h"
#include "StopWatch.h"

#define DADA_UDGEN_XMIT_TIME 5
#define DADA_UDGEN_DATA_RATE 64

void signal_handler(int signalValue);

void usage (void) 
{
  fprintf(stdout,
    "Usage: dada_udgen [options] hostname\n"
    "\t-h            print this help text\n"
    "\t-i <port>     IB port to use [default %d]\n"
    "\t-n <secs>     number of seconds to transmit [default %d]\n"
    "\t-r <rate>     transmit at rate MB/s [default %d]\n"
    "\t-s <level>    IB service level [default 0, range 0-15]\n"
    "\t-t <port>     TCP port to perform setup handshake [default %d]\n"
    "\t-v            verbose output\n"
    "\n\thostname    IP hostname of remote IB device\n\n",
    DADA_IB_DG_IB_PORT, DADA_UDGEN_XMIT_TIME, DADA_UDGEN_DATA_RATE, DADA_IB_DB_TCP_PORT
  );
}

int main(int argc, char *argv[])
{

  // number of microseconds between packets
  double sleep_time = 0;
 
  // verbosity
  int verbose = 0;

  // IB port
  int ib_port = DADA_IB_DG_IB_PORT;

  // tcp comms port for connection setup
  int tcp_port = DADA_IB_DB_TCP_PORT;

  // total time to transmit for
  uint64_t transmission_time = DADA_UDGEN_XMIT_TIME;   

  // DADA logger
  multilog_t *log = 0;

  // Infiniband Service Level
  int service_level = 0;

  // hostname to establish TCP connection 
  char * dest_host;

  // The generated signal arrays
  char packet[IB_PAYLOAD];

  // data rate
  unsigned int data_rate_mbytes = DADA_UDGEN_DATA_RATE;

  // number of packets to send every second
  uint64_t packets_ps = 0;

  // start of transmission
  time_t start_time;

  // end of transmission
  time_t end_time;

  // sequence number
  uint64_t seq_no = 0;

  int c;
  while ((c = getopt(argc, argv, "i:n:r:s:t:v")) != EOF) {
    switch(c) {

      case 'i':
        ib_port = atoi(optarg);
        break;

      case 'n':
        transmission_time = atoi(optarg);
        break;

      case 'r':
        data_rate_mbytes = atoi(optarg);
        break;

      case 's':
        service_level= atoi(optarg);
        break;

      case 't':
        tcp_port = atoi(optarg);
        break;

      case 'v':
        verbose = 1;
        break;

      default:
        usage();
        return EXIT_FAILURE;
        break;
    }
  }

  // check arguements
  if ((argc - optind) != 1)
  {
    fprintf(stderr,"no dest_host was specified\n");
    usage();
    return EXIT_FAILURE;
  }

  // destination host
  dest_host = (char *) argv[optind];

  signal(SIGINT, signal_handler);

  // use multilog to produce time-stamped log messages
  log = multilog_open ("leda_udpgen", 0);
  multilog_add (log, stderr);

  double data_rate = (double) data_rate_mbytes;

  data_rate *= 1024 * 1024;

  if (data_rate)
    multilog(log, LOG_INFO, "rate: %5.2f MB/s \n", data_rate/(1024*1024));
  else
    multilog(log, LOG_INFO, "rate: fast as possible\n");

  multilog(log, LOG_INFO, "sending to %s\n", dest_host);

  int nbufs = 20;
  char gid[30];

  // create memory required for 1 transmit buffer
  multilog(log, LOG_INFO, "main: dada_ib_dg_create(%d)\n", nbufs);
  dada_ib_datagram_t * ib_dg = dada_ib_dg_create (nbufs, log);
  if (!ib_dg)
  {
    multilog(log, LOG_ERR, "main: dada_ib_dg_create failed\n");
    return EXIT_FAILURE;
  }

  ib_dg->ib_port = ib_port;
  ib_dg->port = tcp_port;
  ib_dg->queue_depth = 100;

  // initialize IB resources
  if (dada_ib_dg_init (ib_dg) < 0)
  {
    multilog(log, LOG_ERR, "main: dada_ib_dg_init failed\n");
    dada_ib_dg_destroy (ib_dg);
    return EXIT_FAILURE;
  }

  dada_ib_datagram_dest_t * local;
  dada_ib_datagram_dest_t * remote;


  // get local IB port information
  local = dada_ib_dg_get_local_port (ib_dg);

  inet_ntop(AF_INET6, &local->gid, gid, sizeof gid);
  multilog(log, LOG_INFO, "local address: LID 0x%04x, QPN 0x%06x, PSN 0x%06x, GID %s\n",
            local->lid, local->qpn, local->psn, gid);

  // exchange local information with server/receiver to get remote info
  multilog(log, LOG_INFO, "main: dada_ib_dg_client_exch_dest()\n");
  remote = dada_ib_dg_client_exch_dest (dest_host, ib_dg->port, local);

  inet_ntop(AF_INET6, &(remote->gid), gid, sizeof gid);
  multilog(log, LOG_INFO, "remote address: LID 0x%04x, QPN 0x%06x, PSN 0x%06x, GID %s\n",
           remote->lid, remote->qpn, remote->psn, gid);

  // intialize the IB resources
  multilog(log, LOG_INFO, "main: dada_ib_dg_activate()\n");
  if (dada_ib_dg_activate (ib_dg, local, remote, -1, service_level) < 0)
  {
    multilog(log, LOG_ERR, "main: dada_ib_dg_init failed\n");
    dada_ib_dg_destroy (ib_dg);
    return EXIT_FAILURE;
  }

  uint64_t data_counter = 0;

  // initialise data rate timing library
  StopWatch wait_sw;
  RealTime_Initialise(1);
  StopWatch_Initialise(1);

  // If we have a desired data rate, then we need to set sleep time 
  if (data_rate > 0) 
  {
    packets_ps = floor(((double) data_rate) / ((double) IB_PAYLOAD));
    sleep_time = (1.0/packets_ps)*1000000.0;
  }

  multilog (log, LOG_INFO, "Packets/sec = %"PRIu64"\n",packets_ps);
  multilog (log, LOG_INFO, "sleep_time = %f\n",sleep_time);

  // seed the random number generator
  srand ( time(NULL) );

  uint64_t total_bytes_to_send = data_rate * transmission_time;

  // assume 10GbE speeds
  if (data_rate == 0)
    total_bytes_to_send = 1*1024*1024*1024 * transmission_time;

  size_t bytes_sent = 0;
  uint64_t total_bytes_sent = 0;

  uint64_t bytes_sent_thistime = 0;
  uint64_t prev_bytes_sent = 0;
  
  time_t current_time = time(0);
  time_t prev_time = time(0);

  multilog (log, LOG_INFO, "Total bytes to send = %"PRIu64"\n",total_bytes_to_send);
  multilog (log, LOG_INFO, "IB payload = %"PRIu64" bytes\n", IB_PAYLOAD);
  multilog (log, LOG_INFO, "IB data size = %"PRIu64" bytes\n", IB_DATAGRAM);
  multilog (log, LOG_INFO, "Wire Rate\t\tUseful Rate\tPacket\tSleep Time\n");

  unsigned int s_off = 0;
  unsigned char * buffer;
  seq_no = 0;
  uint64_t decoded_seq = 0;

  struct ibv_cq *ev_cq;
  void          *ev_ctx;
  struct ibv_wc  wcs[nbufs];
  unsigned int num_cq_received = 0;
  
  int i, j;

  // post the WR to send WQ
  for (i=0; i<nbufs; i++)
  {
    buffer = (unsigned char *) ib_dg->bufs[i]->buffer;
    dada_ib_dg_encode_header(buffer + 40, seq_no);
    seq_no ++;
  }

  dada_ib_dg_post_sends (ib_dg, ib_dg->bufs, nbufs, remote->qpn);

  bytes_sent += (nbufs * IB_DATAGRAM);
  total_bytes_sent += (nbufs * IB_PAYLOAD);
  data_counter += nbufs;

  int quit = 0;

  while (!quit && (total_bytes_sent < total_bytes_to_send))
  {

    // wait for a completion
    if (verbose > 1)
      multilog(log, LOG_INFO, "main: ibv_get_cq_event()\n");
    if (ibv_get_cq_event(ib_dg->channel, &ev_cq, &ev_ctx))
    {
      multilog(log, LOG_ERR, "main: ibv_get_cq_event failed: %s\n", strerror(errno));
      quit = 1;
      break; 
    }
    
    num_cq_received++;

    // confirm the ev_cq matches the expected cq
    if (verbose > 1)
      multilog(log, LOG_INFO, "main: testing completion queue matches\n");
    if (ev_cq != ib_dg->cq)
    {
      multilog(log, LOG_ERR, "main: event cq did not match expected cq\n");
      quit = 1;
      break;
    }

    // reset notification for CQEs
    if (verbose > 1)
      multilog(log, LOG_INFO, "main: ibv_req_notify_cq()\n");
    if (ibv_req_notify_cq(ib_dg->cq, 0))
    {
      multilog(log, LOG_ERR, "main: ibv_req_notify_cq() failed\n");
      quit = 1;
      break;
    }

    // see how many completion events are waiting for us
    if (verbose > 1)
      multilog(log, LOG_INFO, "main: ibv_poll_cq on %d bufs\n", nbufs);
    int ne = ibv_poll_cq(ib_dg->cq, nbufs, wcs);
    if (ne < 0)
    {
      multilog(log, LOG_ERR, "main: ibv_poll_cq() failed: ne=%d\n", ne);
      quit = 1;
      break;
    }

    // for each completion event, check the status
    for (i = 0; i < ne; i++)
    {
      if (data_rate)
        StopWatch_Start(&wait_sw);
      if (wcs[i].status != IBV_WC_SUCCESS)
      {
        multilog(log, LOG_WARNING, "main: wcs[%d].status != IBV_WC_SUCCESS "
                                   "[wc.status=%s, wc.wr_id=%"PRIu64"\n",
                                    i, ibv_wc_status_str(wcs[i].status));
      }

      // work request ID -> array index
      j = wcs[i].wr_id - 100;

      if (verbose > 1)
        multilog(log, LOG_INFO, "main: buffer %d recieved data\n", j);

      // setup pointer to data (inc 40 bytes due to some IB header thing...
      buffer = ib_dg->bufs[j]->buffer;

      // decode seq number of this packet
      dada_ib_dg_encode_header (buffer + 40, seq_no);

      if (verbose > 1)
        multilog(log, LOG_INFO, "main: seq_no=%"PRIu64"\n", seq_no);

      // post send on this buffer
      if (verbose > 1)
        multilog(log, LOG_INFO, "main: dada_ib_dg_post_send(%p, %p) [%d]\n", ib_dg, ib_dg->bufs[j], j);
      dada_ib_dg_post_send (ib_dg, ib_dg->bufs[j], remote->qpn);
    
      // This is how much useful data we actaully sent
      bytes_sent += IB_DATAGRAM;
      total_bytes_sent += IB_PAYLOAD;
      data_counter++;
      seq_no++;
    
      if (data_rate)
        StopWatch_Delay(&wait_sw, sleep_time);
    }

    if (num_cq_received)
    {
      if (verbose > 1)
        multilog(log, LOG_INFO, "main: ibv_ack_cq_events() for %d events\n", num_cq_received);
      ibv_ack_cq_events(ib_dg->cq, num_cq_received);
    }
    num_cq_received = 0;

    prev_time = current_time;
    current_time = time(0);
    
    if (prev_time != current_time) 
    {
      double useful_data_only = (double) IB_PAYLOAD;
      double complete_packet = (double) IB_DATAGRAM;

      double wire_ratio = complete_packet / complete_packet;
      double useful_ratio = useful_data_only / complete_packet;
        
      uint64_t bytes_per_second = total_bytes_sent - prev_bytes_sent;
      prev_bytes_sent = total_bytes_sent;
      double rate = ((double) bytes_per_second) / (1024*1024);

      double wire_rate = rate * wire_ratio;
      double useful_rate = rate * useful_ratio;
             
      multilog(log,LOG_INFO,"%5.2f MB/s  %5.2f MB/s  %"PRIu64
                            "  %5.2f, %"PRIu64"\n",
                            wire_rate, useful_rate, data_counter, sleep_time,
                            bytes_sent);
    }



  }

  uint64_t packets_sent = seq_no;

  multilog (log, LOG_INFO, "Sent %"PRIu64" bytes\n",total_bytes_sent);
  multilog (log, LOG_INFO, "Sent %"PRIu64" packets\n",packets_sent);

  dada_ib_dg_destroy (ib_dg);

  return 0;
}


void signal_handler(int signalValue) {
  exit(EXIT_SUCCESS);
}

