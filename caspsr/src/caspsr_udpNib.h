/***************************************************************************
 *  
 *    Copyright (C) 2009 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>
#include <assert.h>
#include <netinet/in.h>
#include <signal.h>

#include "futils.h"
#include "dada_hdu.h"
#include "dada_pwc_main.h"
#include "multilog.h"
#include "ipcio.h"
#include "ascii_header.h"

#include "caspsr_def.h"
#include "caspsr_udp.h"
#include "caspsr_rdma.h"

/* Number of UDP packets to be recived for a called to buffer_function */
#define NUMUDPPACKETS 2000
#define NOTRECORDING 0
#define RECORDING 1
#define CASPSR_DEFAULT_UDPNNIC_PORT 33108
#define CASPSR_DISTRIB_CONTROL_PORT 33440

typedef struct {

  multilog_t* log;            /* logging facility */
  char *   host;              /* destination hostname */
  int      port;              /* comms setup port */
  char *   buffer;            /* memory buffer */
  uint64_t size;              /* size of the memory array */
  uint64_t w_total;           /* total bytes to send in an xfer */
  uint64_t w_count;           /* total bytes written */
  uint64_t r_count;           /* total bytes read */
  int      fd;                /* outgoing file descriptor */
  struct sockaddr_in dagram;  /* UDP socket struct */
  int      verbose;           /* verbosity flag */
  int      ib_message_size;   /* IBV message size */
  int      ib_rx_depth;       /* IBV rx_depth */
  int      ib_use_cq;         /* IBV flag for use completion queue */
  enum ibv_mtu ib_mtu;        /* IBV MTU to use [2048 best]*/
  int      ib_sl;             /* IBV service level */ 

} udpNib_sender_t;

typedef struct {

  multilog_t* log;
  int      verbose;          /* verbosity flag */
  int      fd;               /* incoming UDP socket */
  int      udp_port;         /* port to receive UDP data */
  int      control_port;     /* port to receive control commands */
  char *   interface;        /* NIC/Interface to open socket on */

  /* senders */
  unsigned n_senders;        /* number of senders */
  unsigned sender_i;         /* current sender [for receiving thread] */
  udpNib_sender_t ** senders;
  unsigned i_distrib;        /* number of this distributor */
  unsigned n_distrib;        /* number of distributors */

  /* packets per "transfer" */
  uint64_t packets_this_xfer;
  uint64_t packets_per_xfer;
  uint64_t packets_to_append;
  char  *  zeroed_packet;

  /* Packet and byte statistics */
  stats_t * packets;
  stats_t * bytes;

  /* housekeeping */
  int      clamped_output_rate;
  int64_t  bytes_to_acquire;
  uint64_t ch_id;
  uint64_t next_seq;
  struct   timeval timeout; 
  uint64_t n_sleeps;
  uint64_t ooo_packets;
  uint64_t ooo_ch_ids;
  uint64_t r_bytes;
  uint64_t r_packs;
  int      send_core;
  int      recv_core;
  uint64_t send_sleeps;
  unsigned receive_only;
  int      packet_reset;

  /* stats reporting */
  double   mb_rcv_ps;
  double   mb_drp_ps;
  double   mb_snd_ps;
  double   mb_free;
  double   mb_total;

} udpNib_t;

int udpNib_initialize_senders (udpNib_t * ctx);
void udpNib_reset_senders (udpNib_t * ctx);
int udpNib_dealloc_senders(udpNib_t * ctx);
int udpNib_reset_buffers(udpNib_t * ctx);

time_t udpNib_start_function (udpNib_t * ctx);
int udpNib_new_sender(udpNib_t * ctx);
int udpNib_stop_function (udpNib_t* ctx);

void * receiving_thread(void * arg);
void * sending_thread(void * arg);
void control_thread(void * arg);
void plotting_thread(void * arg);
void stats_thread(void * arg);

void usage();
void signal_handler (int signalValue); 

