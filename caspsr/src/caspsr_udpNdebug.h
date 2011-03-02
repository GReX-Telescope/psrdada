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

/* Number of UDP packets to be recived for a called to buffer_function */
#define NUMUDPPACKETS 2000
#define NOTRECORDING 0
#define RECORDING 1
#define CASPSR_DEFAULT_UDPNDEBUG_PORT 33108

typedef struct {

  multilog_t* log;
  int      verbose;          /* verbosity flag */
  int      fd;               /* incoming UDP socket */
  int      udp_port;         /* port to receive UDP data */
  int      control_port;     /* port to receive control commands */
  char *   interface;        /* NIC/Interface to open socket on */
  char *   buffer;          

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

} udpNdebug_t;

time_t  udpNdebug_start_function (udpNdebug_t * ctx);
int     udpNdebug_stop_function (udpNdebug_t* ctx);

void *  receiving_thread(void * arg);
void    stats_thread(void * arg);

void    usage();
void    signal_handler (int signalValue); 

