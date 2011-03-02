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
#define CASPSR_DEFAULT_UDPNDB_PORT 33108

typedef struct {

  key_t  key;        /* shared memory key containing data stream */
  dada_hdu_t* hdu;   /* DADA Header plus Data unit */

} udpNdb_receiver_t;

typedef struct {

  multilog_t*       log;        // DADA logging interface
  int               verbose;    // verbosity flag 

  caspsr_sock_t *   sock;       // UDP socket for data capture
  int               port;       // port to receive UDP data 
  int               control_port;
  char *            interface;  // IP Address to accept packets on 

  // header information
  char            * obs_header;  // ascii header for this observation 
  char            * header;      // ascii header to write to all DBs
  int               header_size; // header size

  int64_t           obs_xfer;    // counter for the OBS_XFER header variable
  uint64_t          obs_offset;  // counter for the OBS_OFFSET header variable

  // datablocks
  unsigned      ihdu;
  unsigned      nhdus;
  dada_hdu_t ** hdus;
  uint64_t      hdu_bufsz;
  unsigned      hdu_open;        // if the current HDU is open
  unsigned      block_open;      // if the current data block element is open
  char        * current_buffer;  // pointer to current datablock buffer

  uint64_t     buffer_start_byte;
  uint64_t     buffer_end_byte;

  uint64_t     xfer_start_byte; // first byte of the current xfer
  uint64_t     xfer_end_byte;   // last byte of the current xfer

  /* packets per "transfer" */
  uint64_t    packets_per_xfer;   // number of UDP packets per xfer
  uint64_t    packets_per_buffer; // number of UDP packest per datablock buffer
  uint64_t    bytes_this_xfer;
  uint64_t    packets_this_xfer;
  char  *     zeroed_packet;
  unsigned    packet_reset;

  /* Packet and byte statistics */
  stats_t * packets;
  stats_t * bytes;

  uint64_t bytes_to_acquire;
  double mb_rcv_ps;
  double mb_drp_ps;
  double mb_free;
  double mb_total;
  uint64_t rcv_sleeps;


  uint64_t next_seq;        // Next packet we are expecting
  struct   timeval timeout; 

  uint64_t n_sleeps;
  uint64_t ooo_packets;

  unsigned recv_core;
  unsigned i_distrib;
  unsigned n_distrib;

} udpNdb_t;


int caspsr_udpNdb_init_receiver (udpNdb_t * ctx);
void caspsr_udpNnb_reset_receiver (udpNdb_t * ctx);
int caspsr_udpNdb_destroy_receiver (udpNdb_t * ctx);
int caspsr_udpNdb_close_hdu (udpNdb_t * ctx, uint64_t bytes_written);
int caspsr_udpNdb_open_hdu (udpNdb_t * ctx);
int caspsr_udpNdb_new_hdu (udpNdb_t * ctx);
int caspsr_udpNdb_open_buffer (udpNdb_t * ctx);
int caspsr_udpNdb_close_buffer (udpNdb_t * ctx, uint64_t bytes_written);
int caspsr_udpNdb_new_buffer (udpNdb_t * ctx);

time_t udpNdb_start_function (udpNdb_t * ctx);
int udpNdb_open_datablocks(udpNdb_t * ctx);
int udpNdb_close_datablocks(udpNdb_t * ctx);
int udpNdb_new_datablock(udpNdb_t * ctx);
int udpNdb_new_header(udpNdb_t * ctx);
int udpNdb_main_function (udpNdb_t * ctx, int64_t total_bytes);
int udpNdb_stop_function (udpNdb_t* ctx);


void usage();
void signal_handler (int signalValue); 
void stats_thread(void * arg);
void control_thread(void * arg);
