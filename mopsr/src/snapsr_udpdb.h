/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#ifndef __SNAPSR_UDPDB_H
#define __SNAPSR_UDPDB_H

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

#include "snapsr_udp.h"
#include "snapsr_def.h"

typedef struct {

  multilog_t      * log;
  int               verbose;              // verbosity flag 
  int               port;                 // port to receive UDP data 

  snapsr_sock_t    * sock;                 // UDP socket for data capture
  snapsr_udp_hdr_t   hdr;                  // decoded packet header
  char            * interface;            // IP Address to accept packets on 
  char            * pfb_id;               // Identifying string for source PFB

  uint64_t          start_byte;           // seq_byte of first byte for the block
  uint64_t          end_byte;             // seq_byte of first byte of final packet of the block
  uint64_t          obs_bytes;            // number of bytes received during observation

  uint64_t          pkt_start;            // start packet for the observation (can be non-zero)
  uint64_t          start_pkt;            // first packet in the block
  uint64_t          end_pkt;              // last packet in the block

  // packets
  unsigned          got_enough;           // flag for capture loop
  unsigned          capture_started;      // flag for start of UDP data
  unsigned          state;                // flag for start of UDP data
  unsigned          pkt_size;             // bytes used in packet (may be less if antenna ignored)
  uint64_t          packets_per_buffer;   // number of UDP packets per datablock buffer
  uint64_t          prev_seq;
  void *            zeroed_packet;        // packet for handling drops 

  /* Packet and byte statistics */
  stats_t         * packets;
  stats_t         * bytes;

  uint64_t          n_sleeps;
  uint64_t          timeouts;
  struct timeval    timeout; 

  uint64_t          bytes_per_second;

  char * header_file;

} snapsr_udpdb_t;

int snapsr_udpdb_init (snapsr_udpdb_t * ctx);
void snapsr_udpdb_reset (snapsr_udpdb_t * ctx);
int snapsr_udpdb_free (snapsr_udpdb_t * ctx);

int snapsr_udpdb_header_valid (dada_pwc_main_t* pwcm);
int snapsr_udpdb_error (dada_pwc_main_t* pwcm);
time_t snapsr_udpdb_start (dada_pwc_main_t * pwcm, time_t start_utc);

void * snapsr_udpdb_recv (dada_pwc_main_t * pwcm, int64_t * size);
int64_t snapsr_udpdb_recv_block (dada_pwc_main_t * pwcm, void * block,
                                uint64_t block_size, uint64_t block_id);
int snapsr_udpdb_stop (dada_pwc_main_t* pwcm);


void usage();
void signal_handler (int signalValue); 

void plot_thread (void * arg);

#endif
