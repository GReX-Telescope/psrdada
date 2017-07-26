/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#ifndef __MOPSR_UDPDB_H
#define __MOPSR_UDPDB_H

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

#include "mopsr_def.h"
#include "mopsr_udp.h"

typedef struct {

  multilog_t      * log;
  int               verbose;              // verbosity flag 

  unsigned int      ninputs;
  char           ** interfaces;
  int             * ports;
  mopsr_sock_t   ** socks;                
  char            * enough;               // flag for enough data

  char            * pfb_id;

  mopsr_hdr_t       hdr;                  // decoded packet header
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
  void *            zeroed_packet;        // packet for handling drops 
    
  // Packet and byte statistics
  stats_t         * packets;
  stats_t         * bytes;
  char            * mdir;                 // directory for monitoring files
  char            * last_block;           // pointer to last full block of valid data

  uint64_t          n_sleeps;
  uint64_t          timeouts;
  struct timeval    timeout; 
  int               core;

  uint64_t          bytes_per_second;

  pthread_mutex_t   mutex;


} mopsr_udpdb_dual_t;

int mopsr_udpdb_dual_init (mopsr_udpdb_dual_t * ctx);
void mopsr_udpdb_dual_reset (mopsr_udpdb_dual_t * ctx);
int mopsr_udpdb_dual_free (mopsr_udpdb_dual_t * ctx);
int mopsr_udpdb_parse_input (mopsr_udpdb_dual_t * ctx, char * optarg);

int mopsr_udpdb_dual_header_valid (dada_pwc_main_t* pwcm);
int mopsr_udpdb_dual_error (dada_pwc_main_t* pwcm);
time_t mopsr_udpdb_dual_start (dada_pwc_main_t * pwcm, time_t start_utc);
void * mopsr_udpdb_dual_recv (dada_pwc_main_t * pwcm, int64_t * size);
int64_t mopsr_udpdb_dual_recv_block (dada_pwc_main_t * pwcm, void * block,
                                uint64_t block_size, uint64_t block_id);
int mopsr_udpdb_dual_stop (dada_pwc_main_t* pwcm);

void usage();
void signal_handler (int signalValue); 

void mon_thread (void * arg);
//void plot_thread (void * arg);

//void stats_thread(void * arg);
//void control_thread(void * arg);

#endif
