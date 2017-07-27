/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#ifndef __BPSR_CT_H
#define __BPSR_CT_H

#include <inttypes.h>

#include "multilog.h"
#include "ascii_header.h"

#include "bpsr_def.h"

#define IDLE 0
#define PREPARED 1
#define EMPTY 2
#define FULL 3
#define FINISHED 4
#define ERROR 5

// Beam Former cornerturn connection info
typedef struct {

  char host[64]; 

  int port;

  unsigned isend;

  unsigned irecv;

  unsigned recv_offset;

  unsigned send_offset;

  unsigned atomic_size;

} bpsr_conn_t;

typedef struct {

  // number of TCP connections
  unsigned int nconn;

  // cornerturn information
  bpsr_conn_t * conn_info;

  unsigned int nsend;

  unsigned int nrecv;

  uint64_t send_block_size;

  uint64_t send_resolution;

  uint64_t atomic_size;

  uint64_t recv_block_size;

  uint64_t recv_resolution;

  int baseport;

} bpsr_ct_t;

typedef struct {

  // multilog inteface
  multilog_t * log;

  // file descriptors for each socket connection
  int * fds;

  // cornerturn info
  bpsr_ct_t ct;

  unsigned verbose;

  // quit signal for exiting after 1 observation
  int quit;

  // flag to indicate the direction of the cornerturn
  char forward_cornerturn;

  char obs_ending;

} bpsr_ct_send_t;

#define BPSR_CT_SEND_INIT { 0, 0, 0, 0, 0, 0, 0 }

typedef struct bpsr_conn_recv {

  bpsr_conn_t * conn_info;

  struct bpsr_ct_recv * ctx;

} bpsr_conn_recv_t;

typedef struct bpsr_ct_recv {

  // multilog inteface
  multilog_t * log;

  // cornerturn information
  bpsr_conn_t * conn_info;

  // for launching receiving threads
  bpsr_conn_recv_t * ctxs;
  
  // file descriptors for each socket connection
  int * fds;

  // cornerturn info
  bpsr_ct_t ct;
  
  unsigned verbose;

  // quit signal for exiting after 1 observation
  int quit;

  // flag to indicate the direction of the cornerturn
  char forward_cornerturn;

  char obs_ending;

  // headers for each socket connection
  char ** headers;

  pthread_t * threads;

  // mutex for control
  pthread_mutex_t mutex;

  pthread_cond_t cond;

  int * states;

  uint64_t * bytes;

  char * recv_block;

  size_t recv_bufsz;

  void ** results;

} bpsr_ct_recv_t;

bpsr_conn_t * bpsr_setup_send (const char * config_file, bpsr_ct_t * ctx, unsigned send_id);
bpsr_conn_t * bpsr_setup_recv (const char * config_file, bpsr_ct_t * ctx, unsigned recv_id);

#endif //  BPSR_CT_H
