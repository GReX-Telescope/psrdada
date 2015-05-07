/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#ifndef __MOPSR_IB_H
#define __MOPSR_IB_H

#include "dada_ib.h"
#include "ascii_header.h"

#include "mopsr_def.h"

// Beam Former cornerturn connection info
typedef struct {

  dada_ib_cm_t * ib_cm;

  char host[64]; 

  char ib_host[64]; 

  int port;

  int chan;

  int ant_first;

  int ant_last;

} mopsr_bf_conn_t;

typedef struct {

  // multilog inteface
  multilog_t * log;

  // cornerturn information
  mopsr_bf_conn_t * conn_info;

  // IB Connection Managers
  dada_ib_cm_t ** ib_cms;

  // total number of antenna [e.g. 352]
  unsigned int nant;

  // total number of input channels [e.g. 39]
  unsigned int nchan;

  // number of IB connections
  unsigned int nconn;

  unsigned verbose;

  char * header;

  // quit signal for exiting after 1 observation
  int quit;

  char obs_ending;

} mopsr_bf_ib_t;

#define MOPSR_IB_INIT { 0, 0, 0, 0, 0, 0, 0, "", 0, 0 }

mopsr_bf_conn_t * mopsr_setup_cornerturn_send (const char * config_file, mopsr_bf_ib_t * ctx, unsigned int send_id);
mopsr_bf_conn_t * mopsr_setup_cornerturn_recv (const char * config_file, mopsr_bf_ib_t * ctx, unsigned int channel);

// Beam Processor cornerturn connection info
typedef struct {

  dada_ib_cm_t * ib_cm;

  char host[64];

  char ib_host[64];

  int port;

  // sender limits
  unsigned chan_first;
  unsigned chan_last;

  // receiver limits
  unsigned beam_first;
  unsigned beam_last;

} mopsr_bp_conn_t;

typedef struct {

  // multilog inteface
  multilog_t * log;

  // cornerturn information
  mopsr_bp_conn_t * conn_info;

  // IB Connection Managers
  dada_ib_cm_t ** ib_cms;

  // total number of senders
  unsigned nsend;

  // total number of beams [e.g. 352]
  unsigned nbeam_send;

  unsigned nbeam_recv;

  unsigned nchan_send;

  unsigned nchan_recv;

  // number of IB connections
  unsigned nconn;

  // number of bytes per sample
  unsigned nbyte;

  // number of dimensions (2 == complex)
  unsigned ndim;

  unsigned verbose;

  char * header;

  // quit signal for exiting after 1 observation
  int quit;

  char obs_ending;

} mopsr_bp_ib_t;

#define MOPSR_BP_IB_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "", 0, 0 }

int mopsr_setup_bp_read_config (mopsr_bp_ib_t * ctx, const char * config_file, char * config);
mopsr_bp_conn_t * mopsr_setup_bp_cornerturn_send (const char * config_file, mopsr_bp_ib_t * ctx, unsigned int send_id);
mopsr_bp_conn_t * mopsr_setup_bp_cornerturn_recv (const char * config_file, mopsr_bp_ib_t * ctx, unsigned int recv_id);

#endif //  MOPSR_IB_H
