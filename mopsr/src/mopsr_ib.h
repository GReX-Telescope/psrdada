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

#ifdef USE_SEGMENTS
// Cornerturn segment for more efficient memory registration by IB
typedef struct {

  uint64_t local_offset;

  uint64_t local_length;

  uint64_t remote_offset;

  uint64_t remote_length;

} mopsr_ct_segment_t;
#endif

// Beam Former cornerturn connection info
typedef struct {

  dada_ib_cm_t * ib_cm;

  char host[64]; 

  char src_addr[64];

  char dst_addr[64];

  int port;

  int pfb;

  int chan_first;

  int chan_last;

  int nchan;

  int npfb;

  int ant_first;

  int ant_last;

  int nant;

#ifdef USE_SEGMENTS
  unsigned nsegments;

  mopsr_ct_segment_t * segments;
#endif

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

  // byte offsets for each [conn][chan]
  uint64_t ** in_offsets;
  uint64_t ** out_offsets;

} mopsr_bf_ib_t;

#define MOPSR_IB_INIT { 0, 0, 0, 0, 0, 0, 0, "", 0, 0 }

mopsr_bf_conn_t * mopsr_setup_cornerturn_send (const char * config_file, mopsr_bf_ib_t * ctx, unsigned int send_id);
mopsr_bf_conn_t * mopsr_setup_cornerturn_recv (const char * config_file, mopsr_bf_ib_t * ctx, unsigned int channel);

// Beam Processor cornerturn connection info
typedef struct {

  dada_ib_cm_t * ib_cm;

  char host[64];

  char src_addr[64];

  char dst_addr[64];

  int port;

  // sender limits
  unsigned chan_first;
  unsigned chan_last;
  unsigned nchan;

  unsigned isend;
  unsigned nsend;

  // receiver limits
  unsigned beam_first;
  unsigned beam_last;
  unsigned nbeam;
  unsigned irecv;

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
  unsigned nbeam;

  // total nuber of channels [e.g. 320]
  unsigned nchan;

  // number of IB connections
  unsigned nconn;

  // number of bytes per sample
  unsigned nbyte;

  // number of dimensions (2 == complex)
  unsigned ndim;

  // number of polarisations 
  unsigned npol;

  // number of bits per sample
  unsigned nbit;

  unsigned verbose;

  char * header;

  // quit signal for exiting after 1 observation
  int quit;

  char obs_ending;

  // byte offsets for each [conn][ibeam]
  uint64_t ** in_offsets;
  uint64_t ** out_offsets;

} mopsr_bp_ib_t;

#define MOPSR_BP_IB_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "", 0, 0, 0, 0 }

int mopsr_setup_bp_read_config (mopsr_bp_ib_t * ctx, const char * config_file, char * config);
mopsr_bp_conn_t * mopsr_setup_bp_cornerturn_send (const char * config_file, mopsr_bp_ib_t * ctx, unsigned int send_id);
mopsr_bp_conn_t * mopsr_setup_bp_cornerturn_recv (const char * config_file, mopsr_bp_ib_t * ctx, unsigned int recv_id);

#endif //  MOPSR_IB_H
