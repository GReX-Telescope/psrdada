/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_def.h"
#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_cuda.h"

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOPSR_DBDELAYDB_DEFAULT_NTAPS 5

typedef struct {

  int device;

  unsigned verbose;

  uint64_t time_msec;

  uint64_t block_size;

  void * work_buffer;

  size_t work_buffer_size;

  // number of taps in filter - must be odd
  unsigned ntaps;

  // FIR filter coefficients
  float * filter;

  size_t filter_size; 

  // GPU pointers
  cudaStream_t stream;

  // GPU buffers for input and output of delayed data
  size_t d_buffer_size;
  void * d_in;
  void * d_out;

  // GPU buffer for saving overlapping between DADA buffers
  size_t d_overlap_size;
  void * d_overlap;

  // delays for each channel and antenna
  size_t delays_size;
  float * h_delays;
  void  * d_delays;

  // byte offsets for managing the sample delays
  int64_t ** d_byte_offsets;
  int64_t ** h_byte_offsets;
  int64_t ** sample_offsets;

  multilog_t * log;

  dada_hdu_t * out_hdu;

  char * curr_block;

  char first_time;

  // flag for currently open output HDU
  char block_open;

  // number of bytes currently written to output HDU
  uint64_t bytes_written;

  // relevant observation meta data
  unsigned nchan;
  unsigned nant;
  unsigned ndim;
  unsigned nbay;
  time_t utc_start;

  uint64_t bytes_read;
  uint64_t bytes_per_second;

  mopsr_source_t   source;
  mopsr_bay_t    * all_bays;
  mopsr_module_t * mods;
  mopsr_chan_t   * chans;
  mopsr_delay_t ** delays;

  double tsamp;

} mopsr_dbdelaydb_t;

void usage(void);

// application specific memory management
int dbdelaydb_init (mopsr_dbdelaydb_t* ctx, dada_hdu_t *in, dada_hdu_t *out);
int dbdelaydb_destroy (mopsr_dbdelaydb_t * ctx, dada_hdu_t * in, dada_hdu_t * out);

// observation specific memory management
int dbdelaydb_alloc (mopsr_dbdelaydb_t * ctx);
int dbdelaydb_dealloc (mopsr_dbdelaydb_t * ctx);


int dbdelaydb_open (dada_client_t* client);
int dbdelaydb_close (dada_client_t* client, uint64_t bytes_written);

int64_t dbdelaydb_delay (dada_client_t* client, void * buffer, uint64_t bytes);

int64_t dbdelaydb_delay_block_cpu (dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id);

int64_t dbdelaydb_delay_block_gpu (dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id);

int64_t dbdelaydb_transfer (dada_client_t* client, void * buffer, size_t bytes, memory_mode_t mode);

