/***************************************************************************
 *  
 *    Copyright (C) 2014 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_def.h"
#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_cuda.h"

#include "mopsr_delays_cuda.h"

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {

  int device;

  unsigned verbose;

  uint64_t time_msec;

  uint64_t in_block_size;
  uint64_t ou_block_size;

  // GPU pointers
  cudaStream_t stream;

  // GPU buffers for input and output of delayed data
  size_t d_buffer_size;
  void *  d_in;            // input data buffer on GPU
  void *  d_fbs;           // fan beam output on GPU

  unsigned     nant;
  float *      h_ant_factors;     // antenna distances

  int          nbeam;
  float *      h_sin_thetas;    // angle from boresite to beam

  // rephasors required for each FB
  size_t phasors_size;

  complex float * h_phasors;
  void * d_phasors;  
 
  unsigned tdec;           // time decimation factor

  // Tied array beams (TBs)
  unsigned n_tbs;               // number of TBs
  mopsr_source_t * tb_sources;  // TB source positions
  void ** d_tbs;                // TBs GPU pointers
  void ** h_tbs;                // TBs CPU pointers

  multilog_t * log;

  dada_hdu_t * out_hdu;

  char * curr_block;

  char first_time;

  // flag for currently open output HDU
  char block_open;

  // number of bytes currently written to output HDU
  uint64_t bytes_written;

  // relevant observation meta data
  unsigned ndim;

  time_t utc_start;
  uint64_t bytes_read;
  uint64_t bytes_per_second;

  unsigned chan_offset;               // channel offset from original 128 channels
  float channel_bw;                   // width of critically sampled channel
  float base_freq;                    // lowest freq (bottom of chan 0)

  char correct_delays;                // flag to enable/disable delay correction
  char output_stf;                    // flag to transpose output to STF
  char sum_ant;                       // add complex voltages from antenna together

  char pfb_id[5];                     // 4 letter PFB ID

  int npfbs;
  mopsr_pfb_t * pfbs;

  int nbays;
  mopsr_bay_t * all_bays;

  int nmodules;
  mopsr_module_t * all_modules;       // every module in the array (read from file)
  mopsr_module_t * modules;           // modules we are using (nant)

  mopsr_source_t   source;

  mopsr_chan_t   channel;             // each instance processes just 1 channel
  
  mopsr_delay_t ** delays;

  double tsamp;

  char internal_error;

  float ut1_offset;
  char lock_utc_flag;
  time_t lock_utc_time;

  char obs_tracking;
  char geometric_delays;

  unsigned ant_ids[16];               // PFB antenna identifiers


} mopsr_bfdsp_t;

void usage(void);

// application specific memory management
int bfdsp_init (mopsr_bfdsp_t* ctx, dada_hdu_t *in, dada_hdu_t *out, 
                char * bays_file, char * modules_file, char * signal_paths_file);
int bfdsp_destroy (mopsr_bfdsp_t * ctx, dada_hdu_t * in, dada_hdu_t * out);

// observation specific memory management
int bfdsp_alloc (mopsr_bfdsp_t * ctx);
int bfdsp_dealloc (mopsr_bfdsp_t * ctx);

int bfdsp_open (dada_client_t* client);
int bfdsp_close (dada_client_t* client, uint64_t bytes_written);

int64_t bfdsp_io (dada_client_t* client, void * buffer, uint64_t bytes);
int64_t bfdsp_io_block (dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id);

int64_t bfdsp_transfer (dada_client_t* client, void * buffer, size_t bytes, memory_mode_t mode);

