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
  uint64_t fb_block_size;
  uint64_t tb_block_size;

  // GPU pointers
  cudaStream_t stream;

  // GPU buffers for input and output of delayed data
  size_t d_buffer_size;
  void *  d_in;            // input data buffer on GPU
  void *  d_fbs;           // fan beam output on GPU
  void *  d_tb;            // tied array beam output on GPU

  unsigned     nant;
  //float *      h_ant_factors;     // antenna distances

  char         fan_beams;           // flag for fan beam mode [optional]
  int          nbeam;
  float *      h_beam_offsets;
  //float *      h_sin_thetas;    // angle from boresite to beam

  // rephasors required for each FB
  size_t phasors_size;
  float * h_phasors;
  void * d_phasors;  

  // rephasors required for the TB
  size_t tb_phasors_size;
  complex float * h_tb_phasors;  
  void          * d_tb_phasors;  

  unsigned tdec;           // time decimation factor

  // Tied array beams (TBs)
  unsigned n_tbs;               // number of TBs
  //mopsr_source_t * tb_sources;  // TB source positions
  void ** h_tbs;                // TBs CPU pointers
  void ** d_ts;                 // TB GPU pointers

  multilog_t * log;

  dada_hdu_t * fb_hdu;
  dada_hdu_t * tb_hdu;

  char * fb_block;
  char * tb_block;

  char first_time;

  // flag for currently open output HDU
  char tb_block_open;
  char fb_block_open;

  // number of bytes currently written to output HDU
  uint64_t bytes_written;

  // relevant observation meta data
  unsigned ndim;

  time_t utc_start;
  uint64_t bytes_read;
  //uint64_t bytes_per_second;

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
  mopsr_module_t ** modules;          // modules we are using (nant)

  mopsr_source_t   source;

  mopsr_chan_t   channel;             // each instance processes just 1 channel
  
  mopsr_delay_t ** delays;

  char internal_error;

  float ut1_offset;
  char lock_utc_flag;
  time_t lock_utc_time;

  char obs_tracking;
  char geometric_delays;

  unsigned ant_ids[16];               // PFB antenna identifiers
  char zero_data;
  char form_tied_block;
  uint64_t bytes_per_second;          // for input

  char tied_beam;                     // flag for producing a tied array bea,
  char steer_tb;                      // flag for steering towards the tied array beam
  //mopsr_source_t  tb_source;          // current TB source

} mopsr_bfdsp_t;

void usage(void);

// application specific memory management
int bfdsp_init (mopsr_bfdsp_t* ctx, dada_hdu_t *in, dada_hdu_t *fb,  
                dada_hdu_t *tb, char * bays_file, char * modules_file);
int bfdsp_destroy (mopsr_bfdsp_t * ctx, dada_hdu_t * in);

// observation specific memory management
int bfdsp_alloc (mopsr_bfdsp_t * ctx);
int bfdsp_dealloc (mopsr_bfdsp_t * ctx);

int bfdsp_open (dada_client_t* client);
int bfdsp_close (dada_client_t* client, uint64_t bytes_written);

int64_t bfdsp_io (dada_client_t* client, void * buffer, uint64_t bytes);
int64_t bfdsp_io_block (dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id);

int64_t bfdsp_transfer (dada_client_t* client, void * buffer, size_t bytes, memory_mode_t mode);
void bfdsp_update_tb (mopsr_bfdsp_t * ctx);

