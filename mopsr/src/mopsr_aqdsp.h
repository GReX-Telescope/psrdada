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

#include "mopsr_delays_cuda.h"

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOPSR_AQDSP_DEFAULT_NTAPS 5
#define SKZAP

typedef struct {

  int device;

  unsigned verbose;

  uint64_t time_msec;

  uint64_t block_size;

  // number of taps in filter - must be odd
  unsigned ntaps;

  // FIR filter coefficients
  float * filter;
  size_t filter_size; 

  // GPU pointers
  cudaStream_t stream;

  // struct for managing sample delay in GPU RAM
  transpose_delay_t * gtx;

  // GPU buffers for input and output of delayed data
  size_t d_buffer_size;
  void * d_in;            // input data buffer on GPU
  void * d_out;           // delayed & rephased data buffer on GPU
#ifdef SKZAP
  void * d_fbuf;          // float input data buffer on GPU
  void * d_rstates;        // floats for curand states
  void * d_sigmas;        // floats for curand states
#endif

  // delays for each channel and antenna
  size_t delays_size;
  float * h_delays;
  void  * d_delays;

  // fringes for each channel and antenna
  size_t fringes_size;
  float * h_fringes;
  void  * d_fringes;

  // per sample cororections for delay and fringe coefficient
  float * h_delays_ds;
  float * h_fringe_coeffs_ds;

  // scale factors for each antenna
  float * h_ant_scales;
  size_t  ant_scales_size;

  // corrections for rephasing coarse channels
  // void * d_corr;  
 
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

  mopsr_chan_t   * channels;
  
  mopsr_delay_t ** delays;

  double tsamp;

  char internal_error;

  float ut1_offset;
  char lock_utc_flag;
  time_t lock_utc_time;

  char obs_tracking;
  char geometric_delays;

  unsigned ant_ids[16];               // PFB antenna identifiers


} mopsr_aqdsp_t;

void usage(void);

// application specific memory management
int aqdsp_init (mopsr_aqdsp_t* ctx, dada_hdu_t *in, dada_hdu_t *out, 
                char * bays_file, char * modules_file, char * signal_paths_file);
int aqdsp_destroy (mopsr_aqdsp_t * ctx, dada_hdu_t * in, dada_hdu_t * out);

// observation specific memory management
int aqdsp_alloc (mopsr_aqdsp_t * ctx);
int aqdsp_dealloc (mopsr_aqdsp_t * ctx);

int aqdsp_open (dada_client_t* client);
int aqdsp_close (dada_client_t* client, uint64_t bytes_written);

int64_t aqdsp_io (dada_client_t* client, void * buffer, uint64_t bytes);
int64_t aqdsp_io_block (dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id);

int64_t aqdsp_transfer (dada_client_t* client, void * buffer, size_t bytes, memory_mode_t mode);

