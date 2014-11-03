/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <unistd.h>
#include <inttypes.h>

typedef struct {

  // number of channels in input
  unsigned int nchan;

  // number of antenna in input
  unsigned int nant;

  // number of dimensions in input 
  unsigned int ndim;

  // number of histogram bins (256)
  unsigned int nbin;

  // optional channes to plot
  int chans[2];

  // optional antenna to plot
  int ant;

  // code for antenna number
  unsigned int ant_code;

  // identifier for antenna
  unsigned int ant_id;

  // logarithmic plot option
  unsigned plot_log;

  // plot with no axes
  unsigned plot_plain;

  // control lock text (-1 ignore, 0 no lock, 1 lock)
  int lock_flag;

  // long lock flag (if mgt_lock gone since boot)
  int lock_flag_long;

  // zap DC channel (0)
  int zap;

  float ymin;

  float ymax;

} mopsr_util_t;

int mopsr_channelise_frame (char * buffer, uint64_t nbytes, unsigned nant, unsigned npt);
int mopsr_print_pfbframe (char * buffer, ssize_t bytes, mopsr_util_t * opts);
void mopsr_sqld_pfbframe (float * power_spectra, char * buffer,
                          mopsr_util_t * opts, int flag);

void mopsr_form_histogram (unsigned int * histogram, char * buffer, ssize_t bytes,  mopsr_util_t * opts);
void mopsr_form_histogram_range (unsigned int * histogram, char * buffer, ssize_t bytes,  mopsr_util_t * opts, int start_chan, int end_chan);

void mopsr_extract_channel (float * timeseries, char * buffer, ssize_t bytes,
                            unsigned int channel, unsigned int antenna, 
                            unsigned int nchan, unsigned int nant);

void print_packet (char * buffer, unsigned int size);
void dump_packet (char * buffer, unsigned int size);

void mopsr_plot_time_series (float * timeseries, unsigned int chan, unsigned int nsamps, mopsr_util_t * opts);

void mopsr_plot_bandpass (float * power_spectra, mopsr_util_t * opts);
void mopsr_plot_bandpass_vertical (float * power_spectra, mopsr_util_t * opts);

void mopsr_plot_waterfall (float * spectra, unsigned int nsamps, mopsr_util_t * opts);
void mopsr_plot_waterfall_vertical (float * spectra, unsigned int nsamps, mopsr_util_t * opts);
void mopsr_transpose (float * out, float * in, unsigned int nsamps, mopsr_util_t * opts);

void mopsr_plot_histogram (unsigned int * histogram, mopsr_util_t * opts);

void mopsr_zero_float (float * array, unsigned int size);
void mopsr_zero_uint (unsigned int * array, unsigned int size);

void get_scale (int from, int to, float * width, float * height);
void set_resolution (int width_pixels, int height_pixels);
void set_white_on_black();

void mopsr_zap_channel (char * buffer, ssize_t bytes, unsigned int schan, unsigned int echan, mopsr_util_t * opts);

void mopsr_plot_delay_snrs(float * snrs, mopsr_util_t * opts);

