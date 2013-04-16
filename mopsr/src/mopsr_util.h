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

  // optional channel to plot
  int chan;

  // optional antenna to plot
  int ant;

  // code for antenna number
  unsigned int ant_code;

  // logarithmic plot option
  int plot_log;

  // zap DC channel (0)
  int zap;

  float ymin;

  float ymax;

} mopsr_util_t;


int mopsr_print_pfbframe (char * buffer, ssize_t bytes, mopsr_util_t * opts);

void mopsr_sqld_pfbframe (float * power_spectra, char * buffer,
                          mopsr_util_t * opts, int flag);

void mopsr_extract_channel (float * timeseries, char * buffer, ssize_t bytes,
                            unsigned int channel, unsigned int antenna, 
                            unsigned int nchan, unsigned int nant);

void print_packet (char * buffer, unsigned int size);
void dump_packet (char * buffer, unsigned int size);

void mopsr_plot_time_series (float * timeseries, unsigned int nsamps, mopsr_util_t * opts);

void mopsr_plot_bandpass (float * power_spectra, mopsr_util_t * opts);

void mopsr_plot_waterfall (float * spectra, unsigned int nsamps, mopsr_util_t * opts);
