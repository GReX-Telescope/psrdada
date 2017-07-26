/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 *****************************************************************************/

#ifndef __MOPSR_DELAYS_HIRES_H
#define __MOPSR_DELAYS_HIRES_H

#include "mopsr_delays.h"

typedef struct delay_vals_hires
{
  unsigned  samples;         // integer samples to delay
  double    fractional;      // fractional samples to delay
  double    fringe_coeff;    // coefficient for complex fringe correction
  double    tot_samps;       // total delay [samples]
  double    tot_secs;        // total delay [seconds]
} mopsr_delay_hires_t;

int calculate_delays_hires (
    unsigned nbay, mopsr_bay_t * bays, 
    unsigned nmod, mopsr_module_t * mods, 
    unsigned nchan, mopsr_chan_t * chans,
    mopsr_source_t source, struct timeval timestamp,
    mopsr_delay_hires_t ** delays, float starting_md_angle, 
    char apply_instrumental, char apply_geometric, 
    char is_tracking, double tsamp);

#endif
