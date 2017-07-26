/***************************************************************************
 *  
 *    Copyright (C) 2016 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "mopsr_delays_hires.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <cpgplot.h>

int calculate_delays_hires (
    unsigned nbay, mopsr_bay_t * bays, 
    unsigned nmod, mopsr_module_t * mods, 
    unsigned nchan, mopsr_chan_t * chans,
    mopsr_source_t source, struct timeval timestamp,
    mopsr_delay_hires_t ** delays, float start_md_angle,
    char apply_instrumental, char apply_geometric, 
    char  is_tracking, double tsamp
    )
{
  // delays should be an array allocat to nmod * nchan
  if (!delays)
  {
    fprintf (stderr, "calculate_delays: delays not allocated\n");
    return -1;
  }

  double jer_delay = calc_jer_delay (source.ra_curr, source.dec_curr, timestamp);

  unsigned ichan, imod, ibay;
  double C = 2.99792458e8;
  double dist, ant_dist, bay_dist;

  // this is about 20 samples [so greater than 25/2]
  const double fixed_delay  = 204.8e-6;
  double instrumental_delay;
  double geometric_delay = 0;

  double total_delay, coarse_delay, fractional_delay, delta_dist;
  double fractional_delay_samples;
  double module_offset, freq_ratio, frank_dist;
  unsigned int coarse_delay_samples;

  const double sampling_period = tsamp / 1000000;
  char bay[4];

  const double md_angle = asin(jer_delay);
  const double sin_start_md_angle = sin (start_md_angle);
  const double start_minus_md = (start_md_angle - md_angle);
  const double sin_start_minus_md = sin(start_minus_md);

  //fprintf (stderr, "jer_delay=%lf md_angle=%lf\n", jer_delay, md_angle);
  //fprintf (stderr, "start_md_angle=%lf sin_start_md_angle=%lf\n", start_md_angle, sin_start_md_angle);
  //fprintf (stderr, "start_minus_md=%lf sin_start_minus_md=%lf\n", start_minus_md, sin_start_minus_md);

  for (imod=0; imod < nmod; imod++)
  {
    // extract the bay name for this module
    strncpy (bay, mods[imod].name, 3);
    bay[3] = '\0';
    ant_dist = 0;
    bay_dist = 0;

    // if we are tracking then the ring antenna phase each of the 4
    // modules to the bay centre
    for (ibay=0; ibay<nbay; ibay++)
    {
      if (strcmp(bays[ibay].name, bay) == 0)
      {
        bay_dist = bays[ibay].dist;
      }
    }
    ant_dist = mods[imod].dist;

    // antenna_distances will be +ve for the west arm, negative for the east

    // read instrumental delays from file
    instrumental_delay = mods[imod].fixed_delay;

    module_offset = (double) mods[imod].bay_idx * MOLONGLO_MODULE_LENGTH;

    for (ichan=0; ichan<nchan; ichan++)
    {
      total_delay = fixed_delay;
      geometric_delay = 0;

      if (apply_instrumental)
        total_delay -= instrumental_delay;

      // are the ring antennae following the source
      if (is_tracking)
      {
        // the frank distance
        freq_ratio = (1.0 - (843.0 / chans[ichan].cfreq));
        frank_dist = module_offset * freq_ratio;
        dist = bay_dist + frank_dist;
        if (apply_geometric)
          geometric_delay = (jer_delay * dist) / C;
      }
      // the ring antenna are stationary, use the starting MD_ANGLE to determine the base phasing
      else
      {
        // the starting MD angle defines the geometric delay experienced
        geometric_delay = sin_start_md_angle * bay_dist / C;

        // if the ring antennae are not tracking, but we wish to steer the beam to the source
        if (apply_geometric)
        {
          geometric_delay -= sin_start_minus_md * ant_dist / C;
        }

      }

      total_delay -= geometric_delay;

      coarse_delay_samples = (unsigned) floor (total_delay / sampling_period);  
      coarse_delay = coarse_delay_samples * sampling_period;

      // fractional delay will run from 0.0 to 1.0 samples, change to -0.5 to 0.5
      fractional_delay = total_delay - coarse_delay;
      fractional_delay_samples = fractional_delay / sampling_period;

      /* This might be causing the bug with the duplicate responses */
      if (fractional_delay_samples > 0.5)
      {
        coarse_delay_samples++;
        coarse_delay = coarse_delay_samples * sampling_period;
        fractional_delay_samples -= 1.0;
      }

      delays[imod][ichan].tot_secs      = total_delay;
      delays[imod][ichan].tot_samps     = total_delay / sampling_period;

      delays[imod][ichan].samples       = coarse_delay_samples;
      delays[imod][ichan].fractional    = fractional_delay_samples;

      delays[imod][ichan].fringe_coeff    = -2 * M_PI * chans[ichan].cfreq * 1000000 * geometric_delay;

      if (apply_instrumental)
      {
        double delay_in_turns = instrumental_delay / sampling_period;
        double channel_turns = ((double) ichan - ((double) nchan / 2) + 0.5) * delay_in_turns;
        double channel_phase_offset = channel_turns * 2 * M_PI;

        delays[imod][ichan].fringe_coeff    -= (mods[imod].phase_offset + channel_phase_offset);
      }
    }
  }
  return 0;
}
