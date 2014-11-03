/***************************************************************************
 *  
 *    Copyright (C) 2011 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

// 
// library functions for handling delays in MOPSR
//

#include "mopsr_delays.h"
#include "slalib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <cpgplot.h>

#define MOPSR_DELAYS_DEFAULT_NTAPS 5

// Convention here is that the middle of the filter
// will be applied to the first data point.
void sinc_filter(complex16* series_in, complex16* series_out,
                 float * filter, int size, int ntaps, float delay)
{
  int ii,jj,filter_size;;
  float *fptr;
  complex16 *dptr;
  int ntaps_by_2 = ntaps/2;
  float real,imag;;

  //Only perform subsample delay
  if (delay >= 1.0)
  {
    fprintf (stderr, "delay must be between -1 and 1\n");
    return;
  }

  // populate sinc filter
  for (ii=0;ii<ntaps;ii++)
  {
    filter[ii] = sinc(ii-ntaps_by_2+delay);
  }

  // apply filter
  for (ii=0;ii<size;ii++)
  {
    // deal with start of data
    if (ii<ntaps_by_2)
    {
      fptr = filter + ntaps_by_2 - ii;
      dptr = series_in;
      filter_size = ntaps_by_2+ii + 1;
    }
    // deal with end of data
    else if (size-ii<ntaps_by_2+1)
    {
      fptr = filter;
      dptr = series_in + ii - ntaps_by_2;
      filter_size = size - ii + ntaps_by_2;
    }
    // deal with middle of data
    else
    {
      fptr = filter;
      dptr = series_in + ii - ntaps_by_2;
      filter_size = ntaps;
    }

    // multiply by filter and sum
    real = imag = 0;
    for (jj=0;jj<filter_size;jj++)
    {
      real += dptr[jj].real * fptr[jj];
      imag += dptr[jj].imag * fptr[jj];
    }
    series_out[ii].real = (char) floor(real+0.5);
    series_out[ii].imag = (char) floor(imag+0.5);
  }
    
  return;
}

void sinc_filter_float (complex float * series_in, complex float * series_out,
                        float * filter, int size, int ntaps, float delay)
{
  int ii,jj,filter_size;
  float *fptr;
  complex float *dptr;
  int ntaps_by_2 = ntaps/2;

  //Only perform subsample delay
  if (delay >= 1.0)
  {
    fprintf (stderr, "delay must be between -1 and 1\n");
    return;
  }

  // populate sinc filter
  float x, window;
  for (ii=0;ii<ntaps;ii++)
  {
    x = (float) ii - delay;
    window = 0.54 - 0.46 * cos(2.0 * M_PI * (x+0.5) / ntaps);
    filter[ii] = sinc((ii-ntaps_by_2)-delay) * window;
  }

  // apply filter
  for (ii=0;ii<size;ii++)
  {
    // deal with start of data
    if (ii<ntaps_by_2)
    {
      fptr = filter + ntaps_by_2 - ii;
      dptr = series_in;
      filter_size = ntaps_by_2+ii + 1;
    }
    // deal with end of data
    else if (size-ii<ntaps_by_2+1)
    {
      fptr = filter;
      dptr = series_in + ii - ntaps_by_2;
      filter_size = size - ii + ntaps_by_2;
    }
    // deal with middle of data
    else
    {
      fptr = filter;
      dptr = series_in + ii - ntaps_by_2;
      filter_size = ntaps;
    }

    // multiply by filter and sum
    series_out[ii] = 0;
    for (jj=0;jj<filter_size;jj++)
    {
      series_out[ii] += dptr[jj] * fptr[jj];
    }
  }
    
  return;
}



// read all the modules positions in the specified file, return the
// number of modules read, -1 on erro
mopsr_module_t * read_modules_file (const char* fname, int *nmod)
{
  unsigned max_modules = MOPSR_MAX_MODULES;
  mopsr_module_t * modules = (mopsr_module_t *) malloc(sizeof(mopsr_module_t) * max_modules);
  if (!modules)
  {
    fprintf (stderr, "read_modules_file: could not allocate memory\n");
    *nmod = -1;
    return 0;
  }

  FILE * fptr = fopen (fname, "r");
  if (!fptr)
  {
    fprintf (stderr, "read_modules_file: could not read modules file [%s]\n", fname);
    *nmod = -1;
    return 0;
  }
  char line[1024];

  unsigned imod = 0;
  int nscanned;
  double ref_dist = 0;
  while (imod < max_modules && fgets(line, 1024, fptr))
  {
    mopsr_module_t * mod = &(modules[imod]);
    nscanned = sscanf(line, "%5s %lf %lf %f %f", &(mod->name), &(mod->dist), 
                      &(mod->fixed_delay), &(mod->scale), &(mod->phase_offset)); 
    //fprintf (stderr, "name=%s dist=%lf fixed_delay=%20.18lf\n", 
    //         mod->name, mod->dist, mod->fixed_delay);
    if (nscanned != 5)
    {
      fprintf (stderr, "read_modules_file: could not parse [%s]\n", line);
      *nmod = -1;
      return 0;
    }
    else
      imod++;

#ifdef MOPSR_USE_SIGN   
    if ((mod->name[0] == 'E') || (mod->name[0] == 'e'))
      mod->sign = -1;
    else
      mod->sign = 1;
#else
    if ((mod->name[0] == 'E') || (mod->name[0] == 'e'))
      mod->dist *= -1;
#endif

    if (mod->name[4] == 'B')
      mod->bay_idx = 0;
    else if (mod->name[4] == 'G')
      mod->bay_idx = 1;
    else if (mod->name[4] == 'Y')
      mod->bay_idx = 2;
    else if (mod->name[4] == 'R')
      mod->bay_idx = 3;
    else
    {
      fprintf (stderr, "read_modules_file: could not determine bay index from %s\n", mod->name);
      *nmod = -1;
      return 0;
    }

    if (strcmp(mod->name, "E01-B") == 0)
      ref_dist = mod->dist * -1;
  }
  fclose (fptr);
  *nmod = imod;

  // now set the reference module to be E01-B [closest to phase centre in east arm]
/*
  for (imod=0; imod < *nmod; imod++)
  {
    modules[imod].dist -= ref_dist;
  }
*/
  return modules;
}

// read all the bays positions in the specified file, return the
// number of bays read, -1 on erro
mopsr_bay_t * read_bays_file (const char* fname, int *nbay)
{
  unsigned max_bays = MOPSR_MAX_BAYS;
  mopsr_bay_t * bays = (mopsr_bay_t *) malloc(sizeof(mopsr_bay_t) * max_bays);
  if (!bays)
  {
    fprintf (stderr, "read_bays_file: could not allocate memory\n");
    *nbay = -1;
    return 0;
  }

  FILE * fptr = fopen (fname, "r");
  if (!fptr)
  {
    fprintf (stderr, "read_bays_file: could not read bays file [%s]\n", fname);
    *nbay = -1;
    return 0;
  }
  char line[1024];

  unsigned ibay = 0;
  int nscanned;
  double ref_dist = 0;
  while (ibay < max_bays && fgets(line, 1024, fptr))
  {
    mopsr_bay_t * bay = &(bays[ibay]);
    nscanned = sscanf(line, "%3s %lf", &(bay->name), &(bay->dist));
    //fprintf (stderr, "name=%s dist=%lf\n", bay->name, bay->dist);
    if (nscanned != 2)
    {
      fprintf (stderr, "read_bays_file: could not parse [%s]\n", line);
      *nbay = -1;
      return 0;
    }
    else
      ibay++;

    if (strcmp(bay->name, "E01") == 0)
    {
      ref_dist = bay->dist;
      ref_dist *= -1;
    }

#ifdef MOPSR_USE_SIGN
    if ((bay->name[0] == 'E') || (bay->name[0] == 'e'))
      bay->sign = -1;
    else
      bay->sign = 1;
#else
    if ((bay->name[0] == 'E') || (bay->name[0] == 'e'))
    {
      bay->dist *= -1;
    }
#endif
  }

  fclose (fptr);
  *nbay = ibay;

/*
  for (ibay=0; ibay < *nbay; ibay++)
  {
    bays[ibay].dist -= ref_dist;
  }
*/

  return bays;
}


mopsr_pfb_t * read_signal_paths_file (const char * fname, int * npfb)
{
  mopsr_pfb_t * pfbs = (mopsr_pfb_t *) malloc(sizeof(mopsr_pfb_t) * MOPSR_MAX_PFBS);
  if (!pfbs)
  {
    fprintf (stderr, "read_signal_paths_file: could not allocate memory\n");
    *npfb = -1;
    return 0;
  }

  unsigned i, j;
  for (i=0; i<12; i++)
  {
    sprintf(pfbs[i].id,    "EG%02d", i+1);
    sprintf(pfbs[i+12].id, "WG%02d", i+1);
    for (j=0; j<MOPSR_MAX_MODULES_PER_PFB; j++)
    {
      sprintf(pfbs[i].modules[j], "-");
      sprintf(pfbs[i+12].modules[j], "-");
    }
  }

  FILE * fptr = fopen (fname, "r");
  if (!fptr)
  {
    fprintf (stderr, "read_signal_paths_file: could not read modules file [%s]\n", fname);
    *npfb = -1;
    return 0;
  }

  char line[1024];
  char mod[6];
  char pfb[5];
  int  mod_idx;

  int nscanned;
  while (fgets(line, 1024, fptr))
  {
    if ((line[0] == '#') || (line[0] == '\n'))
    {
      // just ignore
    }
    else
    {
      nscanned = sscanf (line, "%5s %4s %d", mod, pfb, &mod_idx);
      if (nscanned != 3)
      {
        nscanned = sscanf (line, "%5s - -", mod);
        //fprintf (stderr, "mod=%s pfb=- idx=-\n", mod);
        if (nscanned != 1)
        {
          fprintf (stderr, "read_signal_paths_file: could not parse [%s]\n", line);
          *npfb = -1;
          return 0;
        }
      }
      else
      {
        //fprintf (stderr, "mod=%s pfb=%s idx=%d\n", mod, pfb, mod_idx);
        for (i=0; i<MOPSR_MAX_PFBS; i++)
        {
          if (strcmp(pfb, pfbs[i].id) == 0)
          {
            strncpy (pfbs[i].modules[mod_idx], mod, 6);
            //fprintf (stderr, "pfbs[%d].modules[%d]='%s'\n", i, mod_idx, pfbs[i].modules[mod_idx]);
          }
        }    
      }
    }
  }
  fclose (fptr);
  *npfb = MOPSR_MAX_PFBS;

  return pfbs;
}

double calc_peckham_delay (double RA, double DEC, struct timeval timestamp)
{
  // calculate the MJD
  double time_ms = timestamp.tv_usec/1000000.000;
  struct tm * local_t = gmtime (&timestamp.tv_sec);
  double mjd = mjd_from_utc (local_t);

  double seconds = ((double) timestamp.tv_usec / 1000000.0) / (24*60*60);
  mjd += seconds;

  // get the LST
  double lst = lmst_from_mjd (mjd);

  double EQ = 2000;   // Epoch
  double PR = 0;      // proper motions
  double PD = 0;
  double PX = 0;      // parallax
  double PV = 0;      // radial velocity

  double RA_app, DEC_app;

  double mjd2000 = 51544.5;

  // Convert RA, DEC to apparent RA, DEC
  slaMap(RA, DEC, PR, PD, PX, PV, EQ, mjd2000, &RA_app, &DEC_app);

  char * sys = "FK5";
  double ep0 = 2000.0;
  double ep1 = (double) (local_t->tm_year + 1900) + ((double) (local_t->tm_yday) / 364.25);
  double RA_app_today = RA_app;
  double DEC_app_today = DEC_app;
  slaPreces(sys, ep0, ep1, &RA_app_today, &DEC_app_today);

  // difference in hours between current LST and
  double hour_angle = lst - RA_app_today;

  if (hour_angle < -M_PI)
    hour_angle += (2 * M_PI);

  double tilt = peckham_delay (MOLONGLO_HA_BASELINE, MOLONGLO_DEC_BASELINE,
                               hour_angle, DEC_app);

  return tilt;
}

/*
 * Calculate the meridian delay from RAJ, DECJ and timestamp
 */
double calc_jer_delay_J2000 (double RAJ, double DECJ, struct timeval timestamp)
{
  // calculate the MJD
  double time_ms = timestamp.tv_usec/1000000.000;
  struct tm * local_t = gmtime (&timestamp.tv_sec);
  double mjd = mjd_from_utc (local_t);

  double seconds = ((double) timestamp.tv_usec / 1000000.0) / (24*60*60);
  mjd += seconds;

  // get the LST
  double lst = lmst_from_mjd (mjd);

  // calculate the apparent RA and DEC 
  double RA_app, DEC_app;
  calc_app_position (RAJ, DECJ, timestamp, &RA_app, &DEC_app);

  // difference in hours between current LST and
  double hour_angle = lst - RA_app;

  if (hour_angle < -M_PI)
    hour_angle += (2 * M_PI);

  //const double deg2rad = M_PI/180.0;
  //const double slope = 1.0/289.9;
  //const double az = (4.9/3600)*DD2R;

  double projected_delay = jer_delay (hour_angle, DEC_app, 
                                      MOLONGLO_ARRAY_SLOPE, 
                                      MOLONGLO_AZIMUTH_CORR, 
                                      MOLONGLO_LATITUDE);

  return projected_delay;
}

/*
 * calculate the apparent position of the J2000 RA & DEC and UTC timestamp
 */
void calc_app_position (double RA, double DEC, struct timeval timestamp, double * RA_app, double * DEC_app)
{
  double time_ms = timestamp.tv_usec/1000000.000;
  struct tm * local_t = gmtime (&timestamp.tv_sec);

  double EQ = 2000;   // Epoch
  double PR = 0;      // proper motions
  double PD = 0;
  double PX = 0;      // parallax
  double PV = 0;      // radial velocity

  double mjd2000 = 51544.5;

  char * sys = "FK5";
  double ep0 = 2000.0;
  double ep1 = (double) (local_t->tm_year + 1900) + ((double) (local_t->tm_yday) / 364.25);

  *RA_app = RA;
  *DEC_app = DEC;

  slaPreces(sys, ep0, ep1, RA_app, DEC_app);

  return;
}

double calc_jer_delay2 (double RA_curr, double DEC_curr, struct timeval timestamp)
{
  // calculate the MJD
  double time_ms = timestamp.tv_usec/1000000.000;
  struct tm * local_t = gmtime (&timestamp.tv_sec);
  double mjd = mjd_from_utc (local_t);

  double seconds = ((double) timestamp.tv_usec / 1000000.0) / (24*60*60);
  mjd += seconds;

  // get the LST
  double lst = lmst_from_mjd (mjd);

  // difference in hours between current LST and
  double hour_angle = lst - RA_curr;
  if (hour_angle < -M_PI)
    hour_angle += (2 * M_PI);


  double projected_delay = jer_delay (hour_angle, DEC_curr,
                                      MOLONGLO_ARRAY_SLOPE,
                                      MOLONGLO_AZIMUTH_CORR,
                                      MOLONGLO_LATITUDE);

  return projected_delay;
}

int calculate_delays (unsigned nbay, mopsr_bay_t * bays, 
                      unsigned nmod, mopsr_module_t * mods, 
                      unsigned nchan, mopsr_chan_t * chans,
                      mopsr_source_t source, struct timeval timestamp,
                      mopsr_delay_t ** delays, char apply_instrumental,
                      char apply_geometric, char  is_tracking,
                      double tsamp)
{
  // delays should be an array allocat to nmod * nchan
  if (!delays)
  {
    fprintf (stderr, "calculate_delays: delays not allocated\n");
    return -1;
  }

  // compute the timestamp for 1 sample in the future
  struct timeval timestamp_next;
  timestamp_next.tv_sec = timestamp.tv_sec;
  timestamp_next.tv_usec = timestamp.tv_usec + tsamp;
  if (timestamp_next.tv_usec >= 1000000)
  {
    timestamp_next.tv_sec ++;
    timestamp_next.tv_usec -= 1000000;
  }
  
  double jer_delay      = calc_jer_delay2 (source.ra_curr, source.dec_curr, timestamp);
  double jer_delay_next = calc_jer_delay2 (source.ra_curr, source.dec_curr, timestamp_next);
  double jer_delay_ds   = jer_delay_next - jer_delay;

  //double peckham_delay = calc_peckham_delay (source.ra, source.dec, timestamp);
  //double md_tilt = calc_md_tilt (source.ra, source.dec, timestamp);
  //double md_tilt_next = calc_md_tilt (source.ra, source.dec, timestamp_next);
  //double md_tilt_ds = (md_tilt_next - md_tilt);
  //fprintf (stderr, "calculate_delays: ra=%lf, dec=%lf timestamp=%ld -> md_tilt=%20.18lf\n", source.ra, source.dec, timestamp.tv_sec, md_tilt);

  unsigned ichan, imod, ibay;
  double C = 2.99792458e8;
  double dist, ant_dist;

  const double fixed_delay  = 5.12e-5;
  double instrumental_delay;
  double geometric_delay = 0;
  double geometric_delay_next = 0;
  double fractional_delay_ds;
  double fringe_coeff_next;

  double total_delay, coarse_delay, fractional_delay, delta_dist;
  double module_offset, freq_ratio, frank_dist;
  unsigned int coarse_delay_samples;

  const double sampling_period = tsamp / 1000000;
  char bay[4];

  for (imod=0; imod < nmod; imod++)
  {
    // extract the bay name for this module
    strncpy (bay, mods[imod].name, 3);
    bay[3] = '\0';
    ant_dist = 0;

    // if we are tracking then the ring antenna phase each of the 4
    // modules to the bay centre
    if (is_tracking)
    {
      for (ibay=0; ibay<nbay; ibay++)
      {
        if (strcmp(bays[ibay].name, bay) == 0)
        {
#ifdef MOPSR_USE_SIGN
          ant_dist = bays[ibay].dist * mods[imod].sign;
#else
          ant_dist = bays[ibay].dist;
#endif
        }
      }
    }
    else
    {
#ifdef MOPSR_USE_SIGN
      ant_dist = mods[imod].dist * mods[imod].sign;
#else
      ant_dist = mods[imod].dist;
#endif
    }

    // antenna_distances will be +ve for the west arm, negative for the east

    // read instrumental delays from file
    instrumental_delay = mods[imod].fixed_delay;

    module_offset = (double) mods[imod].bay_idx * MOLONGLO_MODULE_LENGTH;

    for (ichan=0; ichan<nchan; ichan++)
    {
      total_delay = fixed_delay;

      if (apply_instrumental)
        total_delay -= instrumental_delay;

      if (is_tracking)
      {
        freq_ratio = (1.0 - (843.0 / chans[ichan].cfreq));
        frank_dist = module_offset * freq_ratio;
        dist = ant_dist + frank_dist;
      }
      else
        dist = ant_dist;

      // calculate the geometric delay from md_tilt angle
      geometric_delay = (jer_delay * dist) / C;

      // calculate the geometric delay for 1 sample offset into the future
      geometric_delay_next = (jer_delay_next * dist) / C;

      if (apply_geometric)
        total_delay -= geometric_delay;
      else
      {
        geometric_delay = 0;
        geometric_delay_next = 0;
      }

      //fprintf (stderr, "[%d][%d] total=%le = (%le + %le - %le) dist=%le\n", imod, ichan, total_delay, fixed_delay, instrumental_delay, geometric_delay, dist);

      // now calculate the fractional delay (rate in samples) per sample
      fractional_delay_ds = (geometric_delay_next - geometric_delay) / sampling_period;

      coarse_delay_samples = (unsigned) floor(total_delay / sampling_period);  

      coarse_delay = coarse_delay_samples * sampling_period;

      fractional_delay = total_delay - coarse_delay;

      delays[imod][ichan].tot_secs      = total_delay;
      delays[imod][ichan].tot_samps     = total_delay / sampling_period;

      delays[imod][ichan].samples       = coarse_delay_samples;
      delays[imod][ichan].fractional    = fractional_delay / sampling_period;
      delays[imod][ichan].fractional_ds = fractional_delay_ds;

      //if (imod == 0)
      //  fprintf (stderr, "jer_delay=%lf instrumental_delay=%lf geometric_delay=%lf fractional_delay=%lf [samples]\n", jer_delay, instrumental_delay * 1e9, geometric_delay *1e9,  delays[imod][ichan].fractional);

      delays[imod][ichan].fringe_coeff    = -2 * M_PI * chans[ichan].cfreq * 1000000 * geometric_delay;
      fringe_coeff_next                   = -2 * M_PI * chans[ichan].cfreq * 1000000 * geometric_delay_next;
      delays[imod][ichan].fringe_coeff_ds = fringe_coeff_next - delays[imod][ichan].fringe_coeff;

      double delay_in_turns = instrumental_delay * (1000000 / 1.28);
      double channel_turns = ((double) ichan - ((double) nchan / 2) + 0.5) * delay_in_turns;
      double channel_phase_offset = channel_turns * 2 * M_PI;

      //fprintf (stderr, "[%d][%d] instrumental_delay=%le delay_in_turns=%lf channel_turns=%lf channel_phase_offset=%lf\n", imod, ichan, instrumental_delay, delay_in_turns, channel_turns, channel_phase_offset);

      delays[imod][ichan].fringe_coeff    -= (mods[imod].phase_offset + channel_phase_offset);
    }

    //fprintf (stderr, "geometric_delay=%le, geometric_delay_next=%le, fractional_delay_ds=%le [s], fractional_delay_ds=%le [samps]\n",
    //         geometric_delay, geometric_delay_next, (geometric_delay_next - geometric_delay), fractional_delay_ds);
  }
  return 0;
}

double mjd_from_utc (struct tm * utc)
{
  double mjd;
  int year, month, day, status;

  year  = 1900 + utc->tm_year;
  month = 1 + utc->tm_mon;
  day   = utc->tm_mday; 

  // Calculate the modified Julian Date (JD-2400000.5) at 0hrs   
  slaCaldj (year, month, day, &mjd, &status);

  // now add the hours, minutes and seconds
  mjd += utc->tm_hour / 24.0;
  mjd += utc->tm_min / 1440.0;
  mjd += utc->tm_sec / 86400.0;

  //fprintf (stderr, "mjd_from_utc MJD=%lf\n", mjd);

  return mjd;
}


double lmst_from_mjd (double mjd)
{
  double gmst, lmst;
  double mjd_integer = floor (mjd);
  double day_fraction = mjd - mjd_integer;

  int HMSF[4];
  char sign;
  int NDP = 2;    // number of decimal places

  //fprintf (stderr, "MJD_int=%lf MJD_frac=%lf\n", mjd_integer, day_fraction);

  gmst = slaGmsta(mjd_integer, day_fraction);

  //slaCr2tf(NDP, gmst, &sign, HMSF);
  //fprintf (stderr, "GMST=%d:%d:%d.%d [%lf]\n",HMSF[0],HMSF[1],HMSF[2],HMSF[3], gmst);

  lmst  = gmst + MOLONGLO_LONGITUDE;

  //slaCr2tf(NDP, lmst, &sign, HMSF);
  //fprintf (stderr, "LMST=%d:%d:%d.%d [%lf] MOLONGLO_LONGITUDE=%lf\n",HMSF[0],HMSF[1],HMSF[2],HMSF[3], lmst, MOLONGLO_LONGITUDE);

  // remap to range [0 - 2PI]
  double lmst_2pi = slaDranrm (lmst);

  //slaCr2tf(NDP, lmst_2pi, &sign, HMSF);
  //fprintf (stderr, "LMST_2PI=%d:%d:%d.%d [%lf]\n",HMSF[0],HMSF[1],HMSF[2],HMSF[3], lmst_2pi);

  return lmst_2pi;
}

int mopsr_delays_hhmmss_to_rad (char * hhmmss, double * rads)
{
  int ihour, imin;
  double sec;
  const char *sep = ":";
  char * saveptr;

  char * str = strtok_r(hhmmss, sep, &saveptr);
  if (sscanf(str, "%d", &ihour) != 1)
    return -1;

  str = strtok_r(NULL, sep, &saveptr);
  if (sscanf(str, "%d", &imin) != 1)
    return -1;

  str = strtok_r(NULL, sep, &saveptr);
  if (sscanf(str, "%lf", &sec) != 1)
    return -1;
  int status;
  slaDtf2r (ihour, imin, sec, rads, &status);
  return (status * -1);
}

int mopsr_delays_ddmmss_to_rad (char * ddmmss, double * rads)
{
  int ideg, iamin;
  double asec;
  const char *sep = ":";
  char * saveptr;

  char * str = strtok_r(ddmmss, sep, &saveptr);
  if (sscanf(str, "%d", &ideg) != 1)
    return -1;

  str = strtok_r(NULL, sep, &saveptr);
  if (sscanf(str, "%d", &iamin) != 1)
    return -1;

  str = strtok_r(NULL, sep, &saveptr);
  if (sscanf(str, "%lf", &asec) != 1)
    return -1;
  int status;
  slaDaf2r (abs(ideg), iamin, asec, rads, &status );
  if (ideg < 0)
    *rads *= -1;
  return (status * -1);
}


// MB 6 June 2014
// Returns delay in natural units - multiply by B/c for delay;

// These are all in radians:
// ha_baseline is hour angle of baseline
// dec_baseline is dec of baseline
// ha_source is hour angle of source
// dec_source is dec of source

// For Molonglo: ha_baseline is 90 - cos(lat)*gradient*180/pi - az sin(lat)
//               dec_baseline is slope*sin(lat) - az cos(lat)
//               ha_baseline = 89.79369761 deg
//               dec_baseline = -0.0508515 deg
// az = -4.9"
// slope = 1/289.9 (radians)

double peckham_delay(double ha_baseline, double dec_baseline,
         double ha_source, double dec_source) {

  // ha_baseline = 89.79369761 * M_PI/180.0
  // dec_baseline = -0.0508515 * M_PI/180.0

  return(cos(dec_baseline)*sin(ha_baseline)*cos(dec_source)*sin(ha_source)
       + cos(dec_baseline)*cos(ha_baseline)*cos(dec_source)*cos(ha_source)
       + sin(dec_baseline)*sin(dec_source));

}

double jer_delay(double ha_source, double dec_source,
         double tilt, double skew, double latitude)
{
  return( sin(ha_source)*cos(dec_source)*cos(tilt) + 
          cos(ha_source)*cos(dec_source)* (skew*sin(latitude)-sin(tilt)*cos(latitude))-
          sin(dec_source)*(skew*cos(latitude)+sin(tilt)*sin(latitude)) );
}

// simple PGPLOT of the delays across the array
int mopsr_delays_plot (unsigned nmod, unsigned nchan, mopsr_delay_t ** delays, struct timeval timestamp)
{
  float * xvals = (float *) malloc(sizeof(float) * nmod);
  float * yvals = (float *) malloc(sizeof(float) * nmod);
  float * yvals2 = (float *) malloc(sizeof(float) * nmod);
  //float * samples = (float *) malloc(sizeof(float) * nmod);
  //float * fractional = (float *) malloc(sizeof(float) * nmod);
  //float * fringes = (float *) malloc(sizeof(float) * nmod);
  //float * delay_ds = (float *) malloc(sizeof(float) * nmod);
  //float * fringe_ds = (float *) malloc(sizeof(float) * nmod);

  double time = (double) timestamp.tv_sec + ((double) timestamp.tv_usec / 1000000);

  int nm = nmod / 2;

  int imod;
  for (imod=-nm; imod<nm; imod++)
    xvals[imod+nm] = (float) imod;

  float xmin = (float) -nm;
  float xmax = (float) nm;
  double ymax = -DBL_MAX;
  double ymin = DBL_MAX;
  double yrange;
  double y;
  for (imod=0; imod <nmod; imod++)
  {
    y = delays[imod][0].fractional;
    if (y > ymax) ymax = y;
    if (y < ymin) ymin = y;
    yvals[imod] = (float) y;
  }

  yrange = (ymax - ymin);
  ymax += (yrange / 10);
  ymin -= (yrange / 10);

  cpgbbuf();
  cpgeras();

  char title[60];
  sprintf (title, "Delays: time = %lf", time);

  cpgswin(xmin, xmax, (float) ymin, (float) ymax);
  cpgsvp(0.10, 0.90, 0.7, 0.90);
  cpgbox("BCST", 0.0, 0.0, "BCNSTV", 0.0, 0.0);
  cpglab ("", "Fractional Delay", title);
  cpgline (nmod, xvals, yvals);

  ymax = -DBL_MAX;
  ymin = DBL_MAX;
  for (imod=0; imod <nmod; imod++)
  {
    y = (double) delays[imod][0].samples;
    if (y > ymax) ymax = y;
    if (y < ymin) ymin = y;
    yvals[imod] = (float) y;
  }

  yrange = (ymax - ymin);
  ymax += (yrange / 10);
  ymin -= (yrange / 10);

  cpgswin(xmin, xmax, (float) ymin, (float) ymax);
  cpgsvp(0.10, 0.90, 0.5, 0.7);
  cpgbox("BCST", 0.0, 0.0, "BCNSTV", 0.0, 0.0);
  cpglab ("", "Sample Delay", "");
  cpgline (nmod, xvals, yvals);

  ymax = -DBL_MAX;
  ymin = DBL_MAX;
  for (imod=0; imod <nmod; imod++)
  {
    y = (double) delays[imod][0].tot_secs;
    if (y > ymax) ymax = y;
    if (y < ymin) ymin = y;
    yvals[imod] = (float) y;
  }
  yrange = (ymax - ymin);
  ymax += (yrange / 10);
  ymin -= (yrange / 10);

  cpgswin(xmin, xmax, (float) ymin, (float) ymax);
  cpgsvp(0.10, 0.90, 0.3, 0.5);
  cpgbox("BCST", 0.0, 0.0, "BCNSTV", 0.0, 0.0);
  cpglab ("Module Number", "Total Delay", "");
  cpgline (nmod, xvals, yvals);

  ymax = -DBL_MAX;
  ymin = DBL_MAX;
  for (imod=0; imod <nmod; imod++)
  {
    y = cos(delays[imod][0].fringe_coeff);
    if (y > ymax) ymax = y;
    if (y < ymin) ymin = y;
    yvals[imod] = (float) y;

    y = sin(delays[imod][0].fringe_coeff);
    if (y > ymax) ymax = y;
    if (y < ymin) ymin = y;
    yvals2[imod] = (float) y;
  }
  yrange = (ymax - ymin);
  ymax += (yrange / 10);
  ymin -= (yrange / 10);

  cpgswin(xmin, xmax, (float) ymin, (float) ymax);
  cpgsvp(0.10, 0.90, 0.1, 0.3);
  cpgbox("BCNST", 0.0, 0.0, "BCNSTV", 0.0, 0.0);
  cpglab ("", "Fringe", "");
  cpgsci(2);
  cpgline (nmod, xvals, yvals);
  cpgsci(3);
  cpgline (nmod, xvals, yvals2);

  cpgebuf();

  free (xvals);
  free (yvals);
  //free (samples);
  //free (fractional);

}
