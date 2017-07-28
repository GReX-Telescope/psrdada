/***************************************************************************
 *  
 *    Copyright (C) 2011 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

// 
// library functions for handling delays in MOPSR
//

#include "sofa.h"
#include "mopsr_delays.h"

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
  if (delay >= 1.0 || delay < -1.0)
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

    if ((mod->name[0] == 'E') || (mod->name[0] == 'e'))
      mod->dist *= -1;

    if (mod->name[4] == 'R')
      mod->bay_idx = 0;
    else if (mod->name[4] == 'Y')
      mod->bay_idx = 1;
    else if (mod->name[4] == 'G')
      mod->bay_idx = 2;
    else if (mod->name[4] == 'B')
      mod->bay_idx = 3;
    else
    {
      fprintf (stderr, "read_modules_file: could not determine bay index from %s\n", mod->name);
      *nmod = -1;
      return 0;
    }
  }
  fclose (fptr);
  *nmod = imod;

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

    if ((bay->name[0] == 'E') || (bay->name[0] == 'e'))
    {
      bay->dist *= -1;
    }
  }

  fclose (fptr);
  *nbay = ibay;

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
#ifdef HIRES
  for (i=0; i<11; i++)
  {
    sprintf(pfbs[2*i+0].id,    "EG%02d-B", i+1);
    sprintf(pfbs[2*i+1].id,    "EG%02d-R", i+1);
    sprintf(pfbs[2*i+22+0].id, "WG%02d-B", i+1);
    sprintf(pfbs[2*i+22+1].id, "WG%02d-R", i+1);

    for (j=0; j<MOPSR_MAX_MODULES_PER_PFB; j++)
    {
      sprintf(pfbs[2*i+0].modules[j],    "-");
      sprintf(pfbs[2*i+1].modules[j],    "-");
      sprintf(pfbs[2*i+22+0].modules[j], "-");
      sprintf(pfbs[2*i+22+1].modules[j], "-");
    }
  }
#else
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
#endif

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
#ifdef HIRES
      nscanned = sscanf (line, "%5s %6s %d", mod, pfb, &mod_idx);
#else
      nscanned = sscanf (line, "%5s %4s %d", mod, pfb, &mod_idx);
#endif
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

int cal_app_pos_iau (double RA, double DEC, struct tm * utc, double * RA_app, double * DEC_app)
{
  double pr = 0;  // atan2 ( -354.45e-3 * DAS2R, cos(dc) );
  double pd = 0;  // 595.35e-3 * DAS2R;

  // Parallax (arcsec) and recession speed (km/s).
  double px = 0;  // 164.99e-3;
  double rv = 0;  // 0.0;

  double utc1, utc2;
  double tai1, tai2;
  double tt1, tt2;

  if (iauDtf2d ( "UTC", utc->tm_year + 1900, utc->tm_mon + 1, utc->tm_mday, utc->tm_hour, utc->tm_min, utc->tm_sec, &utc1, &utc2 ) ) return -1;
  if ( iauUtctai ( utc1, utc2, &tai1, &tai2 ) ) return -1;
  if ( iauTaitt ( tai1, tai2, &tt1, &tt2 ) ) return -1;

  double ri, di, eo;

    // catalogue posn to apparent for date output in CIRS
  iauAtci13 ( RA, DEC, pr, pd, px, rv, tt1, tt2, &ri, &di, &eo );

  *RA_app = ri - eo;
  *DEC_app = di;

  return 0 ;
}

void calc_observed_pos (double rc, double dc, struct tm * utc, double dut1, double * RA_obs, double * DEC_obs, double * HA)
{
  // proper motion (RA, DEC derivatives)
  double pr = atan2 ( -354.45e-3 * DAS2R, cos(dc) );
  double pd = 595.35e-3 * DAS2R;

  // Parallax (arcsec) and recession speed (km/s).
  double px = 164.99e-3;
  double rv = 0.0;

  // override these
  pr = pd = px = rv = 0;

  double utc1, utc2;

  int rval = iauDtf2d ( "UTC", utc->tm_year + 1900, utc->tm_mon + 1, utc->tm_mday, utc->tm_hour, utc->tm_min, utc->tm_sec, &utc1, &utc2 );

  double xp = 50.995e-3 * DAS2R;
  double yp = 376.723e-3 * DAS2R;

  double elong, phi;

  iauAf2a ( ' ' , 149, 25, 14.5452, &elong );
  iauAf2a ( '-' , 35,  25, 28.7682, &phi );

  double hm = 751.0;

  /* Ambient pressure (HPa), temperature (C) and rel. humidity (frac). */
  double phpa = 952.0;
  double tc = 18.5;
  double rh = 0.83;

  /* Effective color (35 cm microns). */
  double  wl = 350000;

  double aob, zob, hob, dob, rob, eo;

  rval = iauAtco13 ( rc, dc, pr, pd, px, rv, utc1, utc2, dut1,
    elong, phi, hm, xp, yp, phpa, tc, rh, wl,
    &aob, &zob, &hob, &dob, &rob, &eo );

  fprintf (stderr, "ra=%lf, dec=%lf, hour angle=%lf\n",rob, dob, hob);

  *RA_obs  = rob;
  *DEC_obs = dob;
  *HA = hob;

#ifdef _DEBUG
  int HMSF[4];
  char sign;
  int NDP = 2;    // number of decimal places
  iauA2tf (NDP, hob, &sign, HMSF);
  fprintf (stderr, "  HOB: %02d:%02d:%02d.%d [%20.15lf]\n",
                      HMSF[0],HMSF[1],HMSF[2],HMSF[3], hob);
#endif

  return;
}

double calc_ha_source ( double RA_curr, double DEC_curr, struct timeval timestamp)
{
  // convert the integer time to an MJD
  struct tm * utc_t = gmtime (&timestamp.tv_sec);
  double mjd = mjd_from_utc (utc_t);

  // convert the fractional time from seconds to days
  double fractional_seconds = ((double) timestamp.tv_usec) / 1000000.0;
  double fractional_days    = fractional_seconds / 86400.0;

  mjd += fractional_days;

  // get the LAST from the MJD
  double last = last_from_mjd (mjd);

#ifdef _DEBUG
  int HMSF[4];
  char sign;
  int NDP = 2;    // number of decimal places

  iauA2tf (NDP, last, &sign, HMSF);
  fprintf (stderr, "LAST: %02d:%02d:%02d.%d [%20.15lf]\n", HMSF[0],HMSF[1],HMSF[2],HMSF[3], last);
#endif

  // difference in hours between current LAST and
  double hour_angle = last - RA_curr;
  if (hour_angle < -M_PI)
    hour_angle += (2 * M_PI);

#ifdef _DEBUG
  iauA2tf (NDP, hour_angle, &sign, HMSF);
  fprintf (stderr, "  HA: %02d:%02d:%02d.%d [%20.15lf]\n",
            HMSF[0],HMSF[1],HMSF[2],HMSF[3], hour_angle);
#endif

  return hour_angle;
}


double calc_jer_delay (double RA_curr, double DEC_curr, struct timeval timestamp)
{

  double hour_angle = calc_ha_source (RA_curr, DEC_curr, timestamp);

  double projected_delay = doc_delay (hour_angle, DEC_curr,
                                      MOLONGLO_ARRAY_SLOPE,
                                      MOLONGLO_AZIMUTH_CORR,
                                      MOLONGLO_LATITUDE);

  return projected_delay;
}

void calc_app_ha_dec (double RA_J2000, double DEC_J2000, struct timeval timestamp, double * HA_app, double * DEC_app)
{
  double ra = 0;
  double dec = 0;

  // convert the integer time to an MJD
  struct tm * utc = gmtime (&timestamp.tv_sec);

  // calculate the apparent position (integer second accuracy);
  cal_app_pos_iau (RA_J2000, DEC_J2000, utc, &ra, &dec);

  // determine the HA of the source
  double ha = calc_ha_source (ra, dec, timestamp);

  *HA_app = ha;
  *DEC_app = dec;

  return;
}

// is_tracking: if the ring antenna are following the source
// apply_geometric: steer boresight to source
int calculate_delays (unsigned nbay, mopsr_bay_t * bays, 
                      unsigned nmod, mopsr_module_t * mods, 
                      unsigned nchan, mopsr_chan_t * chans,
                      mopsr_source_t source, struct timeval timestamp,
                      mopsr_delay_t ** delays, float start_md_angle,
                      char apply_instrumental, char apply_geometric, 
                      char is_tracking, double tsamp)
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

#ifdef HIRES
  const double fixed_delay  = 40.96e-5;
#else
  const double fixed_delay  = 5.12e-5;
#endif
  double instrumental_delay;
  double geometric_delay = 0;

  double total_delay, coarse_delay, fractional_delay;
  double module_offset, freq_ratio, frank_dist;
  unsigned int coarse_delay_samples;

  const double sampling_period = tsamp / 1000000;
  char bay[4];

  const double md_angle = asin(jer_delay);
  const double sin_start_md_angle = sin (start_md_angle);
  const double start_minus_md = (start_md_angle - md_angle);
  const double sin_start_minus_md = sin(start_minus_md);

  //const double sin_start_md_angle = sin (start_md_angle);
  //const double cos_start_md_angle = cos (start_md_angle);

  //const double sin_jer_delay = jer_delay;
  //const double cos_jer_delay = cos (asin(jer_delay));

  //const double sin_start_md_minus_jer = (sin_start_md_angle * cos_jer_delay) - (cos_start_md_angle * sin_jer_delay);

  //double R2D = 180.0 / M_PI;
  //fprintf (stderr, "md_angle=%lf, start_md_angle=%lf\n", md_angle*R2D, start_md_angle*R2D);
  //fprintf (stderr, "sin_start_md_angle=%lf sin_start_minus_md=%lf\n", sin_start_md_angle, sin_start_minus_md);

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

      // the frank distance
      freq_ratio = (1.0 - (843.0 / chans[ichan].cfreq));
      frank_dist = module_offset * freq_ratio;

      // are the ring antennae following the source
      if (is_tracking)
      {
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

      //fprintf (stderr, "[%d][%d] total=%le = (%le + %le - %le) dist=%le\n", imod, ichan, total_delay, fixed_delay, instrumental_delay, geometric_delay, dist);

      coarse_delay_samples = (unsigned) floor (total_delay / sampling_period);  
      coarse_delay = coarse_delay_samples * sampling_period;
      fractional_delay = total_delay - coarse_delay;

      delays[imod][ichan].tot_secs      = total_delay;
      delays[imod][ichan].tot_samps     = total_delay / sampling_period;

      delays[imod][ichan].samples       = coarse_delay_samples;
      delays[imod][ichan].fractional    = fractional_delay / sampling_period;

      //if (ichan == 0)
       // fprintf (stderr, "jer_delay=%lf instrumental_delay=%lf geometric_delay=%lf sample_delay=%u fractional_delay=%lf [samples]\n", jer_delay, instrumental_delay * 1e9, geometric_delay *1e9,  delays[imod][ichan].samples, delays[imod][ichan].fractional);

      delays[imod][ichan].fringe_coeff    = -2 * M_PI * chans[ichan].cfreq * 1000000 * geometric_delay;

      if (apply_instrumental)
      {
        double delay_in_turns = instrumental_delay / sampling_period;
        double channel_turns = ((double) ichan - ((double) nchan / 2) + 0.5) * delay_in_turns;
        double channel_phase_offset = channel_turns * 2 * M_PI;

      //fprintf (stderr, "[%d][%d] instrumental_delay=%le delay_in_turns=%lf channel_turns=%lf channel_phase_offset=%lf\n", imod, ichan, instrumental_delay, delay_in_turns, channel_turns, channel_phase_offset);

        delays[imod][ichan].fringe_coeff    -= (mods[imod].phase_offset + channel_phase_offset);
      }

    }

    //fprintf (stderr, "geometric_delay=%le, geometric_delay_next=%le, fractional_delay_ds=%le [s], fractional_delay_ds=%le [samps]\n",
    //         geometric_delay, geometric_delay_next, (geometric_delay_next - geometric_delay), fractional_delay_ds);
  }
  return 0;
}

double mjd_from_utc (struct tm* utc)
{
  int iy = 1900 + utc->tm_year;
  int im = 1 + utc->tm_mon;
  int id = (int) utc->tm_mday;

  double d, d1, d2;

  int rval = iauCal2jd ( iy, im, id, &d1, &d2 );
  if (rval != 0)
    return -1;

  int ihour = (int) utc->tm_hour;
  int imin  = (int) utc->tm_min;
  double sec = (double) utc->tm_sec;

  rval = iauTf2d ('+', ihour, imin, sec, &d );
  if (rval != 0)
    return -1;

  double mjd = d2 + d;

  return mjd;
}

double gmst_from_mjd (double mjd)
{
  // T is julian centuries since 2000 Jan. 1, 12h UT1
  double T = (mjd - 51544.5) / 36525;

  double day_fraction = mjd - floor (mjd);

  // the GMST in seconds at UT1=0
  double gmst_seconds = ( 24110.54841 + (8640184.812866 * T) + (0.093104 * pow(T,2)) - (0.0000062 * pow(T,3)));

  // convert to days and add the UT1 fraction for today
  double gmst_days = gmst_seconds / 86400.0 + day_fraction;

  gmst_days = gmst_days - floor (gmst_days);

  // convert to radians where 2PI = whole day
  double gmst_radians = gmst_days * 2 * M_PI;

  //fprintf (stderr, "gmst_from_mjd: seconds=%lf days=%lf radians=%lf\n", gmst_seconds, gmst_days, gmst_radians);

  gmst_radians = fmod (gmst_radians, 2 * M_PI);

  return gmst_radians;
}

// convert MJD in UT1 to a LAST
double sofa_gast_from_mjd (double mjd)
{
  const unsigned num_leap_seconds = 26;
  const double   tt_offset = 32.184;

  double dt = (double) num_leap_seconds + tt_offset;
  double jd_base = 2400000.5;
  double tt1, tt2;

  if (iauUt1tt (jd_base, mjd, dt, &tt1, &tt2) != 0)
  {
    fprintf (stderr, "ERROR in UT1 to TT conversion\n");
    return 0;
  }

  double gast = iauGst06a (jd_base, mjd, tt1, tt2);
  return gast;
}


/* gets local mean sidereal time for time_of_day given by mjd  
double gmrt_lmst(double mjd)
{
  double lon = 149 + 25/60.0 + 28.7682/3600;
  double ut,tu,res,lmstime ;
  ut = mjd ;
  tu = (ut - 51544.5) / 36525.;  // centuries since J2000 
  res = ut + lon/360. ; res = res - floor(res) ;
  lmstime = res + (((0.093104 - tu * 6.2e-6) * tu + 8640184.812866) * tu + 24110.54841)/86400.0 ;
  lmstime = (lmstime - floor(lmstime))*2.0 * M_PI ;
  return lmstime;
}
*/

double last_from_mjd (double mjd)
{
  double gast, last;

#ifdef _DEBUG
  int HMSF[4];
  char sign;
  int NDP = 4;    // number of decimal places
#endif

  gast = sofa_gast_from_mjd (mjd);
  last  = gast + MOLONGLO_LONGITUDE;
  
  const double two_pi = 2.0 * M_PI;
  double w = fmod (last, two_pi);
  return ( w >= 0.0 ) ? w : w + two_pi;
}

int mopsr_delays_hhmmss_to_rad (char * hhmmss, double * rads)
{
  int ihour = 0;
  int imin = 0;
  double sec = 0;
  const char *sep = ":";
  char * saveptr;

  char * copy = (char *) malloc (strlen(hhmmss) + 1);
  strcpy (copy, hhmmss);

  char * str = strtok_r(copy, sep, &saveptr);
  if (str != NULL)
  {
    if (sscanf(str, "%d", &ihour) != 1)
      return -1;

    str = strtok_r(NULL, sep, &saveptr);
    if (str != NULL)
    {
      if (sscanf(str, "%d", &imin) != 1)
        return -1;

      str = strtok_r(NULL, sep, &saveptr);
      if (str != NULL)
      {
        if (sscanf(str, "%lf", &sec) != 1)
          return -1;
      }
    }
  }
  free (copy);

  char s = '\0';
  if (ihour < 0)
  {
    ihour *= -1;
    s = '-';
  }

  int status = iauTf2a(s, ihour, imin, sec, rads);

  return status;
}

int mopsr_delays_hhmmss_to_sigproc (char * hhmmss, double * sigproc)
{ 
  int ihour = 0;
  int imin = 0;
  double sec = 0; 
  const char *sep = ":";
  char * saveptr;

  char * copy = (char *) malloc (strlen(hhmmss) + 1);
  strcpy (copy, hhmmss);
  
  char * str = strtok_r(copy, sep, &saveptr);
  if (str != NULL)
  { 
    if (sscanf(str, "%d", &ihour) != 1)
      return -1;

    str = strtok_r(NULL, sep, &saveptr);
    if (str != NULL)
    {
      if (sscanf(str, "%d", &imin) != 1)
        return -1;

      str = strtok_r(NULL, sep, &saveptr);
      if (str != NULL)
      {
        if (sscanf(str, "%lf", &sec) != 1)
          return -1;
      }
    }
  }
  free (copy);

  char s = '\0';
  if (ihour < 0)
  {
    ihour *= -1;
    s = '-';
  }

  *sigproc = ((double)ihour*1e4  + (double)imin*1e2 + sec);
  return 0;
}

int mopsr_delays_ddmmss_to_rad (char * ddmmss, double * rads)
{
  int ideg = 0;
  int iamin = 0;
  double asec = 0;
  const char *sep = ":";
  char * saveptr;

  char * copy = (char *) malloc (strlen(ddmmss) + 1);
  strcpy (copy, ddmmss);

  char * str = strtok_r(copy, sep, &saveptr);
  if (str != NULL)
  {
    if (sscanf(str, "%d", &ideg) != 1)
      return -1;

    str = strtok_r(NULL, sep, &saveptr);
    if (str != NULL)
    {
      if (sscanf(str, "%d", &iamin) != 1)
        return -1;

      str = strtok_r(NULL, sep, &saveptr);
      if (str != NULL)
      {
        if (sscanf(str, "%lf", &asec) != 1)
          return -1;
      }
    }
  }
  free (copy);

  char s = '\0';
  if (ideg < 0)
  {
    ideg *= -1;
    s = '-';
  }

  int status = iauAf2a (s, ideg, iamin, asec, rads);

  return status;
}


int mopsr_delays_ddmmss_to_sigproc (char * ddmmss, double * sigproc)
{
  int ideg = 0;
  int iamin = 0;
  double asec = 0;
  const char *sep = ":";
  char * saveptr;

  char * copy = (char *) malloc (strlen(ddmmss) + 1);
  strcpy (copy, ddmmss);

  char * str = strtok_r(ddmmss, sep, &saveptr);
  if (str != NULL)
  {
    if (sscanf(str, "%d", &ideg) != 1)
      return -1;

    str = strtok_r(NULL, sep, &saveptr);
    if (str != NULL)
    {
      if (sscanf(str, "%d", &iamin) != 1)
        return -1;

      str = strtok_r(NULL, sep, &saveptr);
      if (str != NULL)
      {
        if (sscanf(str, "%lf", &asec) != 1)
          return -1;
      }
    }
  }

  free (copy);

  if (ideg < 0)
    *sigproc = ((double) ideg*1e4 - (double) iamin*1e2) - asec;
  else
    *sigproc = ((double) ideg*1e4  + (double) iamin*1e2) + asec;

  return 0;
}


double ns_tilt (double ha_source, double dec_source, double md_angle)
{
  //fprintf (stderr, "ns_tilt: ha_source=%lf\n", ha_source);
  //fprintf (stderr, "ns_tilt: dec_source=%lf\n", dec_source);
  //fprintf (stderr, "ns_tilt: md_angle=%lf\n", md_angle);

  double NS_Tilt_1 = -0.0000237558704*cos(dec_source)*sin(ha_source);
  double NS_Tilt_2 =  0.578881847*cos(dec_source)*cos(ha_source);
  double NS_Tilt_3 =  0.8154114339*sin(dec_source);
  double NS_Tilt   = asin(( NS_Tilt_1 + NS_Tilt_2 + NS_Tilt_3)/(cos(md_angle)));
  
  return NS_Tilt;
}


double jer_delay(double ha_source, double dec_source,
         double tilt, double skew, double latitude)
{

  return( sin(ha_source)*cos(dec_source)*cos(tilt) + 
          cos(ha_source)*cos(dec_source) * (skew*sin(latitude)-sin(tilt)*cos(latitude)) -
          sin(dec_source)*(skew*cos(latitude)+sin(tilt)*sin(latitude)) );
}

double calc_doc_delay (double RA_curr, double DEC_curr, struct timeval timestamp)
{

  double hour_angle = calc_ha_source (RA_curr, DEC_curr, timestamp);

  double projected_delay = doc_delay (hour_angle, DEC_curr,
                                      MOLONGLO_ARRAY_SLOPE,
                                      MOLONGLO_AZIMUTH_CORR,
                                      MOLONGLO_LATITUDE);

  return projected_delay;
}

double doc_delay(double ha, double dec,
             double tilt, double skew, double lat)
{
    // Differences to jer_delay
    //   1. doesn't use the sin(skew) == skew small angle approximation
    //   2. doesn't use the cos(tilt) == 1 approximation

    return( cos(dec)*sin(ha)*cos(tilt)*cos(skew)
            - cos(dec)*cos(ha) * (sin(lat)*cos(tilt)*sin(skew) + cos(lat)*sin(tilt))
            - sin(dec) * (sin(lat)*sin(tilt) - cos(lat)*cos(tilt)*sin(skew)) );
}


// simple PGPLOT of the delays across the array
int mopsr_delays_plot (unsigned nmod, unsigned nchan, mopsr_delay_t ** delays, struct timeval timestamp)
{
  float * xvals = (float *) malloc(sizeof(float) * nmod);
  float * yvals = (float *) malloc(sizeof(float) * nmod);
  float * yvals2 = (float *) malloc(sizeof(float) * nmod);
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

  return 0;
}
