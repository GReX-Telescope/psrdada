/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 *****************************************************************************/

#ifndef __MOPSR_DELAYS_H
#define __MOPSR_DELAYS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "mopsr_def.h"

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <complex.h>

// degrees to radians
#ifndef DD2R
#define DD2R (M_PI/180.0)
#endif

// arc seconds to degrees
#define AS2D 1.0/(60.0*60.0)

// for the peckham function
#define MOLONGLO_HA_BASELINE   -89.838052 * DD2R 
#define MOLONGLO_DEC_BASELINE  -0.113297  * DD2R

// constants for the site
#define MOLONGLO_LATITUDE     -( 35 + 22/60.0 + 14.5452 / 3600.0) * DD2R /* radians */
#define MOLONGLO_LONGITUDE     (149 + 25/60.0 + 28.7682 / 3600.0) * DD2R /* radians */

//#define MOLONGLO_LATITUDE     -( 35 + 22/60.0 + 18.97/3600.0) * DD2R /* radians */
//#define MOLONGLO_LONGITUDE     (149 + 25/60.0 + 25.25/3600.0) * DD2R /* radians */

#define MOLONGLO_MODULE_LENGTH  4.42
#define MOLONGLO_HALF_MODULE_LENGTH  2.21
#define MOLONGLO_AZIMUTH_CORR  (4.9 * AS2D * DD2R)
#define MOLONGLO_ARRAY_SLOPE   (1.0 / 289.9)
#define MOLONGLO_ARRAY_SLOPE_EAST   (1.0 / 290.80)
#define MOLONGLO_ARRAY_SLOPE_WEST   (1.0 / 289.52)

typedef struct {
  char  name[4];        // bay name
  double sign;          // -1 for east, 1 for west
  double dist;          // distance in metres from array centre to the bay centre
} mopsr_bay_t;

typedef struct {
  char  name[6];        // antenna named [E01-0 -> W44-4]
  double sign;          // -1 for east, 1 for west
  double dist;          // distance in metres from array centre to module centre
  double fixed_delay;   // fixed delay of antenna [seconds]
  float  scale;         // fixed delay of antenna [seconds]
  float  phase_offset;  // fixed phase offset of antenna [radians]
  unsigned bay_idx;     // 0, 1, 2 or 3 for B, G, Y, R
} mopsr_module_t;

typedef struct {
  // PFB ID (WG44 -> WG01 -> EG01 -> EG44)
  char id[7];
  
  // 16 inputs per PFB each begin [E|W][\d\d]-\d
  char modules[MOPSR_MAX_MODULES_PER_PFB][6];
} mopsr_pfb_t;

typedef struct {
  unsigned number;      // coarse filterbank channel number [important!]
  double bw;             // bandwidth of channel
  double cfreq;          // centre frequnecy of channel
} mopsr_chan_t;

typedef struct {
  char name[16];        // source name [e.g. J0437-4715]
  double mjd0;          // fractional mjd to whjich ra, dec refer
  double raj;           // J2000 RA [radians]
  double decj;          // J2000 DEC [radians]
  double ra_curr;       // RA current [radians]
  double dec_curr;      // DEC current [radians]
  double ra_mean;       // mean RA [radians]
  double dec_mean;      // mean DEC [radians]
  double dra;           // RA rate of change [radians/s]
  double ddec;          // DEC rate of change [radians/s]
} mopsr_source_t;

typedef struct delay_vals
{
  unsigned  samples;         // integer samples to delay
  double    fractional;      // fractional samples to delay
  double    fringe_coeff;    // coefficient for complex fringe correction
  double    tot_samps;       // total delay [samples]
  double    tot_secs;        // total delay [seconds]
} mopsr_delay_t;

typedef struct {
  char real;
  char imag;
} complex16;

static inline float sinc(float x)
{
  x *= M_PI;
  if (x == 0)
    return 1.0;
  else
    return sin(x)/x;
}

void sinc_filter(complex16* series_in, complex16* series_out,
                 float * filter, int size, int ntaps, float delay);

void sinc_filter_float (float _Complex * series_in, float _Complex * series_out,
                        float * filter, int size, int ntaps, float delay);

mopsr_module_t * read_modules_file (const char* fname, int * nmod);

mopsr_bay_t * read_bays_file (const char* fname, int *nbay);

mopsr_pfb_t * read_signal_paths_file (const char* fname, int * npfb);

int calculate_delays (unsigned nbay, mopsr_bay_t * bays, 
                      unsigned nmod, mopsr_module_t * mods, 
                      unsigned nchan, mopsr_chan_t * chans,
                      mopsr_source_t source, struct timeval timestamp,
                      mopsr_delay_t ** delays, float starting_md_angle, 
                      char apply_instrumental, char apply_geometric, 
                      char is_tracking, double tsamp);

int cal_app_pos_iau (double RA, double DEC, struct tm * utc, double * RA_app, double * DEC_app);
void calc_observed_pos (double rc, double dc, struct tm * utc, double dut1, double * RA_obs, double * DEC_obs, double * sha);

double mjd_from_utc (struct tm * utc);
double lmst_from_mjd (double mjd);
double last_from_mjd (double mjd);

int mopsr_delays_plot (unsigned nmod, unsigned nchan, mopsr_delay_t ** delays, struct timeval timestamp);

int mopsr_delays_hhmmss_to_rad (char * hhmmss, double * rads);
int mopsr_delays_hhmmss_to_sigproc (char * hhmmss, double * sigprocs);

int mopsr_delays_ddmmss_to_rad (char * ddmmss, double * rads);
int mopsr_delays_ddmmss_to_sigproc (char * ddmmss, double * sigprocs);

void calc_app_ha_dec (double RA_J2000, double DEC_J2000, struct timeval timestamp, double * HA_app, double * DEC_app);

double calc_ha_source ( double RA_curr, double DEC_curr, struct timeval timestamp);
double ns_tilt (double ha_source, double dec_source, double md_tilt);
double jer_delay(double ha_source, double dec_source,
             double tilt, double skew, double latitude);
double calc_jer_delay (double RA, double DEC, struct timeval timestamp);
double calc_jer_delay_west (double RA, double DEC, struct timeval timestamp);
double calc_jer_delay_east (double RA, double DEC, struct timeval timestamp);
double calc_doc_delay (double RA, double DEC, struct timeval timestamp);
double jer_delay(double ha_source, double dec_source, double tilt, double skew, double latitude);
double doc_delay(double ha, double dec, double tilt, double skew, double latitude);


#ifdef __cplusplus
}
#endif

#endif
