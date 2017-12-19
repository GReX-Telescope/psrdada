/*
 * read a file from disk and create the associated images
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>
#include <float.h>
#include <complex.h>
#include <math.h>
#include <cpgplot.h>
#include <fftw3.h>

#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"
#include "mopsr_udp.h"
#ifdef HIRES
#include "mopsr_delays_hires.h"
#else
#include "mopsr_delays.h"
#endif

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

#define CASPSR_NDIM 1

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

void usage ();
void plot_delays (float * yvals, unsigned npts, char * device);
void plot_power_spectrum (float * yvals, unsigned npts, char * device, const char * title, float ymin_in, float ymax_in, char use_pts);
void plot_complex_spectrum (complex float * data, unsigned npts, char * device);
void plot_cross_vs_time (float * amps, float * phases, unsigned npts, char * device, float seconds_per_pt, char * title);
void plot_cross_vs_freq (float * amps, float * phases, unsigned npts, char * device);
void plot_histogram (float * vals, unsigned npts, char * device);
void plot_lags (float * yvals, unsigned npts, char * device);

void usage()
{
  fprintf (stdout,
     "caspsr_dadacorrplot [options] file\n"
     " file must be dual-pol, single channel\n"
     " -b nset         number of ffts in each block [32]\n"
     " -d delay        number of samples to delay files by [default 0]\n"
     " -e turns        fractional turns to delay files2 by [default 0]\n"
     " -f fractional   fractional samples to delay files2 by [default 0]\n"
     " -g dir          read config files from dir\n"
     " -D device       pgplot device name\n"
     " -l ntap         number of FIR taps\n"
     " -n npt          number of points in cross-correlation signal [default 256]\n"
     " -p bin          plot the specified cross-correlation bin's timeseries [default npt/2]\n"
     " -v              be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = "/xs";

  int arg = 0;

  unsigned int npt = 256;

  unsigned nset = 32;

  char * config_dir = 0;

  unsigned int plot_bin = -1;

  int delay = 0;

  float fractional_delay = 0.0;

  float fractional_turns = 0.0;

  int ntap = 25;

  while ((arg=getopt(argc,argv,"b:d:e:f:g:D:hl:n:p:v")) != -1)
  {
    switch (arg)
    {
      case 'b':
        nset = atoi(optarg);
        break;

      case 'd':
        delay = atoi(optarg);
        break;

      case 'e':
        fractional_turns = atof(optarg);
        break;

      case 'f':
        fractional_delay = atof(optarg);
        break;

      case 'g':
        config_dir = strdup(optarg);
        break;

      case 'D':
        device = strdup(optarg);
        break;

      case 'l':
        ntap = atoi(optarg);
        break;

      case 'n':
        npt = atoi(optarg);
        break;

      case 'p':
        plot_bin = atoi(optarg);
        break;

      case 'v':
        verbose++;
        break;

      case 'h':
      default:
        usage ();
        return 0;
    } 
  }
  
  // number of channels in output of FFT
  const unsigned ochan = npt / 2;

  if (config_dir == 0)
    config_dir = "/home/dada/linux_64/share";

  // check and parse the command line arguments
  if (argc-optind != 1)
  {
    fprintf(stderr, "ERROR: 1 command line argument is required\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  char file[1024];

  strcpy(file, argv[optind]);

  size_t file_size;
  struct stat buf;
  if (stat (file, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat dada file [%s]: %s\n", file, strerror(errno));
    return (EXIT_FAILURE);
  }
  file_size = buf.st_size;
  if (verbose)
    fprintf (stderr, "filesize for %s is %d bytes\n", file, file_size);

  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd = open (file, flags, perms);
  if (fd < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", file, strerror(errno));
    exit(EXIT_FAILURE);
  }

  const size_t dada_header_size = 4096;
  char * header = (char *) malloc(dada_header_size);

  size_t bytes_read = read (fd, (void *) header, dada_header_size);
  if (bytes_read != dada_header_size)
  {
    fprintf (stderr, "failed to read header from %s\n", file);
    close(fd);
    return (EXIT_FAILURE);
  }

  int nchan = 0;
  if (ascii_header_get( header, "NCHAN", "%d", &nchan) != 1)
  {
    fprintf (stderr, "failed to read nchan from header\n");
    close(fd);
    return (EXIT_FAILURE);
  }
  if (nchan != 1)
  {
    fprintf (stderr, "CASPSR does not support NCHAN != 1 input\n");
    close (fd);
    return (EXIT_FAILURE);
  }

  float tsamp;
  if (ascii_header_get( header, "TSAMP", "%f", &tsamp) != 1)
  {
    fprintf (stderr, "failed to read TSAMP from header\n");
    close(fd);
    return (EXIT_FAILURE);
  }

  float freq;
  if (ascii_header_get( header, "FREQ", "%f", &freq) != 1)
  {
    fprintf (stderr, "failed to read FREQfrom header\n");
    close(fd);
    return (EXIT_FAILURE);
  }

  char position[32];
  mopsr_source_t source;

  if (ascii_header_get (header, "RA", "%s", position) != 1)
  {
    fprintf (stderr, "could not read RA from header\n");
    return -1;
  }

  if (mopsr_delays_hhmmss_to_rad (position, &(source.raj)) < 0)
  {
    fprintf(stderr, "could not parse RA from %s\n", position);
    return -1;
  }

  if (ascii_header_get (header, "DEC", "%s", position) != 1)
  {
    fprintf(stderr, "could not read RA from header\n");
    return -1;
  }
  if (mopsr_delays_ddmmss_to_rad (position, &(source.decj)) < 0)
  {
    fprintf(stderr, "could not parse DEC from %s\n", position);
    return -1;
  }

  char tmp[32];
  if (ascii_header_get (header, "UTC_START", "%s", tmp) != 1)
  {
    fprintf (stderr, "Could not read UTC_START from header\n");
    return -1;
  }
  fprintf(stderr, "UTC_START=%s\n", tmp);
  time_t utc_start = str2utctime (tmp);

  float ut1_offset;
  if (ascii_header_get (header, "UT1_OFFSET", "%f", &ut1_offset) != 1)
  {
    fprintf (stderr, "Could not read UT1_OFFSET from header\n");
    return -1;
  }
  fprintf (stderr, "UT1_OFFSET=%f\n", ut1_offset);

  uint64_t obs_offset;
  if (ascii_header_get (header, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1)
  {
    fprintf (stderr, "could not read OBS_OFFSET header\n");
    return -1;
  }

  int chan_offset;
  if (ascii_header_get (header, "CHAN_OFFSET", "%d", &chan_offset) != 1)
  {
    fprintf (stderr, "could not read CHAN_OFFSET header\n");
    return -1;
  }

  char pfb_id_str1[5];
  if (ascii_header_get (header, "PFB_ID1", "%s", pfb_id_str1) != 1)
  {
    fprintf(stderr, "could not read PFB_ID from header\n");
    return -1;
  }
  char pfb_id_str2[5];
  if (ascii_header_get (header, "PFB_ID2", "%s", pfb_id_str2) != 1)
  {
    fprintf(stderr, "could not read PFB_ID from header2\n");
    return -1;
  }

  int ant_id1;
  if (ascii_header_get (header, "ANT_ID1", "%d", &ant_id1) != 1)
  {
    fprintf(stderr, "could not read ANT_ID from header\n");
    return -1;
  }
  int ant_id2;
  if (ascii_header_get (header, "ANT_ID2", "%d", &ant_id2) != 1)
  {     
    fprintf(stderr, "could not read ANT_ID from header2\n");
    return -1;    
  }                 

  float start_md_angle;
  if (ascii_header_get (header, "MD_ANGLE", "%f", &start_md_angle) != 1)
  {
    fprintf(stderr, "could not read MD_ANGLE from header\n");
    start_md_angle = 0;
  }

  float bw;
  if (ascii_header_get (header, "BW", "%f", &bw) != 1)
  {
    fprintf(stderr, "could not read BW from header\n");
    bw = 0;
  }

  float freq;
  if (ascii_header_get (header, "FREQ", "%f", &freq) != 1)
  {
    fprintf(stderr, "could not read FREQ from header\n");
    freq = 0;
  }

  uint64_t bytes_per_second;
  if (ascii_header_get (header, "BYTES_PER_SECOND", "%"PRIu64, &bytes_per_second) != 1)
  {
    fprintf(stderr, "could not read BYTES_PER_SECOND from header\n");
    bytes_per_second = (uint64_t) ((double) (nant * CASPSR_NDIM * 1e6) / tsamp);
  }

  unsigned iblock, ichan, idat, ipt, ibatch;

  const unsigned half_ntap = ntap / 2;
  unsigned ndat = nset * npt;
  const int fixed_delay = half_ntap + 5;
  const int overlap = ntap + fixed_delay;
  const int ndat_overlap = ndat + overlap;

  // for reading raw blocks of data from input file
  unsigned raw_size = nant * ndat_overlap * CASPSR_NDIM;
  int8_t * raw = (int8_t *) malloc (raw_size);

  unsigned block_size = nant * ndat * CASPSR_NDIM;
  const unsigned nblocks = (file_size - 4096) / block_size;
  if (verbose)
    fprintf (stderr, "main: file_size=%u raw_size=%u nblocks=%u\n", (file_size - 4096), block_size, nblocks);

  // width of 1 sample
  size_t samp_stride = CASPSR_NDIM;
  off_t offset1 = dada_header_size;
  off_t offset2 = dada_header_size;

  if (delay < 0)
    offset1 += (abs(delay) * samp_stride);
  if (delay > 0)
    offset2 += (abs(delay) * samp_stride);

  fprintf (stderr, "delay=%d offset1=%d offset2=%d\n", delay, offset1, offset2);

  //lseek (fd1, offset1, SEEK_SET);
  //lseek (fd2, offset2, SEEK_SET);

  /*
  if (delay)
  {
    if (delay > 0)
    {
      off_t offset = delay * samp_stride;
      lseek (fd2, offset, SEEK_CUR);
      fprintf (stderr, "FD2 seeking %d samples (%d bytes)\n", delay, offset);
    }
    else
    {
      off_t offset = (-1 * delay * samp_stride);
      lseek (fd1, offset, SEEK_CUR);
      fprintf (stderr, "FD1 seeking %d samples, (%d bytes)\n", delay, offset);
    }
  }
*/

  // unpacked data size (npt * nsamp)
  size_t unpacked_size = ndat_overlap * sizeof(fftwf_complex);
  complex float * unpacked1 = (complex float *) fftwf_malloc (unpacked_size);
  complex float * unpacked2 = (complex float *) fftwf_malloc (unpacked_size);

  // tapped data size (unpacked - ntap)
  size_t tapped_size = ndat_overlap * sizeof(complex float);
  complex float * tapped1 = (complex float *) fftwf_malloc (tapped_size);
  complex float * tapped2 = (complex float *) fftwf_malloc (tapped_size);

  // FFT output
  size_t spectrum_size = ochan * sizeof(complex float);
  complex float * spectrum1 = (complex float *) malloc (spectrum_size);
  complex float * spectrum2 = (complex float *) malloc (spectrum_size);

  // cross correlation 
  size_t xcorr_size = ochan * sizeof(complex float);
  complex float * xcorr = (complex float *) malloc (xcorr_size);
  bzero (xcorr, xcorr_size);

  complex float * lag = (complex float *) malloc (xcorr_size);
  float * lag_power = (float *) malloc (ochan * sizeof(float));

  // amps, phases and power spectra with full frequnecy resolution
  size_t nbytes = sizeof(float) * ochan;
  float * amps   = (float *) malloc (nbytes);
  float * phases = (float *) malloc (nbytes);
  float * power1 = (float *) malloc (nbytes);
  float * power2 = (float *) malloc (nbytes);

  bzero (power1, nbytes);
  bzero (power2, nbytes);

  // amps and phases with time resolution
  nbytes = sizeof(float) * nblocks;
  float * amps_t = (float *) malloc(sizeof(float) * nblocks);
  float * phases_t = (float *) malloc(sizeof(float) * nblocks);

  // FWD filterbank plan TODO R2C
  fftwf_plan plan_fwd  = fftwf_plan_dft_1d (npt, unpacked1[0], spectrum1, FFTW_FORWARD, FFTW_ESTIMATE);
  fftwf_plan plan_bwd  = fftwf_plan_dft_1d (ochan, xcorr, lag, FFTW_BACKWARD, FFTW_ESTIMATE);

  // memory for FIR 
  float * filter = (float *) malloc (ntap * sizeof(float));

  // number of seconds per time point 
  float seconds_per_pt = (float) (nset * ochan * tsamp) / 1e6;
  fprintf (stderr, "bytes_per_second=%lu seconds_per_pt=%f \n", bytes_per_second, seconds_per_pt);

  uint64_t total_bytes_read = 0;

  int nbays;
  const char * bays_file = "/home/dada/linux_64/share/molonglo_bays.txt";
  mopsr_bay_t * all_bays = read_bays_file (bays_file, &nbays);

  // read the modules file
  int nmodules;
  const char * modules_file = "/home/dada/linux_64/share/molonglo_modules.txt";
  mopsr_module_t * all_modules = read_modules_file (modules_file, &nmodules);

  // read the signal paths file
  int npfbs;
  char signal_paths_file[1024];
  sprintf (signal_paths_file, "%s/mopsr_signal_paths.txt", config_dir);
  mopsr_pfb_t * pfbs = read_signal_paths_file (signal_paths_file, &npfbs);
  fprintf (stderr, "signal_paths=%s, npfbs=%d\n", signal_paths_file, npfbs);

  const unsigned nant= 2;
  unsigned iant;

#ifdef HIRES
  size_t delays_size = sizeof(mopsr_delay_hires_t *) * nant;
  mopsr_delay_hires_t ** delays = (mopsr_delay_hires_t **) malloc (delays_size);
  for (iant=0; iant<nant; iant++)
    delays[iant] = (mopsr_delay_hires_t *) malloc (sizeof(mopsr_delay_hires_t) * nchan);
#else
  size_t delays_size = sizeof(mopsr_delay_t *) * nant;
  mopsr_delay_t ** delays = (mopsr_delay_t **) malloc (delays_size);
  for (iant=0; iant<nant; iant++)
    delays[iant] = (mopsr_delay_t *) malloc (sizeof(mopsr_delay_t) * nchan);
#endif

  size_t modules_ptr_size = sizeof(mopsr_module_t) * nant;
  mopsr_module_t * modules = (mopsr_module_t *) malloc(modules_ptr_size);

  unsigned ipfb;
  unsigned pfb_id1 = -1;
  unsigned pfb_id2 = -1;
  for (ipfb=0; ipfb<MOPSR_MAX_PFBS; ipfb++)
  {
    if (strcmp(pfbs[ipfb].id, pfb_id_str1) == 0)
      pfb_id1 = ipfb;
    if (strcmp(pfbs[ipfb].id, pfb_id_str2) == 0)
      pfb_id2 = ipfb;
  }
  if (pfb_id1 == -1)
  {
    fprintf(stderr, "alloc: failed to find pfb_id_str1=%s in signal path configuration file\n",  pfb_id_str1);
    return -1;
  }
  if (pfb_id2 == -1) 
  {
    fprintf(stderr, "alloc: failed to find pfb_id_str2=%s in signal path configuration file\n",  pfb_id_str2);
    return -1;
  }

  if (verbose)
  {
    fprintf (stderr, "file: pfb_id_str=%s -> pfb_id=%d ant_id=%d\n", pfb_id_str1, pfb_id1, ant_id1);
    fprintf (stderr, "file2: pfb_id_str=%s -> pfb_id=%d and_id=%d\n", pfb_id_str2, pfb_id2, ant_id2);
  }

  unsigned ipfb_mod, imod;
  for (ipfb_mod=0; ipfb_mod < MOPSR_MAX_MODULES_PER_PFB; ipfb_mod++)
  {
    if (strcmp(pfbs[pfb_id1].modules[ipfb_mod], "-") != 0)
    {
      for (imod=0; imod<nmodules; imod++)
        if (strcmp(pfbs[pfb_id1].modules[ant_id1],all_modules[imod].name) == 0)
        {
          modules[0] = all_modules[imod];
        }
    }
    if (strcmp(pfbs[pfb_id2].modules[ipfb_mod], "-") != 0)
    {
      for (imod=0; imod<nmodules; imod++)
        if (strcmp(pfbs[pfb_id2].modules[ant_id2],all_modules[imod].name) == 0)
          modules[1] = all_modules[imod];
    }
  }

  double baseline = abs(modules[1].dist - modules[0].dist);
  fprintf (stderr, "file: module.name=%s dist=%lf\n", modules[0].name, modules[0].dist);
  fprintf (stderr, "file2: module.name=%s dist=%lf\n", modules[1].name, modules[1].dist);
  fprintf (stderr, "baseline = %lf\n", baseline);

  const double channel_bw = (double) 1.0 / tsamp;
  mopsr_chan_t * channels = (mopsr_chan_t *) malloc(sizeof(mopsr_chan_t));
  channels[0].number = 0;
  channels[0].bw     = channel_bw;
  channels[0].cfreq  = (double) freq;
  fprintf (stderr, "bw=%lf freq=%lf\n", channels[ichan].bw, channels[ichan].cfreq); }

  // get initial delays
  char title[128];

  char apply_instrumental = 1;
  char apply_geometric = 1;
  char is_tracking = 0;

  // get initial integer delays
  total_bytes_read = obs_offset;

  // add any fractional delay to the system
  //modules[1].fixed_delay += (fractional_delay * tsamp) / 1e6;
  modules[1].fixed_delay += (fractional_turns * (1.0 / bw)) / 1e6;

  float max_power = 0;
  int max_idx = 0;

  struct timeval timestamp;
  double obs_offset_seconds = (double) obs_offset / bytes_per_second;
  time_t now = utc_start + obs_offset_seconds;
  struct tm * utc = gmtime (&now);
  cal_app_pos_iau (source.raj, source.decj, utc, &(source.ra_curr), &(source.dec_curr));

  for (iblock=0; iblock<nblocks; iblock++)
  {
    complex float xcorr_fscr = 0 + 0 * I;


    // middle byte of block
    const double mid_byte = (double) total_bytes_read + (block_size/ 2);

    // determine the timestamp corresponding to the middle byte of this block
    obs_offset_seconds = mid_byte / (double) bytes_per_second;

    // the time (seconds) that we wish to compute for
    double ut1_time = (double) utc_start + obs_offset_seconds + (double) ut1_offset;

    timestamp.tv_sec = (long) floor (ut1_time);
    timestamp.tv_usec = (long) floor ((ut1_time - (double) timestamp.tv_sec) * 1e6);

    if (verbose > 1)
    {
      fprintf (stderr, "obs_offset_seconds=%lf, timestamp=%ld.%ld\n", 
                        obs_offset_seconds, timestamp.tv_sec, timestamp.tv_usec);
    }

#ifdef HIRES
    if (calculate_delays_hires ( nbays, all_bays, nant, modules, nchan, channels,
         source, timestamp, delays, start_md_angle, apply_instrumental,
         apply_geometric, is_tracking, tsamp) < 0)
#else
    if (calculate_delays ( nbays, all_bays, nant, modules, nchan, channels,
         source, timestamp, delays, start_md_angle, apply_instrumental,
         apply_geometric, is_tracking, tsamp) < 0)
#endif
    {
      fprintf (stderr, "failed to calculate delays!\n");
      return -1;
    }

    if (verbose > 1)
    {
      ichan = 0;
      fprintf (stderr, "1: tot_secs=%le tot_samps=%7.4f samples=%u "
                       "fractional=%7.4f fringe_coeff=%7.4lf\n", 
                       delays[0][ichan].tot_secs,
                       delays[0][ichan].tot_samps,
                       delays[0][ichan].samples,
                       delays[0][ichan].fractional,
                       delays[0][ichan].fringe_coeff);

      fprintf (stderr, "2: tot_secs=%le tot_samps=%7.4f samples=%u "
                       "fractional=%7.4f fringe_coeff=%7.4lf\n", 
                       delays[1][ichan].tot_secs,
                       delays[1][ichan].tot_samps,
                       delays[1][ichan].samples,
                       delays[1][ichan].fractional,
                       delays[1][ichan].fringe_coeff);
    }

    // read a block of data from each file, including ntap extra samples
    if (verbose > 1)
      fprintf (stderr, "[%d] reading %ld bytes from files\n", iblock, raw_size);

    bytes_read = read (fd, (void *) raw, raw_size);

    // unpack the samples from each input
    uint64_t i = 0;
    const float scale = 127.5;
    for (idat=0; idat<ndat_overlap; idat+=4)
    {
      for (ival=0; ival<4; ival++)
      {
        unpacked1[idat+ival] = (((float) raw[i+ival]) + 0.38) / scale;
      }
      i += 4; 

      for (ival=0; ival<4; ival++)
      {
        unpacked2[idat+ival] = (((float) raw[i+ival]) + 0.38) / scale;
      }
      i += 4; 
    }
    if (verbose > 1)
      fprintf (stderr, "[%d] unpacked %lu samples from raw\n", iblock, i);

    // integer + fractional delay
    float f1 = (float) delays[0][0].fractional;
    float f2 = (float) delays[1][0].fractional;

    int d1 = delays[0][0].samples;
    int d2 = delays[1][0].samples;

    complex float * delayed1 = unpacked1 + d1;
    complex float * delayed2 = unpacked2 + d2;

    sinc_filter_float (delayed1, tapped1, filter, ndat + 2*half_ntap, ntap, f1);
    sinc_filter_float (delayed2, tapped2, filter, ndat + 2*half_ntap, ntap, f2);

    // fringe rotation
    const float fringe1 = (float) delays[0][0].fringe_coeff;
    const float fringe2 = (float) delays[1][0].fringe_coeff;

    const complex float phasor1 = cos(fringe1) + sin(fringe1) * I;
    const complex float phasor2 = cos(fringe2) + sin(fringe2) * I;

    for (idat=0; idat<ndat; idat++)
    {
      tapped1[idat] = tapped1[idat + half_ntap] * phasor1;
      tapped2[idat] = tapped2[idat + half_ntap] * phasor2;
    }

    // Forward FFT each channel
    const unsigned nbatch = ndat / npt;
    fftwf_complex * out1 = (fftwf_complex *) spectrum1;
    fftwf_complex * out2 = (fftwf_complex *) spectrum2;

    fftwf_complex * in1 = (fftwf_complex *) tapped1;
    fftwf_complex * in2 = (fftwf_complex *) tapped2;

    for (ibatch=0; ibatch<nbatch; ibatch++)
    {
      fftwf_execute_dft (plan_fwd, in1, out1);
      fftwf_execute_dft (plan_fwd, in2, out2);

      // zap dc channels
      //out1[0] = 0 + 0 * I;
      //out2[0] = 0 + 0 * I;

      // compute cross correlation products
      for (ipt=0; ipt<npt; ipt++)
      {
        // fft shift
        const unsigned opt = (ipt + (ochan/2)) % npt;

        // normalise 
        const complex float p1 = out1[opt] / ochan;
        const complex float p2 = out2[opt] / ochan;
        const complex float cross_power = p1 * conj(p2);

        // power spectrum for each input [accumulated over all samples]
        power1[ipt] += cabsf(p1);
        power2[ipt] += cabsf(p2);

        xcorr[ipt] += cross_power;
        xcorr_fscr += cross_power;
      }

      in1 += npt;
      in2 += npt;
    }

    // inverse fft cross correlation
    fftwf_execute_dft (plan_bwd, xcorr, lag);
    
    // compute the xcorr phase and amplitude [vs time]
    phases_t[iblock] = cargf(xcorr_fscr) * (180/M_PI);
    amps_t[iblock]   = cabsf(xcorr_fscr);

    // compute the xcorr phase and amplitude [vs freq]
    for (ipt=0; ipt<ochan; ipt++)
    {
      unsigned idx = ipt;
      unsigned odx = (idx + (nchan*npt/2)) % (nchan*npt);

      phases[idx] = cargf(xcorr[idx]);
      amps[idx]   = cabsf(xcorr[idx]);

      lag_power[odx] = cabsf(lag[idx]);
      if (lag_power[odx] > max_power)
      {
        max_power = lag_power[odx];
        max_idx = odx - (nchan * npt / 2);
      }
    }

    sprintf (title, "X-Corr %5.3lf MHz bw centred at %5.3lf MHz over %5.2lf m", bw, freq, baseline);
    plot_power_spectrum (power1, ochan, "1/xs", "Bandpass A", 0, -1, 0);
    plot_power_spectrum (power2, ochan, "2/xs", "Bandpass B", 0, -1, 0);
    plot_cross_vs_freq (amps, phases, ochan, "3/xs");
    plot_lags(lag_power, ochan, "4/xs");

    fprintf (stderr, "Delay peak = %d turns\n", max_idx);

    if (iblock > 0)
    {
      plot_cross_vs_time (amps_t, phases_t, iblock+1, "5/xs", seconds_per_pt, title);
    }

    // now fseek back into the file by ntap/2
    offset1 += block_size;
    offset2 += block_size;

    lseek (fd1, offset1, SEEK_SET);
    lseek (fd2, offset2, SEEK_SET);

    total_bytes_read += block_size;
    if (verbose > 1)
      fprintf (stderr, "processed total of %lu bytes\n", total_bytes_read);
  }

  if (verbose)
    fprintf (stderr, "freeing allocated memory\n");

  if (filter)
    free (filter);
  free (raw);

  free (unpacked1);
  free (unpacked2);
  free (tapped1);
  free (tapped2);
  free (spectrum1);
  free (spectrum2);
  free (xcorr);

  free (power1);
  free (power2);
  free (amps);
  free (phases);
  free (amps_t);
  free (phases_t);

  fftwf_destroy_plan (plan_fwd);
  fftwf_destroy_plan (plan_bwd);

  close (fd1);
  close (fd2);

  return EXIT_SUCCESS;
}


void plot_cross_vs_time (float * amps, float * phases, unsigned npts, char * device, float seconds_per_pt, char * title)
{
  float xmin = 0;
  float xmax = (float) npts * seconds_per_pt;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float xvals[npts];
  char label[64];

  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (amps[i] > ymax)
      ymax = amps[i];
    if (amps[i] < ymin)
      ymin = amps[i];

    xvals[i] = (float) i * seconds_per_pt;
  }

  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  cpgbbuf();

  cpgswin(xmin, xmax, ymin, ymax);
  cpgsvp(0.1, 0.9, 0.5, 0.9);
  cpgbox("BCST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("", "Amplitude", title);

  cpgsci(3);
  cpgslw(5);
  cpgpt(npts, xvals, amps, -1);
  cpgline(npts, xvals, amps);
  cpgslw(1);
  cpgsci(1);

  ymin = FLT_MAX;
  ymax = -FLT_MAX;

  for (i=0; i<npts; i++)
  {
    if (phases[i] > ymax)
      ymax = phases[i];
    if (phases[i] < ymin)
      ymin = phases[i];
  }

  cpgswin(xmin, xmax, -180, 180);
  cpgsvp(0.1, 0.9, 0.1, 0.5);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Time (seconds)", "Phase", "");

  cpgsci(3);
  cpgslw(5);
  cpgpt(npts, xvals, phases, -1);
  cpgslw(1);

  cpgebuf();
  cpgend();
}

void plot_cross_vs_freq (float * cross, float * phases, unsigned npts, char * device)
{
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  float xmin = 0;
  float xmax = (float) npts;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float xvals[npts];

  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (cross[i] > ymax)
      ymax = cross[i];
    if (cross[i] < ymin)
      ymin = cross[i];

    xvals[i] = (float) i;
  }

  cpgbbuf();

  cpgswin(xmin, xmax, 0, ymax);

  cpgsvp(0.05, 0.95, 0.5, 0.9);
  cpgbox("BCST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("", "", "Cross Power Spectrum");

  cpgsci(2);
  cpgline(npts, xvals, cross);
  cpgsci(1);

  cpgswin(xmin, xmax, (-1* M_PI), M_PI);
  //cpgswin(xmin, xmax, ymin, ymax);
  cpgsvp(0.05, 0.95, 0.1, 0.5);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Channel", "Phase", "");

  cpgsci(2);
  cpgpt(npts, xvals, phases, -2);

  cpgebuf();
  cpgend();

}



void plot_delays (float * yvals, unsigned npts, char * device)
{
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  float half_npts = (float) npts / 2;

  float xmin = -1 * half_npts;
  float xmax = half_npts;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float xvals[npts];
  unsigned xpeak = 0;

  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (yvals[i] > ymax)
    {
      ymax = yvals[i];
      xpeak = i;
    }

    if (yvals[i] < ymin)
      ymin = yvals[i];
    xvals[i] = (float) i - half_npts;
  }

  fprintf (stderr, "xpeak=%d LAG=%d samples\n", xpeak, xpeak - (npts / 2));

  cpgbbuf();
  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  cpglab("", "", "");

  cpgsci(2);

  cpgpt(npts, xvals, yvals, -2);

  cpgebuf();
  cpgend();
}


void plot_power_spectrum (float * yvals, unsigned npts, char * device, const char * title, float ymin_in, float ymax_in, char use_pts)
{
  float xmin = 0;
  float xmax = npts;
  float ymin, ymax;
  float xvals[npts];
  unsigned xpeak = 0;
  unsigned i;

  if (ymin_in == -1)
    ymin = FLT_MAX;
  else
    ymin = ymin_in;

  if (ymax_in == -1)
    ymax = -FLT_MAX;
  else
    ymax = ymax_in;

  for (i=0; i<npts; i++)
  {
    if ((ymax_in == -1) && (yvals[i] > ymax))
    {
      ymax = yvals[i];
      xpeak = i;
    }

    if ((ymin_in == -1) && (yvals[i] < ymin))
    {
      ymin = yvals[i];
    }

    xvals[i] = 800.0 + (((float) i / npts) * 100);
    xvals[i] = (float) i;
  }

  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  cpgbbuf();
  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  cpglab("Freq Channel", "Flux", title);

  cpgsci(2);

  if (use_pts)
    cpgpt(npts, xvals, yvals, -3);
  else
    cpgline(npts, xvals, yvals);

  cpgebuf();
  cpgend();
}


void plot_complex_spectrum (complex float * data, unsigned npts, char * device)
{

  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  float xmin = 0;
  float xmax = npts;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float xvals[npts];
  float yvals[npts];

  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (creal(data[i]) > ymax) 
    {
      ymax = creal(data[i]);
    }

    if (cimag(data[i]) > ymax)
    {
      ymax = cimag(data[i]);
    }

    if (creal(data[i]) < ymin) 
    {
      ymin = creal(data[i]);
    }

    if (cimag(data[i]) < ymin) 
    {
      ymin = cimag(data[i]);
    }

    xvals[i] = (float) i;
  }

  cpgbbuf();
  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  cpglab("", "", "");

  cpgsci(2);
  for (i=0; i<npts; i++)
    yvals[i] = creal(data[i]);
  cpgline(npts, xvals, yvals);

  cpgsci(3);
  for (i=0; i<npts; i++)
    yvals[i] = cimag(data[i]);
  cpgline(npts, xvals, yvals);

  cpgebuf();
  cpgend();
}

void plot_histogram (float * vals, unsigned npts, char * device)
{
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

/*
  unsigned i;
  unsigned hist[256];
  float ymin = 0;
  float ymax = 2 * M_PI;
  float val;

  for (i=0; i<256; i++)
    hist[i] = 0;

  for (i=0; i<npts; i++)
  {
    val = vals[i] + M_PI;
    val /= (2*M_PI);
    val *= 256;
    bin = (unsigned) floor(val);
    fprintf (stderr, "bin %d ++ \n", bin);
    hist[bin]++;
  }

  for (i=0; i<256; i++)
  {
    if (hist[i] > ymax)
      ymax = hist[i];
  }

  float xmin = -1 * M_PI;
  float xmax = M_PI;

  cpgbbuf();
  cpgswin(ymin, ymax
*/

  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;
  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (vals[i] > ymax) ymax = vals[i];
    if (vals[i] < ymin) ymin = vals[i];
  }

  int N = (int) npts;
  int nbin = 128;
  int flag = 0;
  cpgbbuf();
  cpghist (N, vals, ymin, ymax, nbin, flag);
  cpgebuf();
  cpgend();

}

void plot_lags (float * yvals, unsigned npts, char * device)
{
  if (cpgbeg(0, device, 1, 1) != 1)
  { 
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  int offset = (npts/2);
  
  float xmin = -1 * (float) offset;
  float xmax = 1 * (float) offset;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;
  
  float xvals[npts];
  int xpeak = 0;
  
  int i;
  for (i=0; i<npts; i++)
  {
    if (yvals[i] > ymax)
    { 
      ymax = yvals[i];
      xpeak = i - offset;
    }
    
    if (yvals[i] < ymin)
      ymin = yvals[i];
    
    xvals[i] = (float) (i - offset);
  }
  
  //xmin = -500;
  //xmax = 500;

  cpgbbuf();
  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  cpglab("Sample", "Strength", "Integer Sample Delays");
  
  cpgsci(1); 
  cpgsls(5);
  cpgline(npts, xvals, yvals);
  
  cpgsci(5);
  cpgsls(1);
  cpgpt(npts, xvals, yvals, -5);

  cpgebuf();
  cpgend();
}
