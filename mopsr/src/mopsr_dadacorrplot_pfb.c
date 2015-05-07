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
#include "mopsr_delays.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

void usage ();
void plot_delays (float * yvals, unsigned npts, char * device);
void plot_power_spectrum (float * yvals, unsigned npts, char * device, const char * title, float ymin_in, float ymax_in, char use_pts);
void plot_complex_spectrum (fftwf_complex * data, unsigned npts, char * device);
void plot_cross_vs_time (float * amps, float * phases, unsigned npts, char * device, float seconds_per_pt, char * title);
void plot_cross_vs_freq (float * amps, float * phases, unsigned npts, char * device);
void plot_histogram (float * vals, unsigned npts, char * device);

void usage()
{
  fprintf (stdout,
     "mopsr_dadacorrplot_pfb [options] file1 file2\n"
     " files must be single antenna, multi channel files\n"
     " -b blocksize    number of bytes in which to block data\n"
     " -c channel      channel to use [default 0]\n"
     " -d delay        number of samples to delay files by [default 0]\n"
     " -f fractional   fractional sample to delay files by [default 0]\n"
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

  unsigned block_size = 1048576;


  unsigned int plot_bin = -1;

  int delay = 0;

  float fractional_delay = 0.0;

  int channel = 0;

  int ntap = 15;

  while ((arg=getopt(argc,argv,"b:c:d:f:D:hl:n:p:v")) != -1)
  {
    switch (arg)
    {
      case 'b':
        block_size = atoi(optarg);
        break;

      case 'c':
        channel = atoi(optarg);
        break;

      case 'd':
        delay = atoi(optarg);
        break;

      case 'f':
        fractional_delay = atof(optarg);
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

  // check and parse the command line arguments
  if (argc-optind != 2)
  {
    fprintf(stderr, "ERROR: 2 command line arguments are required\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  char file1[1024];
  char file2[1024];

  strcpy(file1, argv[optind]);
  strcpy(file2, argv[optind+1]);

  size_t file1_size, file2_size;
  struct stat buf;
  if (stat (file1, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat dada file [%s]: %s\n", file1, strerror(errno));
    return (EXIT_FAILURE);
  }
  file1_size = buf.st_size;
  if (verbose)
    fprintf (stderr, "filesize for %s is %d bytes\n", file1, file1_size);

  if (stat (file2, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat dada file [%s]: %s\n", file2, strerror(errno));
    return (EXIT_FAILURE);
  }
  file2_size = buf.st_size;
  if (verbose)
    fprintf (stderr, "filesize for %s is %d bytes\n", file2, file2_size);

  if (file1_size != file2_size)
  {
    fprintf (stderr, "file sizes differed\n");
    return (EXIT_FAILURE);
  }

  off_t offset;
  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd1 = open (file1, flags, perms);
  if (fd1 < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", file1, strerror(errno));
    exit(EXIT_FAILURE);
  }

  if (verbose)
    fprintf (stderr, "reading file2\n");
  int fd2 = open (file2, flags, perms);
  if (fd2 < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", file2, strerror(errno));
    exit(EXIT_FAILURE);
  }

  const size_t dada_header_size = 4096;
  char * header1 = (char *) malloc(dada_header_size);
  char * header2 = (char *) malloc(dada_header_size);

  size_t bytes_read = read (fd1, (void *) header1, dada_header_size);
  if (bytes_read != dada_header_size)
  {
    fprintf (stderr, "failed to read header from %s\n", file1);
    close(fd1);
    close(fd2);
    return (EXIT_FAILURE);
  }

  bytes_read = read (fd2, (void *) header2, dada_header_size);
  if (bytes_read != dada_header_size)
  { 
    fprintf (stderr, "failed to read header from %s\n", file2);
    close(fd1);
    close(fd2);
    return (EXIT_FAILURE);
  }

  int nchan = 0;
  if (ascii_header_get( header1, "NCHAN", "%d", &nchan) != 1)
  {
    fprintf (stderr, "failed to read nchan from header1\n");
    close(fd1);
    close(fd2);
    return (EXIT_FAILURE);
  }

  float tsamp;
  if (ascii_header_get( header1, "TSAMP", "%f", &tsamp) != 1)
  {
    fprintf (stderr, "failed to read nchan from header1\n");
    close(fd1);
    close(fd2);
    return (EXIT_FAILURE);
  }

  char position[32];
  mopsr_source_t source;

  if (ascii_header_get (header1, "RA", "%s", position) != 1)
  {
    fprintf (stderr, "could not read RA from header\n");
    return -1;
  }

  if (mopsr_delays_hhmmss_to_rad (position, &(source.raj)) < 0)
  {
    fprintf(stderr, "could not parse RA from %s\n", position);
    return -1;
  }

  if (ascii_header_get (header1, "DEC", "%s", position) != 1)
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
  if (ascii_header_get (header1, "UTC_START", "%s", tmp) == 1)
  {
    fprintf(stderr, "open: UTC_START=%s\n", tmp);
  }
  else
  {
    return -1;
  }

  uint64_t obs_offset;
  if (ascii_header_get (header1, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1)
  {
    fprintf (stderr, "could not read OBS_OFFSET header\n");
    return -1;
  }

  int chan_offset;
  if (ascii_header_get (header1, "CHAN_OFFSET", "%d", &chan_offset) != 1)
  {
    fprintf (stderr, "could not read CHAN_OFFSET header\n");
    return -1;
  }

  char pfb_id_str1[5];
  if (ascii_header_get (header1, "PFB_ID", "%s", pfb_id_str1) != 1)
  {
    fprintf(stderr, "could not read PFB_ID from header1\n");
    return -1;
  }
  char pfb_id_str2[5];
  if (ascii_header_get (header2, "PFB_ID", "%s", pfb_id_str2) != 1)
  {
    fprintf(stderr, "could not read PFB_ID from header2\n");
    return -1;
  }

  int ant_id1;
  if (ascii_header_get (header1, "ANT_ID", "%d", &ant_id1) != 1)
  {
    fprintf(stderr, "could not read ANT_ID from header1\n");
    return -1;
  }
  int ant_id2;
  if (ascii_header_get (header2, "ANT_ID", "%d", &ant_id2) != 1)
  {     
    fprintf(stderr, "could not read ANT_ID from header2\n");
    return -1;    
  }                 

  // convert UTC_START to a unix UTC
  time_t utc_start = str2utctime (tmp);

  size_t bytes_packed = nchan * npt * MOPSR_NDIM;
  fprintf (stderr, "bytes_packed=%ld\n", bytes_packed);

  unsigned iblock;
  const unsigned nblocks = ((file1_size - 4096) / nchan) / block_size;
  fprintf (stderr, "main: nblocks=%u\n", nblocks);

  // width of 1 sample
  size_t samp_stride = nchan * MOPSR_NDIM;
  size_t chan_offset_bytes = channel * MOPSR_NDIM;

  // seek to the first sample of the channel of choice
  lseek (fd1, chan_offset_bytes, SEEK_CUR);
  lseek (fd2, chan_offset_bytes, SEEK_CUR);
  fprintf (stderr, "seeking for channel=%d offset bytes %ld\n", channel, chan_offset_bytes);

  /*
  delay = 130 * (1000000.0 / tsamp);
  offset = delay * samp_stride;
  lseek (fd2, offset, SEEK_CUR);
  lseek (fd1, offset, SEEK_CUR);
  */
  /*
  offset = delay * samp_stride;
  if (delay > 0)
  {
    lseek (fd2, offset, SEEK_CUR);
    fprintf (stderr, "FD2 seeking delay bytes %ld\n", offset);
  }
  else
  {
    lseek (fd1, offset, SEEK_CUR);
  }
  */

  size_t nbytes;
  nbytes = sizeof(fftwf_complex) * npt;
  fprintf (stderr, "FWD FFT npt=%d nbytes=%ld\n", npt, nbytes);
  fftwf_complex * fwd1_in  = (fftwf_complex*) fftwf_malloc (nbytes);
  fftwf_complex * fwd1_out = (fftwf_complex*) fftwf_malloc (nbytes);
  fftwf_complex * fwd2_in  = (fftwf_complex*) fftwf_malloc (nbytes);
  fftwf_complex * fwd2_out = (fftwf_complex*) fftwf_malloc (nbytes);

  nbytes = sizeof(fftwf_complex) * npt;
  fftwf_complex * result_sum = (fftwf_complex *) fftwf_malloc (nbytes);
  memset (result_sum, 0, nbytes);

  nbytes = sizeof(float) * npt;
  float * result   = (float *) malloc (nbytes);
  float * phases   = (float *) malloc (nbytes);
  float * file1_sp = (float *) malloc (nbytes);
  float * file2_sp = (float *) malloc (nbytes);

  fftwf_complex cross_power;

  memset (file1_sp, 0, nbytes);
  memset (file1_sp, 0, nbytes);

  // FWD filterbank plan
  fftwf_plan plan_fb  = fftwf_plan_dft_1d (npt, fwd1_in, fwd1_out, FFTW_FORWARD, FFTW_ESTIMATE);

  unsigned ipt, iset;
  float re, im;

  // number of blocks to read
  const unsigned nreads = (file1_size - 4096) / block_size;
  fprintf (stderr, "main: nreads= %d\n", nreads);

  unsigned half_ntap = ntap / 2;

  nbytes = sizeof(complex float) * (npt + 2 * half_ntap);
  complex float * raw1 = (complex float *) malloc (nbytes);
  complex float * raw2 = (complex float *) malloc (nbytes);
  memset (raw1, 0, nbytes);
  memset (raw2, 0, nbytes);

  complex float * tapped1 = 0;
  complex float * tapped2 = 0;
  float * filter = 0;
 
  tapped1 = (complex float *) malloc (nbytes);
  tapped2 = (complex float *) malloc (nbytes);
  filter = (float *) malloc (ntap * sizeof(float));

  unsigned nsamps_per_block = block_size / (nchan * MOPSR_NDIM);
  float seconds_per_pt = (float) nsamps_per_block * (tsamp / 1000000);

  float * cross_amps_bin = (float *) malloc(sizeof(float) * nblocks);
  float * cross_phases_bin = (float *) malloc(sizeof(float) * nblocks);
  float cross_amps_sum_squares = 0;
  float theta;
  complex float ramp;

  size_t sample_bytes = MOPSR_NDIM;
  int8_t sample[1];

  fftwf_complex raw_cross;
  fftwf_complex p1, p2, cross_power_fscr;
  unsigned pt, i;

  double bytes_per_second = (double) (1000000 / tsamp) * MOPSR_NDIM;
  fprintf (stderr, "bytes_per_second=%lf\n", bytes_per_second);

  double total_bytes_read = 0;

  int nbays;
  const char * bays_file = "/home/dada/linux_64/share/molonglo_bays.txt";
  mopsr_bay_t * all_bays = read_bays_file (bays_file, &nbays);

  // read the modules file
  int nmodules;
  const char * modules_file = "/home/dada/linux_64/share/molonglo_modules.txt";
  mopsr_module_t * all_modules = read_modules_file (modules_file, &nmodules);

  // read the signal paths file
  int npfbs;
  const char * signal_paths_file = "/home/dada/linux_64/share/mopsr_signal_paths.txt";
  mopsr_pfb_t * pfbs = read_signal_paths_file (signal_paths_file, &npfbs);

  const unsigned nant= 2;
  unsigned iant;

  size_t delays_size = sizeof(mopsr_delay_t *) * nant;
  mopsr_delay_t ** delays = (mopsr_delay_t **) malloc (delays_size);
  for (iant=0; iant<nant; iant++)
    delays[iant] = (mopsr_delay_t *) malloc (sizeof(mopsr_delay_t) * nchan);

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

  fprintf (stderr, "file1: pfb_id_str=%s -> pfb_id=%d\n", pfb_id_str1, pfb_id1);
  fprintf (stderr, "file2: pfb_id_str=%s -> pfb_id=%d\n", pfb_id_str2, pfb_id2);

  unsigned ipfb_mod, imod;
  //for (ipfb_mod=0; ipfb_mod < MOPSR_MAX_MODULES_PER_PFB; ipfb_mod++)
  {
    //if (strcmp(pfbs[pfb_id1].modules[ipfb_mod], "-") != 0)
    {
      for (imod=0; imod<nmodules; imod++)
        if (strcmp(pfbs[pfb_id1].modules[ant_id1],all_modules[imod].name) == 0)
          modules[0] = all_modules[imod];
    }
    //if (strcmp(pfbs[pfb_id2].modules[ipfb_mod], "-") != 0)
    {
      for (imod=0; imod<nmodules; imod++)
        if (strcmp(pfbs[pfb_id2].modules[ant_id2],all_modules[imod].name) == 0)
          modules[1] = all_modules[imod];
    }
  }

  double baseline = fabs( (modules[1].dist * modules[1].sign) - (modules[0].dist * modules[0].sign));
  fprintf (stderr, "file1: module.name=%s dist=%lf\n", modules[0].name, modules[0].dist);
  fprintf (stderr, "file2: module.name=%s dist=%lf\n", modules[1].name, modules[1].dist);
  fprintf (stderr, "baseline = %lf\n", baseline);

  double mod0_delay_secs = modules[0].fixed_delay;
  double mod0_delay_fractional = mod0_delay_secs / (1.28 / 1000000);
  fprintf (stderr, "mod0: delay=%le secs = %7.5lf samples\n", mod0_delay_secs, mod0_delay_fractional);

  double mod1_delay_secs = modules[1].fixed_delay;
  double mod1_delay_fractional = mod1_delay_secs / (1.28 / 1000000);
  fprintf (stderr, "mod1: delay=%le secs = %7.5lf samples\n", mod1_delay_secs, mod1_delay_fractional);
  //fractional_delay = (float) mod1_delay_fractional;

  const double channel_bw = 0.78125;
  const double base_freq = 800 - (channel_bw/2);

  unsigned ichan = 0;
  mopsr_chan_t * channels = (mopsr_chan_t *) malloc(sizeof(mopsr_chan_t) * nchan);
  for (ichan=0; ichan < nchan; ichan++)
  {
    channels[ichan].number = chan_offset + ichan;
    channels[ichan].bw     = channel_bw;
    channels[ichan].cfreq  = base_freq + (channel_bw/2) + (channels[ichan].number * channel_bw);
  }

  fprintf (stderr, "channels[0].number = %d\n", channels[0].number);
  fprintf (stderr, "channels[0].bw = %lf\n", channels[0].bw);
  fprintf (stderr, "channels[0].cfreq = %lf\n", channels[0].cfreq);
  fprintf (stderr, "file channel=%d pfb_channel=%d, bw=%lf, cfreq=%lf\n", channel, channels[channel].number, channels[channel].bw, channels[channel].cfreq);

  // get initial delays
  char title[128];

  char apply_instrumental = 0;
  char apply_geometric = 1;
  char is_tracking = 1;

  float fringe1, fringe2, diff_fringe;
  complex float phasor1;
  complex float phasor2;
  complex float diff_phasor;
  float phase1, phase2, diff_phase;
  unsigned start_ipt = 0;

  double obs_offset_seconds;
  struct timeval timestamp;
  cross_power_fscr = 0;

  // get initial integer delays
  total_bytes_read = obs_offset / nchan;

  unsigned sample_offset1 = 0;
  unsigned sample_offset2 = 0;
  int diff;

  for (iblock=0; iblock<nblocks; iblock++)
  {
    // read in a block of data
    obs_offset_seconds = (total_bytes_read + block_size/2) / bytes_per_second;
    timestamp.tv_sec = floor(obs_offset_seconds);
    timestamp.tv_usec = (obs_offset_seconds - (double) timestamp.tv_sec) * 1000000;
    timestamp.tv_sec += utc_start;
    fprintf (stderr, "obs_offset_seconds=%lf, timestamp=%ld.%ld\n", 
                      obs_offset_seconds, timestamp.tv_sec, timestamp.tv_usec);

    if (calculate_delays (nbays, all_bays, nant, modules, nchan, channels,
                          source, timestamp, delays, apply_instrumental,
                          apply_geometric, is_tracking, tsamp) < 0)
    {
      fprintf (stderr, "failed to calculate delays!\n");
      return -1;
    }

    if (delays[0][channel].samples != sample_offset1)
    {
      diff = delays[0][channel].samples - sample_offset1;
      offset = (diff) * samp_stride;
      fprintf (stderr, "FD1 seeking sample delay by %d samples, bytes %ld\n", diff, offset);
      lseek (fd1, offset, SEEK_CUR);
      sample_offset1 = delays[0][channel].samples;
    }

    if (delays[1][channel].samples != sample_offset2)
    {
      diff = delays[1][channel].samples - sample_offset2;
      offset = diff * samp_stride;
      fprintf (stderr, "FD2 seeking sample delay by %d samples, bytes %ld\n", diff, offset);
      lseek (fd2, offset, SEEK_CUR);
      sample_offset2 = delays[1][channel].samples;
    }

    fringe1 = (float) delays[0][channel].fringe_coeff;
    fringe2 = (float) delays[1][channel].fringe_coeff;

    phasor1 = cos(fringe1) + sin(fringe1) * I;
    phasor2 = cos(fringe2) + sin(fringe2) * I;

    phase1 = atan2f(cimag(phasor1), creal(phasor1));
    phase2 = atan2f(cimag(phasor2), creal(phasor2));

    diff_fringe = fringe2 - fringe1;
    diff_phasor = cos(diff_fringe) + sin(diff_fringe) * I;
    diff_phase  = atan2f(cimag(diff_phasor), creal(diff_phasor));

    fprintf (stderr, "1: tot_secs=%le tot_samps=%7.4f samples=%u "
                     "fractional=%7.4f fringe_coeff=%7.4lf\n", 
                     delays[0][channel].tot_secs,
                     delays[0][channel].tot_samps,
                     delays[0][channel].samples,
                     delays[0][channel].fractional,
                     delays[0][channel].fringe_coeff);

    fprintf (stderr, "2: tot_secs=%le tot_samps=%7.4f samples=%u "
                     "fractional=%7.4f fringe_coeff=%7.4lf\n", 
                     delays[1][channel].tot_secs,
                     delays[1][channel].tot_samps,
                     delays[1][channel].samples,
                     delays[1][channel].fractional,
                     delays[1][channel].fringe_coeff);

    if (verbose)
    {
      fprintf (stderr, "toff=%lf fringe1=%f (%f + %f) phase=%f\n", 
                        obs_offset_seconds, fringe1, 
                        creal(phasor1), cimag(phasor1), phase1);
      fprintf (stderr, "toff=%lf fringe2=%f (%f + %f) phase=%f\n", 
                        obs_offset_seconds, fringe2, 
                        creal(phasor2), cimag(phasor2), phase2);
    }
    fprintf (stderr, "toff=%lf fringe=%f (%f + %f) phase=%f\n",
                      obs_offset_seconds, diff_fringe, 
                      creal(diff_phasor), cimag(diff_phasor), 
                      diff_phase * (180 / M_PI));
    cross_power_fscr = 0;
    raw_cross = 0;
    
    float f1 = (float) delays[0][channel].fractional;
    float f2 = (float) delays[1][channel].fractional;

    // each block is divided up into nparts
    unsigned ipart;
    const unsigned npart = block_size / (npt * MOPSR_NDIM);
    for (ipart=0; ipart < npart; ipart++)
    {
      // first load up the raw1 and raw2 arrays with the original data
      for (ipt=start_ipt; ipt<npt + (2*half_ntap); ipt++)
      {
        bytes_read = read (fd1, (void *) &sample, sample_bytes);
        lseek (fd1, samp_stride - sample_bytes, SEEK_CUR); 
        re = ((float) sample[0]) + 0.5;
        im = ((float) sample[1]) + 0.5;
        re /= 128;
        im /= 128;
        raw1[ipt] = (re + im * I);
      }

      for (ipt=start_ipt; ipt<npt + (2*half_ntap); ipt++)
      {
        bytes_read = read (fd2, (void *) &sample, sample_bytes);
        lseek (fd2, samp_stride - sample_bytes, SEEK_CUR);
        re = ((float) sample[0]) + 0.5;
        im = ((float) sample[1]) + 0.5;
        re /= 128;
        im /= 128;
        raw2[ipt] = (re + im * I);
      }

      f2 = fractional_delay;

      sinc_filter_float (raw1, tapped1, filter, npt + 2*half_ntap, ntap, f1);
      sinc_filter_float (raw2, tapped2, filter, npt + 2*half_ntap, ntap, f2);

      for (ipt=0; ipt<npt; ipt++)
      {
        raw_cross += (fwd1_in[ipt] * conj(fwd2_in[ipt]));
        fwd1_in[ipt] = tapped1[ipt + half_ntap] * phasor1;
        fwd2_in[ipt] = tapped2[ipt + half_ntap] * phasor2;
      }

      // FFT both sets of samples
      fftwf_execute_dft (plan_fb, fwd1_in, fwd1_out);
      fftwf_execute_dft (plan_fb, fwd2_in, fwd2_out);

      // now compute cross correlation
      for (ipt=0; ipt<npt; ipt++)
      {
        // since PFB, this is dual side band and we must FFT-shift
        pt = (ipt + (npt/2)) % npt;
        if (pt == 0)
        {
          p1 = 0;
          p2 = 0;
        }
        else
        {
          p1 = fwd1_out[pt] / npt;
          p2 = fwd2_out[pt] / npt;
        }

        if (iblock > 0)
        {
          // this is the bandpass for each input, never reset
          file1_sp[ipt] += ((creal(p1) * creal(p1)) + (cimag(p1) * cimag(p1)));
          file2_sp[ipt] += ((creal(p2) * creal(p2)) + (cimag(p2) * cimag(p2)));

          cross_power = p1 * conj(p2); 
          cross_power_fscr += cross_power;
          result_sum[ipt] += cross_power;
        }
      }

      // this is the rewind pointer for NTAP
      for (ipt=0; ipt<(2*half_ntap); ipt++)
      {
        raw1[ipt] = raw1[ipt + npt];
        raw2[ipt] = raw2[ipt + npt];
      }
      start_ipt = 2 * half_ntap;

      total_bytes_read += (double) (npt * MOPSR_NDIM);  
    }

    if (iblock == 0)
    {

    }
    else
    {
      cross_phases_bin[iblock-1] = atan2f(cimag(cross_power_fscr), creal(cross_power_fscr)) * (180/M_PI);
      cross_amps_bin[iblock-1]   = sqrtf(creal(cross_power_fscr) * creal(cross_power_fscr) + cimag(cross_power_fscr) * cimag(cross_power_fscr));

    //raw_phases_bin[iblock] = diff_phase * (180/M_PI);
    //raw_amps_bin[iblock]   = iblock;

      for (ipt=0; ipt<npt; ipt++)
      {
        re = creal(result_sum[ipt]);
        im = cimag(result_sum[ipt]);
        phases[ipt] = atan2f(im, re);
        result[ipt] = sqrt(re * re + im * im);
      }
  
      sprintf (title, "Cross-correlation in %5.2lf Mhz channel at %5.2lf MHz over %5.1lf m", channels[channel].bw, channels[channel].cfreq, baseline);
      plot_power_spectrum (file1_sp, npt, "1/xs", "Bandpass A", 0, -1, 0);
      plot_power_spectrum (file2_sp, npt, "2/xs", "Bandpass B", 0, -1, 0);
      plot_cross_vs_freq (result, phases, npt, "3/xs");
      plot_cross_vs_time (cross_amps_bin, cross_phases_bin, iblock-1, "4/xs", seconds_per_pt, title);
    }
  }

  /*
  for (ipt=0; ipt<npt; ipt++)
  {
    phases[ipt] = atan2f(cimag(result_sum[ipt]), creal(result_sum[ipt]));
    result[ipt] = sqrt(creal(result_sum[ipt]) * creal(result_sum[ipt]) + cimag(result_sum[ipt]) * cimag(result_sum[ipt]));
  }

  sprintf (title, "Cross-correlation in %5.2;f Mhz channel at %5.2;f MHz over %5.1;f m", channels[channel].bw, channels[channel].cfreq, baseline);
  plot_power_spectrum (file1_sp, npt, "1/xs", "Bandpass A", 0, -1, 0);
  plot_power_spectrum (file2_sp, npt, "2/xs", "Bandpass B", 0, -1, 0);
  plot_cross_vs_freq (result, phases, npt, "3/xs");
  plot_cross_vs_time (cross_amps_bin, cross_phases_bin, nsets_fac, "4/xs", seconds_per_pt, title);
  //plot_cross_bin (raw_amps_bin, raw_phases_bin, nsets_fac, "6/xs", seconds_per_pt, "");
  //plot_histogram (cross_phases_bin, nsets, "6/xs");

  FILE * fptr = fopen("phase_vs_time.bin", "w");
  if (!fptr)
  {
    fprintf(stderr, "Could not open file: phase_vs_time.bin\n");
    return -1;
  }
  fwrite( cross_phases_bin, sizeof(float), nsets_fac, fptr);
  fclose (fptr);

  fptr = fopen("amps_vs_time.bin", "w");
  if (!fptr)
  {
    fprintf(stderr, "Could not open file: amps_vs_time.bin\n");
    return -1;
  }
  fwrite( cross_amps_bin, sizeof(float), nsets_fac, fptr);
  fclose (fptr);

  float ymax = -FLT_MAX;
  int bin = 0;
  for (i=0; i<npt; i++)
  {
    if ((result[i] > ymax) && (i < npt/2))
    {
      ymax = result[i];
      bin = i;
    }
  }

  fprintf (stderr, "ymax=%f bin=%d\n", ymax, bin);
  */


  if (verbose)
    fprintf (stderr, "freeing allocated memory\n");

  if (filter)
    free (filter);
  free (raw1);
  free (raw2);
  if (tapped1)
    free (tapped1);
  if (tapped2)
    free (tapped2);

  free (file1_sp);
  free (file2_sp);
  free (phases);

  fftwf_free (result);
  fftwf_free (fwd1_in);
  fftwf_free (fwd2_in);
  fftwf_free (fwd1_out);
  fftwf_free (fwd2_out);
  fftwf_destroy_plan (plan_fb);

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
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

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


void plot_complex_spectrum (fftwf_complex * data, unsigned npts, char * device)
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


