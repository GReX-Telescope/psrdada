/*
 * read a file from disk and create the associated images
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include <readline/readline.h>
#include <readline/history.h>


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
void plot_complex_spectrum (complex float * data, unsigned npts, char * device);
void plot_cross_vs_time (float * amps, float * phases, unsigned npts, char * device, float seconds_per_pt, char * title);
void plot_cross_vs_freq (float * amps, float * phases, unsigned npts, char * device);
void plot_lags (float * yvals, unsigned npts, char * device);

void usage()
{
  fprintf (stdout,
     "mopsr_dadacorrplot_tiedbeam [options] file1 file2\n"
     " files must be single antenna, multi channel files\n"
     " -b nset         number of ffts in each block [32]\n"
     " -d delay        number of samples to delay files by [default 0]\n"
     " -e turns        fractional turns to delay files2 by [default 0]\n"
     " -f fractional   fractional samples to delay files2 by [default 0]\n"
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

  if (config_dir == 0)
    config_dir = "/home/dada/linux_64/share";

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

  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd1 = open (file1, flags, perms);
  if (fd1 < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", file1, strerror(errno));
    exit(EXIT_FAILURE);
  }

  int fd2 = open (file2, flags, perms);
  if (fd2 < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", file2, strerror(errno));
    exit(EXIT_FAILURE);
  }

  const size_t dada_header_size = 16384;
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
  int nchan_out = nchan * npt;

  float tsamp;
  if (ascii_header_get( header1, "TSAMP", "%f", &tsamp) != 1)
  {
    fprintf (stderr, "failed to read nchan from header1\n");
    close(fd1);
    close(fd2);
    return (EXIT_FAILURE);
  }

  char tmp[32];
  if (ascii_header_get (header1, "UTC_START", "%s", tmp) != 1)
  {
    fprintf (stderr, "Could not read UTC_START from header\n");
    return -1;
  }
  fprintf(stderr, "UTC_START=%s\n", tmp);
  time_t utc_start = str2utctime (tmp);

  float ut1_offset;
  if (ascii_header_get (header1, "UT1_OFFSET", "%f", &ut1_offset) != 1)
  {
    fprintf (stderr, "Could not read UT1_OFFSET from header\n");
    return -1;
  }
  fprintf (stderr, "UT1_OFFSET=%f\n", ut1_offset);

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

  float start_md_angle;
  if (ascii_header_get (header1, "MD_ANGLE", "%f", &start_md_angle) != 1)
  {
    fprintf(stderr, "could not read MD_ANGLE from header1\n");
    start_md_angle = 0;
  }

  float bw;
  if (ascii_header_get (header1, "BW", "%f", &bw) != 1)
  {
    fprintf(stderr, "could not read BW from header1\n");
    bw = 0;
  }

  float freq;
  if (ascii_header_get (header1, "FREQ", "%f", &freq) != 1)
  {
    fprintf(stderr, "could not read FREQ from header1\n");
    freq = 0;
  }

  uint64_t bytes_per_second;
  if (ascii_header_get (header1, "BYTES_PER_SECOND", "%"PRIu64, &bytes_per_second) != 1)
  {
    fprintf(stderr, "could not read BYTES_PER_SECOND from header1\n");
    bytes_per_second = (uint64_t) ((double) (nchan * MOPSR_NDIM * 1e6) / tsamp);
  }

  unsigned iblock, ichan, idat, ipt, ibatch;

  const unsigned half_ntap = ntap / 2;
  unsigned ndat = nset * npt;
  const int fixed_delay = half_ntap + 5;
  const int overlap = ntap + fixed_delay;
  const int ndat_overlap = ndat + overlap;
  fprintf (stderr, "ndat=%u ntap=%u fixed_delay=%d overlap=%d ndat_overlap=%d\n", ndat, ntap, fixed_delay, overlap, ndat_overlap);

  // for reading raw blocks of data from input files
  unsigned raw_size = nchan * ndat_overlap * MOPSR_NDIM * sizeof(float);
  float * raw1 = (float *) malloc (raw_size);
  float * raw2 = (float *) malloc (raw_size);

  unsigned block_size = nchan * ndat * MOPSR_NDIM * sizeof(float);
  const unsigned nblocks = (file1_size - dada_header_size) / block_size;
  if (verbose)
    fprintf (stderr, "main: file_size=%u raw_size=%u nblocks=%u\n", (file1_size - dada_header_size), block_size, nblocks);

  // width of 1 sample in bytes
  size_t samp_stride = nchan * MOPSR_NDIM;
  off_t offset1 = dada_header_size;
  off_t offset2 = dada_header_size;

  if (delay < 0)
    offset1 += (abs(delay) * samp_stride * sizeof(float));
  if (delay > 0)
    offset2 += (abs(delay) * samp_stride * sizeof(float));

  fprintf (stderr, "delay=%d offset1=%d offset2=%d\n", delay, offset1, offset2);

  lseek (fd1, offset1, SEEK_SET);
  lseek (fd2, offset2, SEEK_SET);

  // unpacked data size (npt * nsamp)
  complex float ** unpacked1 = (complex float **) malloc (sizeof(complex float *) * nchan);
  complex float ** unpacked2 = (complex float **) malloc (sizeof(complex float *) * nchan);
  size_t unpacked_size = ndat_overlap * sizeof(fftwf_complex);
  for (ichan=0; ichan<nchan; ichan++)
  {
    unpacked1[ichan] = (complex float *) fftwf_malloc (unpacked_size);
    unpacked2[ichan] = (complex float *) fftwf_malloc (unpacked_size);
  }

  // tapped data size (unpacked - ntap)
  complex float ** tapped1 = (complex float **) malloc (sizeof(complex float *) * nchan);
  complex float ** tapped2 = (complex float **) malloc (sizeof(complex float *) * nchan);
  size_t tapped_size = ndat_overlap * sizeof(complex float);
  for (ichan=0; ichan<nchan; ichan++)
  {
    tapped1[ichan] = (complex float *) fftwf_malloc (tapped_size);
    tapped2[ichan] = (complex float *) fftwf_malloc (tapped_size);
  }

  // FFT output
  size_t spectrum_size = npt * sizeof(complex float);
  complex float * spectrum1 = (complex float *) malloc (spectrum_size);
  complex float * spectrum2 = (complex float *) malloc (spectrum_size);

  // cross correlation 
  size_t xcorr_size = nchan * npt * sizeof(complex float);
  complex float * xcorr = (complex float *) malloc (xcorr_size);
  bzero (xcorr, xcorr_size);

  complex float * lag = (complex float *) malloc (xcorr_size);
  float * lag_power = (float *) malloc (nchan * npt * sizeof(float));

  // amps, phases and power spectra with full frequnecy resolution
  size_t nbytes = sizeof(float) * nchan_out;
  float * amps   = (float *) malloc (nbytes);
  float * phases = (float *) malloc (nbytes);
  float * power1 = (float *) malloc (nbytes);
  float * power2 = (float *) malloc (nbytes);

  fprintf (stderr, "zeroing power[1|2] with %ld bytes\n", nbytes);
  bzero (power1, nbytes);
  bzero (power2, nbytes);

  // amps and phases with time resolution
  nbytes = sizeof(float) * nblocks;
  float * amps_t = (float *) malloc(sizeof(float) * nblocks);
  float * phases_t = (float *) malloc(sizeof(float) * nblocks);

  // FWD filterbank plan
  fftwf_plan plan_fwd  = fftwf_plan_dft_1d (npt, unpacked1[0], spectrum1, FFTW_FORWARD, FFTW_ESTIMATE);
  fftwf_plan plan_bwd  = fftwf_plan_dft_1d (npt*nchan, xcorr, lag, FFTW_BACKWARD, FFTW_ESTIMATE);

  // memory for FIR 
  float * filter = (float *) malloc (ntap * sizeof(float));

  // number of seconds per time point 
  float seconds_per_pt = (float) (nset * npt * tsamp) / 1e6;
  fprintf (stderr, "bytes_per_second=%lu seconds_per_pt=%f \n", bytes_per_second, seconds_per_pt);

  uint64_t total_bytes_read = 0;

  char title[128];

  // get initial integer delays
  total_bytes_read = obs_offset;

  float max_power = 0;
  int max_idx = 0;

  struct timeval timestamp;
  double obs_offset_seconds = (double) obs_offset / bytes_per_second;
  time_t now = utc_start + obs_offset_seconds;
  struct tm * utc = gmtime (&now);

  float f1 = 0.0;
  float f2 = 0.0;
  if (fractional_delay > 0)
    f1 = fractional_delay;
  else
    f2 = fractional_delay;
  int d1 = 0.0;
  int d2 = 0.0;

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

    // read a block of data from each file, including ntap extra samples
    if (verbose > 1)
      fprintf (stderr, "[%d] reading %ld bytes from files\n", iblock, raw_size);

    bytes_read = read (fd1, (void *) raw1, raw_size);
    bytes_read = read (fd2, (void *) raw2, raw_size);

    float tstart = (float) iblock * seconds_per_pt;
    // unpack the samples from each input
    if (verbose > 1)
      fprintf (stderr, "[%d] unpacked ndat_overlap=%d\n", iblock, ndat_overlap);
    uint64_t i = 0;
    float re, im;
    for (idat=0; idat<ndat_overlap; idat++)
    {
      for (ichan=0; ichan<nchan; ichan++)
      {
        re = raw1[i+0];
        im = raw1[i+1];
        unpacked1[ichan][idat] = re + im * I;

        re = raw2[i+0];
        im = raw2[i+1];
        unpacked2[ichan][idat] = re + im * I;

        i += 2; 
      }
    }

    if (verbose > 1)
      fprintf (stderr, "[%d] unpacked %lu samples from raw f1=%f f2=%f2\n", iblock, i, f1, f2);

    // integer + fractional delay
    for (ichan=0; ichan<nchan; ichan++)
    {
      complex float * delayed1 = unpacked1[ichan] + d1;
      complex float * delayed2 = unpacked2[ichan] + d2;

      sinc_filter_float (delayed1, tapped1[ichan], filter, (int) ndat + 2*half_ntap, ntap, f1);
      sinc_filter_float (delayed2, tapped2[ichan], filter, (int) ndat + 2*half_ntap, ntap, f2);

      double channel_turns_f1 = ((double) ichan - ((double) nchan / 2) + 0.5) * (double) f1;
      double channel_phase_offset_f1 = channel_turns_f1 * 2 * M_PI;

      double channel_turns_f2 = ((double) ichan - ((double) nchan / 2) + 0.5) * (double) f2;
      double channel_phase_offset_f2 = channel_turns_f2 * 2 * M_PI;

      const float fringe1 = (float) channel_phase_offset_f1;
      const float fringe2 = (float) channel_phase_offset_f2;
                  
      const complex float phasor1 = cos(fringe1) + sin(fringe1) * I;
      const complex float phasor2 = cos(fringe2) + sin(fringe2) * I;

      //fprintf (stderr, "channel_turns=(%lf, %lf) fringes = %f, %f\n", channel_turns_f1, channel_turns_f2,fringe1, fringe2);
                              
      for (idat=0; idat<ndat; idat++)
      {
        if (f1 != 0.0)
          tapped1[ichan][idat] = tapped1[ichan][idat] * phasor1;
        if (f2 != 0.0)
          tapped2[ichan][idat] = tapped2[ichan][idat] * phasor2;
      }
    }

    // Forward FFT each channel
    const unsigned nbatch = ndat / npt;
    if (verbose > 1)
      fprintf (stderr, "[%d] performing %u pt FFT in %u batches\n", iblock, npt, nbatch);
    fftwf_complex * out1 = (fftwf_complex *) spectrum1;
    fftwf_complex * out2 = (fftwf_complex *) spectrum2;
    for (ichan=0; ichan<nchan; ichan++)
    {
      fftwf_complex * in1 = (fftwf_complex *) tapped1[ichan];
      fftwf_complex * in2 = (fftwf_complex *) tapped2[ichan];
      //fftwf_complex * in1 = (fftwf_complex *) unpacked1[ichan];
      //fftwf_complex * in2 = (fftwf_complex *) unpacked2[ichan];

      for (ibatch=0; ibatch<nbatch; ibatch++)
      {
        fftwf_execute_dft (plan_fwd, in1, out1);
        fftwf_execute_dft (plan_fwd, in2, out2);

        // compute cross correlation products
        for (ipt=0; ipt<npt; ipt++)
        {
          // fft shift
          const unsigned opt = (ipt + (npt/2)) % npt;

          // normalise 
          complex float p1 = out1[opt] / npt;
          complex float p2 = out2[opt] / npt;
/*
          const unsigned bad = 2;
          if (ipt < bad || ipt > npt-bad)
          {
            p1 = 0 + 0 * I;
            p2 = 0 + 0 * I;
          }
*/

          const complex float cross_power = p1 * conj(p2);

          unsigned odx = ichan*npt + ipt;

          // power spectrum for each input [accumulated over all samples]
          power1[odx] += cabsf(p1);
          power2[odx] += cabsf(p2);

          // complex power spectrum
          //if (channels[ichan].cfreq < 849.3 || channels[ichan].cfreq > 849.5)
          {
            xcorr[odx] += cross_power;
            xcorr_fscr += cross_power;
          }
        }

        in1 += npt;
        in2 += npt;
      }
    }


    if (verbose > 1)
      fprintf (stderr, "[%d] executing inverse fft", iblock);

    // inverse fft cross correlation
    fftwf_execute_dft (plan_bwd, xcorr, lag);
    
    // compute the xcorr phase and amplitude [vs time]
    phases_t[iblock] = cargf(xcorr_fscr) * (180/M_PI);
    amps_t[iblock]   = cabsf(xcorr_fscr);

    // compute the xcorr phase and amplitude [vs freq]
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (ipt=0; ipt<npt; ipt++)
      {
        unsigned idx = ichan*npt + ipt;
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
    }

    //lag_power[nchan_out/2] = 0;

    sprintf (title, "X-Corr %5.3lf MHz bw centred at %5.3lf MHz", bw, freq);
    plot_power_spectrum (power1, nchan_out, "1/xs", "Bandpass A", 0, -1, 0);
    plot_power_spectrum (power2, nchan_out, "2/xs", "Bandpass B", 0, -1, 0);
    plot_cross_vs_freq (amps, phases, nchan_out, "3/xs");
    plot_lags(lag_power, nchan_out, "4/xs");

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

    bzero(xcorr, nchan_out * sizeof(float) * 2);

    /*
    if (tstart > 0.35 && tstart < 0.37)
    {
      char * buf;
      buf = readline (">>");
    }
    */
  }

  if (verbose)
    fprintf (stderr, "freeing allocated memory\n");

  if (filter)
    free (filter);
  free (raw1);
  free (raw2);

  for (ichan=0; ichan<nchan; ichan++)
  {
    free (unpacked1[ichan]);
    free (unpacked2[ichan]);
    free (tapped1[ichan]);
    free (tapped2[ichan]);
  }
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

  float ph_min = -M_PI;
  float ph_max =  M_PI;

  cpgswin(xmin, xmax, 0, ymax);

  cpgsvp(0.05, 0.95, 0.5, 0.9);
  cpgbox("BCST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("", "", "Cross Power Spectrum");

  cpgsci(2);
  cpgline(npts, xvals, cross);
  cpgsci(1);

  cpgswin(xmin, xmax, ph_min, ph_max);
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

void plot_lags (float * yvals, unsigned npts, char * device)
{
  if (cpgbeg(0, device, 1, 1) != 1)
  { 
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  int offset = (npts/2);

  // TODO remove
  //offset = 100;
  
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
  
  //xmin = -100;
  //xmax = 100;

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
