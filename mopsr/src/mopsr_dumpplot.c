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

#include <cpgplot.h>

#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"
#include "mopsr_udp.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

#define  MOPSR_DUMPPLOT_TYPE_ALL 0
#define  MOPSR_DUMPPLOT_TYPE_WF 1
#define  MOPSR_DUMPPLOT_TYPE_TS 2
#define  MOPSR_DUMPPLOT_TYPE_BP 3
#define  MOPSR_DUMPPLOT_TYPE_HG 4

void usage ();

void usage()
{
  fprintf (stdout,
     "mopsr_dumpplot [options] dumpfile\n"
     " -a ant     use the specified ant for interactive plots [default 0]\n"
     " -c chan    use the specified channel for the hist and timeseries\n"
     " -d chan    use the specified channel for the hist and timeseries\n"
     " -D device  pgplot device name [default create PNG files]\n"
     " -g XxY     plot with resolution X pixels by Y pixels\n"
     " -l         plot logarithmically\n"
     " -p         plot plain (no axes, etc)\n"
     " -t type    plot only the specified plot type [wf, ts, bp, hg]\n"
     " -z         zap DC channel\n"
     " -v         be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = 0;

  int plot_type = MOPSR_DUMPPLOT_TYPE_ALL;

  // Mopsr UDP header
  mopsr_hdr_t hdr;

  // Plotting options
  mopsr_util_t opts;

  // plotting defaults
  opts.lock_flag  = -1;
  opts.lock_flag_long  = -1;
  opts.plot_log   = 0;
  opts.zap        = 0;
  opts.ant        = -1;
  opts.chans[0]   = -1;
  opts.chans[1]   = 127;
  opts.nbin       = 256;
  opts.ndim       = 2;
  opts.ant_code   = 0;
  opts.ant_id     = 0;
  opts.plot_plain = 0;
  int xres       = 800;
  int yres       = 600;

  int arg = 0;

  int ant = 0;

  while ((arg=getopt(argc,argv,"a:c:d:D:g:lpt:vz")) != -1)
  {
    switch (arg)
    {
      case 'a':
#ifdef HIRES
          ant = mopsr_get_hires_ant_index(atoi(optarg));
#else
          ant = mopsr_get_new_ant_index(atoi(optarg));
#endif
        break;

      case 'c':
        opts.chans[0] = atoi(optarg);
        break;

      case 'd':
        opts.chans[1] = atoi(optarg);
        break;

      case 'D':
        device = strdup(optarg);
        break;

      case 'g':
        if (sscanf (optarg, "%ux%u", &(xres), &(yres)) != 2)
        {
          fprintf (stderr, "ERROR: could not parse width and height "
                   "from %s\n", optarg);
          usage();
          return (EXIT_FAILURE);
        }
        break;

      case 'l':
        opts.plot_log = 1;
        break;

      case 'p':
        opts.plot_plain = 1;
        break;

      case 't':
        if (strcmp("wf", optarg) == 0)
          plot_type = MOPSR_DUMPPLOT_TYPE_WF;
        else if (strcmp("ts", optarg) == 0)
          plot_type = MOPSR_DUMPPLOT_TYPE_TS;
        else if (strcmp("bp", optarg) == 0)
          plot_type = MOPSR_DUMPPLOT_TYPE_BP;
        else if (strcmp("hg", optarg) == 0)
          plot_type = MOPSR_DUMPPLOT_TYPE_HG;
        else
        {
          fprintf (stderr, "ERROR: unrecognized plot type\n");
          usage();
          return (EXIT_FAILURE);
        } 
        break;
        
      case 'v':
        verbose++;
        break;

      case 'z':
        opts.zap = 1;
        break;

      default:
        usage ();
        return 0;
    } 
  }

  // check and parse the command line arguments
  if (argc-optind != 1)
  {
    fprintf(stderr, "ERROR: 1 command line arguments are required\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  char filename[256];
  char png_file[256];
  strcpy(filename, argv[optind]);

  struct stat buf;
  if (stat (filename, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat dump file [%s]: %s\n", filename, strerror(errno));
    exit(EXIT_FAILURE);
  }

  size_t filesize = buf.st_size;
  if (verbose)
    fprintf (stderr, "filesize for %s is %d bytes\n", filename, filesize);

  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd = open (filename, flags, perms);
  if (fd < 0)
  {
    fprintf(stderr, "failed to open dump file[%s]: %s\n", filename, strerror(errno));
    exit(EXIT_FAILURE);
  }

  const size_t nbytes = filesize - UDP_HEADER;

  if (verbose)
    fprintf (stderr, "reading %lu bytes\n",filesize);
  char * raw = (char *) malloc (filesize);
  char * packet = raw + UDP_HEADER;
  size_t bytes_read = read (fd, raw, filesize);

  if (verbose)
    fprintf (stderr, "read %lu bytes\n", bytes_read);

  // close the FD now that we are done with it.
  close (fd);

  // decode the header to find out NANT, NCHAN, etc
  mopsr_decode ((unsigned char *) raw, &hdr);

  opts.nant  = hdr.nant;
  opts.nchan = hdr.nchan;

  unsigned nsamp = (filesize - UDP_HEADER) / (hdr.nchan * hdr.nant * MOPSR_NDIM);
#ifdef _DEBUG
  fprintf (stderr, "filesize=%d UDP_HEADER=%d, nsamp=%d\n", filesize, UDP_HEADER, nsamp);
#endif

  // special case for "raw" data mode
  if ((hdr.nchan == 1) && (hdr.nsamp > 0))
  {
    if (hdr.nsamp != nsamp)
    {
      fprintf (stderr, "hdr.nsamp=%u != nsamp=%u\n", hdr.nsamp, nsamp);
    }
  }

  // allocate memory for timeseries and histogram
  float * timeseries = (float *) malloc (sizeof(float) * nsamp * MOPSR_NDIM);
  unsigned int * histogram = (unsigned int *) malloc (sizeof(unsigned int) * opts.ndim * opts.nbin);

  if (verbose)
    fprintf (stderr, "NANT=%u, NCHAN=%u, NDIM=%u, NSAMP=%u\n", opts.nant, opts.nchan, opts.ndim, nsamp);

  if (verbose)
    fprintf (stderr, "hdr.ant_id=%u hdr.ant_id2=%u\n", hdr.ant_id, hdr.ant_id2);

  unsigned int isamp, ichan, iant, ires;

  char local_time[32];
  unsigned int start_ant, end_ant, nres;
  char lock_char;

  // get the ANT ID and MGT lock idenifiers
  unsigned int ant_ids[opts.nant];
  unsigned char lock_flags[opts.nant];
  unsigned char lock_flags_long[opts.nant];
  for (iant=0; iant<opts.nant; iant++)
  {
    if (opts.nant == 8)
    {
      ant_ids[iant]    = mopsr_get_hires_ant_number (iant);
      lock_flags[iant] = mopsr_get_bit_from_16 (hdr.mgt_locks, ant_ids[iant]);
      lock_flags_long[iant] = mopsr_get_bit_from_16 (hdr.mgt_locks_long, ant_ids[iant]);
    }
    else if (opts.nant == 16)
    {
      ant_ids[iant]    = mopsr_get_new_ant_number (iant);
      lock_flags[iant] = mopsr_get_bit_from_16 (hdr.mgt_locks, ant_ids[iant]);
      lock_flags_long[iant] = mopsr_get_bit_from_16 (hdr.mgt_locks_long, ant_ids[iant]);
    }
    else
    {
      // TODO - revert! special case for 4 Input mode
      if ((iant == 0) || (iant == 1))
      {
        ant_ids[iant]    = mopsr_get_ant_number (hdr.ant_id, iant);
        lock_flags[iant] = mopsr_get_bit_from_16 (hdr.mgt_locks, ant_ids[iant]);
        lock_flags_long[iant] = -1;
      }
      else
      {
        ant_ids[iant]    = mopsr_get_ant_number (hdr.ant_id2, iant-2);
        lock_flags[iant] = mopsr_get_bit_from_16 (hdr.mgt_locks, ant_ids[iant]);
        lock_flags_long[iant] = -1;
      }
    }
  }



  if (!device)
  {
    start_ant = 0;
    end_ant = opts.nant;
  }
  else
  {
    start_ant = ant;
    end_ant = ant+1;
  }
  int lower_bin, upper_bin;
  unsigned max_ibin, max_counts;
  unsigned * dim_ptr;

  FILE * fptr = 0;
  if (!device)
  {
    strcpy(png_file, filename);
    char * str_ptr = strstr (png_file, ".dump");
    sprintf (str_ptr, ".stats");
    fptr = fopen (png_file, "w");
    fprintf (fptr, "#ANT FWHM\n");
  }

  // do the timeseries and histogram first
  for (iant=start_ant; iant<end_ant; iant++)
  {
    opts.ant = iant;
    opts.ant_id = ant_ids[iant];
    opts.lock_flag = lock_flags[iant];
    opts.lock_flag_long = lock_flags_long[iant];
    lock_char = opts.lock_flag ? 'L' : 'U';

    unsigned chan;
    if (opts.nchan == 1)
    {
      chan = 0;
      opts.chans[0] = 0;
      opts.chans[1] = 0;
    }
    else
    {
      if (opts.chans[0] == -1)
        chan = 50;
      else
        chan = (opts.chans[0] + opts.chans[1]) / 2;
    }

    // timeseries
    if (verbose)
      fprintf (stderr, "mopsr_extract_channel(%p, %p, %"PRIu64", %u, %u, %u, %u\n",
               (void *) timeseries, (void *) packet, nbytes, chan, iant, opts.nchan, opts.nant);
    mopsr_extract_channel (timeseries, packet, nbytes, chan, iant, opts.nchan, opts.nant);

    if (plot_type == MOPSR_DUMPPLOT_TYPE_ALL || plot_type == MOPSR_DUMPPLOT_TYPE_TS)
    {
      if (!device)
      {
        strcpy(png_file, filename);
        char * str_ptr = strstr (png_file, ".dump");
        sprintf (str_ptr, ".%d.ts.%c.%dx%d.png/png", opts.ant_id, lock_char, xres, yres);
      }
      else
        strcpy (png_file, device);

      if (verbose)
        fprintf (stderr, "creating %s\n", png_file);
      if (cpgopen (png_file) != 1)
        fprintf (stderr, "mopsr_dumpplot: error opening plot device [%s]\n", png_file);

      set_resolution (xres, yres);
      opts.ymin = -128;
      opts.ymax = 127;
      mopsr_plot_time_series (timeseries, chan, nsamp, &opts);
      opts.ymin = 0;
      opts.ymax = 0;
      cpgclos();
      if (device)
        sleep(1);
    }

    // count histogram statistics
    if (verbose)
      fprintf (stderr, "mopsr_form_histogram(%p, %p, %"PRIu64")\n",
               (void *) histogram, (void *) packet, nbytes);
    mopsr_form_histogram (histogram, packet, nbytes, &opts);

    // compute the FWHM for this histogram
    dim_ptr = histogram;
    unsigned fwhm_sum = 0;
    unsigned idim, ibin, ubin;

    for (idim=0; idim<opts.ndim; idim++)
    {
      upper_bin = -1;
      lower_bin = -1;
      max_ibin = 0;
      max_counts = 0;

      for (ibin=0; ibin<opts.nbin; ibin++)
      {
        if (dim_ptr[ibin] > max_counts)
        {
          max_counts = dim_ptr[ibin];
          max_ibin = ibin;
        }
      }
      
      if (verbose)
        fprintf (stderr, "max_counts=%u max_ibin=%u\n", max_counts, max_ibin);

      for (ibin=0; ibin<opts.nbin; ibin++)
      {
        if (lower_bin<0 && dim_ptr[ibin] > max_counts/2)
          lower_bin = ibin;

        ubin = (opts.nbin-1)-ibin;
        if (upper_bin<0 && dim_ptr[ubin] > max_counts/2)
          upper_bin = ubin;
      }

      fwhm_sum += (upper_bin - lower_bin);

      dim_ptr += opts.nbin;
    }

    fwhm_sum /= opts.ndim;
    if (verbose)
      fprintf (stderr, "fwhm=%u\n", fwhm_sum);

    if (plot_type == MOPSR_DUMPPLOT_TYPE_ALL || plot_type == MOPSR_DUMPPLOT_TYPE_HG)
    {
      if (!device)
      {
        strcpy(png_file, filename);
        char * str_ptr = strstr (png_file, ".dump");
        sprintf (str_ptr, ".%d.hg.%c.%dx%d.png/png", opts.ant_id, lock_char, xres, yres);
      }
      else
        strcpy (png_file, device);

      if (verbose)
        fprintf (stderr, "creating %s\n", png_file);
      if (cpgbeg(0, png_file, 1, 1) != 1)
        fprintf (stderr, "mopsr_dumpplot: error opening plot device [%s]\n", png_file);
      else
      {
        set_resolution (xres, yres);
        mopsr_plot_histogram (histogram, &opts);
        cpgend();
        if (device)
          sleep(1);
      }

      if (!device)
      {
        fprintf (fptr, "%u %u\n",opts.ant_id, fwhm_sum);
      }
    }
  }

  // now we need to FFT the data if only 1 channel
  if ((hdr.nchan == 1) && (hdr.nsamp > 0))
  {
    opts.nchan = 256;
    if (verbose)
      fprintf (stderr, "mopsr_channelise_frame(%p,%"PRIu64", %u, %u)\n",
                        (void *) packet, nbytes, opts.nant, opts.nchan);
    if (mopsr_channelise_frame (packet, nbytes, opts.nant, opts.nchan) < 0)
    {
      fprintf (stderr, "failed to channelise input\n");
      return EXIT_FAILURE;
    }
    nsamp /= opts.nchan;
  }

  // perform SQLD on buffer / frame
  float * bandpass = (float *) malloc (sizeof(float) * opts.nchan);
  float * spectra = (float *) malloc (sizeof(float) * opts.nchan);
  float * waterfall = (float *) malloc (sizeof(float) * nsamp * opts.nchan);
  float * waterfall_h = (float *) malloc (sizeof(float) * nsamp * opts.nchan);

  for (iant=start_ant; iant<end_ant; iant++)
  {
    opts.ant_id = ant_ids[iant];
    opts.lock_flag = lock_flags[iant];
    opts.lock_flag_long = lock_flags_long[iant];
    opts.ant = iant;
    lock_char = opts.lock_flag ? 'L' : 'U';

    if (verbose)
      fprintf (stderr, "mopsr_dumpplot: iant=%d, hdr.ant_id=%d, hdr.ant_id2=%d, "
                       "opts.ant_id=%d\n", iant, hdr.ant_id, hdr.ant_id2, opts.ant_id);

    mopsr_zero_float (bandpass, opts.nchan);

    // these operations work for all antenna [perhaps should change this?]
    for (isamp = 0; isamp <nsamp; isamp++)
    {
      // SQLD this antenna
      mopsr_sqld_pfbframe (spectra, packet + (isamp * opts.nchan * opts.nant * opts.ndim), &opts, 0);

      // copy spectra into waterfall array
      memcpy (waterfall + (isamp * opts.nchan), spectra, opts.nchan * sizeof(float));

      // sum spectra into bandpass plot
      for (ichan=0; ichan < opts.nchan; ichan++)
        bandpass[ichan] += spectra[ichan];
    }

    // plot the bandpass for an antenna
    if (plot_type == MOPSR_DUMPPLOT_TYPE_ALL || plot_type == MOPSR_DUMPPLOT_TYPE_BP)
    {
      if (!device)
      {
        strcpy(png_file, filename);
        char * str_ptr = strstr (png_file, ".dump");
        sprintf (str_ptr, ".%d.bp.%c.%dx%d.png/png", opts.ant_id, lock_char, xres, yres);
      }
      else
        strcpy (png_file, device);
    
      if (verbose)
        fprintf (stderr, "creating %s\n", png_file);
      if (cpgbeg(0, png_file, 1, 1) != 1)
        fprintf (stderr, "mopsr_dumpplot: error opening plot device [%s]\n", png_file);
      else
      {
        set_resolution (xres, yres);
        mopsr_plot_bandpass(bandpass, &opts);
        cpgend();
        if (device)
          sleep(1);
      }
    }

    // waterfall plot
    mopsr_transpose (waterfall_h, waterfall, nsamp, &opts);

    if (plot_type == MOPSR_DUMPPLOT_TYPE_ALL || plot_type == MOPSR_DUMPPLOT_TYPE_WF)
    {
      if (!device)
      {
        strcpy(png_file, filename);
        char * str_ptr = strstr (png_file, ".dump");
        sprintf (str_ptr, ".%d.wf.%c.%dx%d.png/png", opts.ant_id, lock_char, xres, yres);
      }
      else
        strcpy (png_file, device);

      if (verbose)
        fprintf (stderr, "creating %s\n", png_file);
      if (cpgbeg(0, png_file, 1, 1) != 1)
        fprintf (stderr, "mopsr_dumpplot: error opening plot device [%s]\n", png_file);
      else
      {
        set_resolution (xres, yres);
        mopsr_plot_waterfall (waterfall_h, nsamp, &opts);
        cpgend();
        if (device)
          sleep(1);
      }
    }
  }

  if (!device && fptr)
    fclose(fptr);

  free (raw);
  free (bandpass);
  free (spectra);
  free (waterfall);
  free (waterfall_h);
  free (timeseries);
  free (histogram);

  return EXIT_SUCCESS;
}
