/*
 * Open data block as a viewer, read some data and then optionally:
 *   write filterbank data to file
 *   create png files
 *   plot multiple pgplot issues
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <cpgplot.h>

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"

#include "node_array.h"
#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

typedef struct {

  multilog_t * log;

  unsigned int nant;

  unsigned int nchan;

  unsigned int nsamp;

  char * buffer;

  size_t buffer_size;

  char verbose;

} mopsr_dbmonitor_t;


int quit = 0;
void usage ();
int ipcio_view_eod (ipcio_t* ipcio, unsigned byte_resolution);

void usage()
{
  fprintf (stdout,
     "mopsr_dbmonitor [options]\n"
     " -c chan    use the specified channel for the hist and timeseries\n"
     " -k key     connect to key data block\n"
     " -l         plot logarithmically\n"
     " -s secs    sleep time between updates\n"
     " -t num     integrate num frames into monitoring output\n"
     " -D device  pgplot device name [default create PNG files]\n"
     " -z         zap DC channel\n"
     " -v         be verbose\n");
}

int main (int argc, char **argv)
{
  /* DADA Header plus Data Unit */
  dada_hdu_t * hdu = 0;

  mopsr_dbmonitor_t dbmonitor;

  mopsr_dbmonitor_t * ctx;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  unsigned int verbose = 0;

  /* Quit flag */
  char quit = 0;

  /* PGPLOT device name */
  char * device = 0;

  char zap = 0;

  /* dada key for SHM */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;

  ctx = &dbmonitor;

  ctx->nsamp = 1024;

  unsigned int plot_log = 0;

  unsigned int sleep_secs = 2;

  int chan = -1;

  while ((arg=getopt(argc,argv,"c:D:k:ls:t:vz")) != -1)
  {
    switch (arg)
    {
      
      case 'c':
        chan = atoi(optarg);
        break;

      case 'D':
        device = strdup(optarg);
        break;

      case 'l':
        plot_log = 1;
        break;

      case 't':
        ctx->nsamp = atoi(optarg);
        break;

      case 's':
        sleep_secs = atoi(optarg);
        break;

      case 'v':
        ctx->verbose++;
        break;

      case 'z':
        zap = 1;
        break;

      case 'k':
        if (sscanf (optarg, "%x", &dada_key) != 1) 
        {
          fprintf (stderr, "dada_db: could not parse key from %s\n", optarg);
          return -1;
        }
        break;
      
      default:
        usage ();
        return 0;
    } 
  }

  // setup multilogger
  ctx->log = multilog_open ("mopsr_dbmonitor", 0);
  multilog_add (ctx->log, stderr);

  // init HDU 
  hdu = dada_hdu_create (ctx->log);
  dada_hdu_set_key(hdu, dada_key);

  // connect to HDU
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "main: dada_hdu_connect()\n");
  if (dada_hdu_connect (hdu) < 0)
  {
    multilog (ctx->log, LOG_ERR, "could not connect to HDU on key %x\n", dada_key);
    return EXIT_FAILURE;
  }

  while ( !quit )
  {
    // open HDU as a passive viewer
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "main: dada_hdu_open_view()\n");
    if (dada_hdu_open_view(hdu) < 0)
    {
      dada_hdu_disconnect (hdu);
      multilog (ctx->log, LOG_ERR, "could not open HDU for viewing\n");
      return EXIT_FAILURE;
    }

    // open the HDU (check if this necessary)
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "main: dada_hdu_open()\n");
    if (dada_hdu_open (hdu) < 0)
    {
      dada_hdu_disconnect (hdu);
      multilog (ctx->log, LOG_ERR, "could not open HDU\n");
      return EXIT_FAILURE;
    }

    // extract key paramters from the HDU's headera
    unsigned int byte_resolution;
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "main: ascii_header_get (RESOULTION)\n");
    if (ascii_header_get (hdu->header, "RESOLUTION", "%u", &byte_resolution) < 0)
    {
      multilog (ctx->log, LOG_WARNING, "HEADER with no RESOLUTION\n");
      byte_resolution = 0;
    }

    mopsr_util_t opts;
    opts.plot_log = plot_log;
    opts.zap = 0;
    opts.ant = -1;
    opts.chans[0] = chan;
    opts.chans[1] = chan;
    opts.ymin =  1000;
    opts.ymax = -1000;
    opts.nbin = 256;
    opts.ant_code = 0;
    opts.ant_id = 0;
    opts.plot_plain = 1;
    opts.zap = zap;

    if (ascii_header_get (hdu->header, "NCHAN", "%u", &(opts.nchan)) < 0)
    {
      multilog (ctx->log, LOG_ERR, "HEADER with no NCHAN\n");
      dada_hdu_disconnect (hdu);
      return EXIT_FAILURE;
    }

    if (ascii_header_get (hdu->header, "NANT", "%u", &(opts.nant)) < 0)
    {
      multilog (ctx->log, LOG_ERR, "HEADER with no NANT\n");
      dada_hdu_disconnect (hdu);
      return EXIT_FAILURE;
    }
  
    if (ascii_header_get (hdu->header, "NDIM", "%u", &(opts.ndim)) < 0)
    {
      multilog (ctx->log, LOG_ERR, "HEADER with no NDIM\n");
      dada_hdu_disconnect (hdu);
      return EXIT_FAILURE;
    }

    if (ascii_header_get (hdu->header, "ANT_ID", "%u", &(opts.ant_id)) < 0)
    {
      multilog (ctx->log, LOG_WARNING, "HEADER with no ANT_ID\n");
    }
    multilog (ctx->log, LOG_INFO, "ANT_ID=%u\n", opts.ant_id);

    if (!byte_resolution)
      byte_resolution = opts.nchan * opts.nant * opts.ndim;

    if (byte_resolution != opts.nchan * opts.nant * opts.ndim) 
    {
      multilog (ctx->log, LOG_WARNING, "RESOLUTION not correct\n");
      byte_resolution = opts.nchan * opts.nant * opts.ndim;
    }

    multilog (ctx->log, LOG_INFO, "NANT=%u, NCHAN=%u, NDIM=%u, NSAMP=%u\n", opts.nant, opts.nchan, opts.ndim, ctx->nsamp);

    uint64_t bytes_to_read = byte_resolution * ctx->nsamp;
    void * buffer = malloc (bytes_to_read);
    int64_t bytes_read;
    unsigned int ispectra = 0;
    unsigned int nspectra = ctx->nsamp;
    char local_time[32];
    char png_file[128];

    multilog (ctx->log, LOG_INFO, "bytes_to_read=%"PRIu64"\n", bytes_to_read);

    unsigned int ichan, iant;

    // perform SQLD on buffer / frame
    float * bandpass = (float *) malloc (sizeof(float) * opts.nchan * opts.nant);
    float * spectra = (float *) malloc (sizeof(float) * opts.nchan);
    float * waterfall = (float *) malloc (sizeof(float) * nspectra * opts.nchan);
    float * waterfall_h = (float *) malloc (sizeof(float) * nspectra * opts.nchan);
    float * timeseries = (float *) malloc (sizeof(float) * nspectra * opts.ndim);
    unsigned int * histogram = (unsigned int *) malloc (sizeof(unsigned int) * opts.ndim * opts.nbin);

    time_t now, now_plus;

    int reconnect = 0;

    while ( !reconnect)
    {
      // current localtime as DADA time string
      now = time(0);

      // move to the last block/byte based on resolution
      if (ctx->verbose)
        multilog (ctx->log, LOG_INFO, "main: ipcio_view_eod()\n");
      if (ipcio_view_eod (hdu->data_block, byte_resolution) < 0)
      {
        multilog (ctx->log, LOG_ERR, "main: ipcio_view_eod failed\n");
        dada_hdu_disconnect (hdu);
        return EXIT_FAILURE;
      }

      // read the required amount from the data block
      if (ctx->verbose)
        multilog (ctx->log, LOG_INFO, "main: ipcio_read(%"PRIu64")\n", bytes_to_read);
      bytes_read = ipcio_read (hdu->data_block, (char *) buffer, bytes_to_read);
      if (bytes_read < 0)
      {
        multilog (ctx->log, LOG_ERR, "main: ipcio_read failed\n");
        free (buffer);
        dada_hdu_disconnect (hdu);
        return EXIT_FAILURE;
      }

      if (bytes_read < bytes_to_read)
      {
        multilog (ctx->log, LOG_WARNING, "main: ipcio_read returned %"PRIi64" bytes, "
                  "requested %"PRIu64"\n", bytes_read, bytes_to_read);
        reconnect = 1;
        break;
      }

      if (ctx->verbose)
        multilog (ctx->log, LOG_INFO, "main: zero arrays()\n");
      mopsr_zero_float (bandpass, opts.nchan * opts.nant);

      // current localtime as DADA time string
      strftime (local_time, 32, DADA_TIMESTR, localtime(&now));

      int nplot = 4;

      for (iant=0; iant<opts.nant; iant++)
      {
        opts.ant_id = iant;
        opts.ant = iant;
          
        if (!device)
          sprintf (png_file, "%s.%d.ts.png/png", local_time, iant);
        else
          sprintf (png_file, "%d/xs", 4*iant + 1);

        if (ctx->verbose)
          multilog (ctx->log, LOG_INFO, "opening %s\n", png_file);
        if (cpgopen (png_file) != 1)
        {
          multilog(ctx->log, LOG_WARNING, "mopsr_dbmonitor: error opening plot device [%s]\n", png_file);
        }

        opts.ymin =  1000;
        opts.ymax = -1000;

        // choose a channel to plot for a timeseries 
        int chan = opts.chans[0];
        if (opts.chans[0] == -1)
          chan = 64;
        mopsr_extract_channel (timeseries, buffer, bytes_read,
                               chan, iant, opts.nchan, opts.nant);

        set_resolution (640, 180);
        // plot the timeseries
        mopsr_plot_time_series (timeseries, chan, nspectra, &opts);
        cpgclos();
      }

      opts.ymin =  1000;
      opts.ymax = -1000;

      for (iant=0; iant < opts.nant; iant++)
      {
        opts.ant = iant;

        // these operations work for all antenna [perhaps should change this?]
        for (ispectra = 0; ispectra < nspectra; ispectra++)
        {
          // SQLD this antenna
          mopsr_sqld_pfbframe (spectra, buffer + (ispectra * opts.nchan * opts.nant * opts.ndim), &opts, 0);

          // copy spectra into waterfall array
          memcpy (waterfall + (ispectra * opts.nchan), spectra, opts.nchan * sizeof(float));

          // sum spectra into bandpass plot
          for (ichan=0; ichan < opts.nchan; ichan++)
            bandpass[ichan] += spectra[ichan];
        }

        // plot the bandpass for an antenna
        if (!device)
          sprintf (png_file, "%s.%d.bp.png/png", local_time, iant);
        else
          sprintf (png_file, "%d/xs", 4*iant + 2);

        if (ctx->verbose)
          multilog (ctx->log, LOG_INFO, "opening %s\n", png_file);

        if (cpgopen (png_file) != 1) 
        {
          multilog(ctx->log, LOG_WARNING, "mopsr_dbmonitor: error opening plot device [%s]\n", png_file);
        }

        set_resolution (180, 480);
        mopsr_plot_bandpass_vertical (bandpass, &opts);
        cpgclos();

        if (!device)
          sprintf (png_file, "%s.%d.wf.png/png", local_time, iant);
        else
          sprintf (png_file, "%d/xs",4*iant+3);

        if (ctx->verbose)
          multilog (ctx->log, LOG_INFO, "opening %s\n", png_file);
        if (cpgopen (png_file) != 1)
        {
          multilog(ctx->log, LOG_WARNING, "mopsr_dbmonitor: error opening plot device [%s]\n", png_file);
        }             

        set_resolution (640, 480);
        mopsr_transpose (waterfall_h, waterfall, nspectra, &opts);
        mopsr_plot_waterfall (waterfall_h, nspectra, &opts);
        cpgclos();
      }

      for (iant=0; iant < opts.nant; iant++)
      {             
        // count histogram statistics
        if (!device)
          sprintf (png_file, "%s.%d.hg.png/png", local_time, iant);
        else
          sprintf (png_file, "%d/xs", 4*iant+4);

        if (ctx->verbose)
          multilog (ctx->log, LOG_INFO, "opening %s\n", png_file);
        if (cpgopen (png_file) != 1)
        {
          multilog(ctx->log, LOG_WARNING, "mopsr_dbmonitor: error opening plot device [%s]\n", png_file);
        }

        mopsr_form_histogram (histogram, buffer, bytes_read, &opts);
        set_resolution (180, 180);
        mopsr_plot_histogram (histogram, &opts);

        cpgclos();
      }

      // wait the appointed amount of time
      now_plus = time(0);
      while (now + sleep_secs >= now_plus)
      {
        usleep (100000);
        now_plus = time(0);
        // multilog (ctx->log, LOG_INFO, "main: [sleep] now_plus - now (%d)\n", now_plus - now);
      }
    }

    free (bandpass);
    free (spectra);
    free (waterfall);
    free (waterfall_h);
    free (timeseries);
    free (histogram);

    if (dada_hdu_close_view (hdu) < 0)
    {
      dada_hdu_disconnect (hdu);
      multilog (ctx->log, LOG_ERR, "could not close HDU\n");
      return EXIT_FAILURE;
    }
  }

  if (dada_hdu_disconnect (hdu) < 0)
  {
    fprintf (stderr, "ERROR: could not disconnect from HDU\n");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int ipcio_view_eod (ipcio_t* ipcio, unsigned byte_resolution)
{
  ipcbuf_t* buf = &(ipcio->buf);

#ifdef _DEBUG
  fprintf (stderr, "ipcio_view_eod: write_buf=%"PRIu64"\n",
     ipcbuf_get_write_count( buf ) );
#endif

  buf->viewbuf ++;

  if (ipcbuf_get_write_count( buf ) > buf->viewbuf)
    buf->viewbuf = ipcbuf_get_write_count( buf ) + 1;

  ipcio->bytes = 0;
  ipcio->curbuf = 0;

  uint64_t current = ipcio_tell (ipcio);
  uint64_t too_far = current % byte_resolution;
  if (too_far)
  {
    int64_t absolute_bytes = ipcio_seek (ipcio,
           current + byte_resolution - too_far,
           SEEK_SET);
    if (absolute_bytes < 0)
      return -1;
  }

  return 0;
}
