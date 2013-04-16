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

void usage ();
int ipcio_view_eod (ipcio_t* ipcio, unsigned byte_resolution);

void usage()
{
  fprintf (stdout,
     "mopsr_dbmonitor [options]\n"
     " -k key     connect to key data block\n"
     " -t num     integrate num frames into monitoring output\n"
     " -D device  pgplot device name [default ?]\n"
     " -v         be verbose\n");
}

int main (int argc, char **argv)
{
  /* DADA Header plus Data Unit */
  dada_hdu_t * hdu = 0;

  /* DADA Logger */
  multilog_t * log = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  unsigned int verbose = 0;

  /* Quit flag */
  char quit = 0;

  /* PGPLOT device name */
  char * device = "/xs";

  /* number of frames to process */
  unsigned int nframe = 1024;

  /* dada key for SHM */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;

  while ((arg=getopt(argc,argv,"D:t:vk:")) != -1)
  {
    switch (arg)
    {
      
      case 'D':
        device = strdup(optarg);
        break;

      case 't':
        nframe = atoi(optarg);
        break;

      case 'v':
        verbose++;
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

  fprintf (stderr, "main: initialization\n");

  log = multilog_open ("mopsr_dbmonitor", 0);
  multilog_add (log, stderr);

  hdu = dada_hdu_create (log);
  dada_hdu_set_key(hdu, dada_key);

  fprintf (stderr, "main: dada_hdu_connect()\n");
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "ERROR: could not connect to HDU on key %x\n", dada_key);
    return EXIT_FAILURE;
  }

  fprintf (stderr, "main: dada_hdu_open_view()\n");
  if (dada_hdu_open_view(hdu) < 0)
  {
    dada_hdu_disconnect (hdu);
    fprintf (stderr, "ERROR: could not open HDU for viewing\n");
    return EXIT_FAILURE;
  }

  fprintf (stderr, "main: dada_hdu_open()\n");
  if (dada_hdu_open (hdu) < 0)
  {
    dada_hdu_disconnect (hdu);
    fprintf (stderr, "ERROR: could not open HDU\n");
    return EXIT_FAILURE;
  }

  if (cpgopen(device) != 1) 
  {
    multilog(log, LOG_INFO, "mopsr_dbplot: error opening plot device\n");
    exit(1);
  }
  cpgask(0);

  // extract key paramters from the HDU's headera
  unsigned int byte_resolution;
  fprintf (stderr, "main: ascii_header_get (RESOULTION)\n");
  if (ascii_header_get (hdu->header, "RESOLUTION", "%u", &byte_resolution) < 0)
  {
    multilog (log, LOG_WARNING, "HEADER with no RESOLUTION\n");
    byte_resolution = 0;
  }

  mopsr_util_t opts;
  opts.plot_log = 1;
  opts.zap = 0;
  opts.ant = -1;
  opts.chan = -1;
  opts.ymin =  1000;
  opts.ymax = -1000;

  if (ascii_header_get (hdu->header, "NCHAN", "%u", &(opts.nchan)) < 0)
  {
    multilog (log, LOG_ERR, "HEADER with no NCHAN\n");
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  if (ascii_header_get (hdu->header, "NANT", "%u", &(opts.nant)) < 0)
  {
    multilog (log, LOG_ERR, "HEADER with no NANT\n");
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }
  
  if (ascii_header_get (hdu->header, "NDIM", "%u", &(opts.ndim)) < 0)
  {
    multilog (log, LOG_ERR, "HEADER with no NDIM\n");
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  if (!byte_resolution)
    byte_resolution = opts.nchan * opts.nant * opts.ndim;

  if (byte_resolution != opts.nchan * opts.nant * opts.ndim) 
  {
    multilog (log, LOG_WARNING, "RESOLUTION not correct\n");
    byte_resolution = opts.nchan * opts.nant * opts.ndim;
  }

  fprintf (stderr, "main: byte_resolution=%u\n", byte_resolution);
  uint64_t bytes_to_read = byte_resolution * nframe;
  void * buffer = malloc (bytes_to_read);
  int64_t bytes_read;
  unsigned int ispectra = 0;
  unsigned int nspectra = nframe;

  // TODO improve this
  while ( 1 )
  {
    multilog (log, LOG_INFO, "main: ipcio_view_eod()\n");
    // move to the last block/byte based on resolution
    if (ipcio_view_eod (hdu->data_block, byte_resolution) < 0)
    {
      multilog (log, LOG_ERR, "main: ipcio_view_eod failed\n");
      dada_hdu_disconnect (hdu);
      return EXIT_FAILURE;
    }

    // read the required amount from the data block
    bytes_read = ipcio_read (hdu->data_block, (char *) buffer, bytes_to_read);
    if (bytes_read < 0)
    {
      multilog (log, LOG_ERR, "main: ipcio_read failed\n");
      free (buffer);
      dada_hdu_disconnect (hdu);
      return EXIT_FAILURE;
    }

    if (bytes_read < bytes_to_read)
    {
      multilog (log, LOG_WARNING, "main: ipcio_read returned %"PRIi64" bytes, "
                "requested %"PRIu64"\n", bytes_read, bytes_to_read);
    }

    // now do something with the data that has been read
    if (verbose)
    {
      multilog (log, LOG_INFO, "main: mopsr_print_frame\n");
      mopsr_print_pfbframe (buffer, (ssize_t) bytes_read, &opts);
    }

    // perform SQLD on buffer / frame
    float spectra[opts.nchan * opts.nant];
    float waterfall[nspectra * opts.nchan];

    for (ispectra = 0; ispectra < nspectra; ispectra++)
    {
      mopsr_sqld_pfbframe (spectra, buffer + (ispectra * opts.nchan * opts.nant * 2), &opts, 0);
      memcpy (waterfall + (ispectra * opts.nchan), spectra, opts.nchan * sizeof(float));
    }

    if (ispectra >= nspectra)
    {
      //mopsr_plot_bandpass (power_spectra, &opts);
      mopsr_plot_waterfall (waterfall, nspectra, &opts);
    }

    multilog (log, LOG_INFO, "main: sleep (1)\n");
    sleep (1);
  }
  if (dada_hdu_disconnect (hdu) < 0)
  {
    fprintf (stderr, "ERROR: could not disconnect from HDU\n");
    return EXIT_FAILURE;
  }

  cpgclos();

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
