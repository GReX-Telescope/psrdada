/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

//#define PGPLOT_DEBUG 1

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "mopsr_def.h"
#include "dada_generator.h"
#include "dada_affinity.h"
#include "ascii_header.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <math.h>
#include <byteswap.h>
#include <complex.h>
#include <float.h>
#ifdef PGPLOT_DEBUG
#include <cpgplot.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>
#include <inttypes.h>
#include <fftw3.h>

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

void usage()
{
  fprintf (stdout,
           "mopsr_dbresampdb [options] in_key out_key\n"
           " -b core   bind process to CPU core\n"
           " -s        1 transfer, then exit\n"
           " -v        verbose mode\n");
}

typedef struct {

  // output DADA key
  key_t key;

  // output HDU
  dada_hdu_t * hdu;

  // number of bytes read
  uint64_t bytes_in;

  // number of bytes written
  uint64_t bytes_out;

  // verbose output
  int verbose;

  uint8_t * block;

  unsigned block_open;

  uint64_t bytes_written;

  uint8_t * d;

  unsigned nchan;

  unsigned npol;

  unsigned ndim;

  unsigned nant;

  unsigned nfft_fwd;

  unsigned nfft_bwd;

  void * fft_in;

  void * fft_mid1;

  void * fft_mid2;

  void * fft_out;

  fftwf_plan plan_fwd;

  fftwf_plan plan_bwd;

  unsigned int first;

} mopsr_dbupdb_t;

#define MOPSR_DBRESAMPDB_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }

/*! Function that opens the data transfer target */
int dbupdb_open (dada_client_t* client)
{

  // the mopsr_dbupdb specific data
  mopsr_dbupdb_t* ctx = 0;

  // status and error logging facilty
  multilog_t* log = 0;

  // header to copy from in to out
  char * header = 0;

  // header parameters that will be adjusted
  unsigned old_nbit = 0;
  unsigned new_nbit = 0;

  assert (client != 0);

  log = client->log;
  assert (log != 0);

  ctx = (mopsr_dbupdb_t *) client->context;
  assert (ctx != 0);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbupdb_open()\n");

  // lock writer status on the out HDU
  if (dada_hdu_lock_write (ctx->hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot lock write DADA HDU (key=%x)\n", ctx->key);
    return -1;
  }

  if (ascii_header_get (client->header, "NCHAN", "%d", &(ctx->nchan)) != 1)
  {
    ctx->nchan = 128;
    multilog (log, LOG_WARNING, "header had no NCHAN, assuming %d\n", ctx->nchan);
  }

  if (ascii_header_get (client->header, "NPOL", "%d", &(ctx->npol)) != 1)
  {
    ctx->npol = 1;
    multilog (log, LOG_WARNING, "header had no NPOL assuming %d\n", ctx->npol);
  }

  if (ascii_header_get (client->header, "NANT", "%d", &(ctx->nant)) != 1)
  {
    ctx->nant = 1;
    multilog (log, LOG_WARNING, "header had no NANT assuming %d\n", ctx->nant);
  }

  if (ascii_header_get (client->header, "NDIM", "%d", &(ctx->ndim)) != 1)
  {
    ctx->ndim = 2;
    multilog (log, LOG_WARNING, "header had no NDIM assuming %d\n", ctx->ndim);
  }

  float tsamp;
  if (ascii_header_get (client->header, "TSAMP", "%f", &tsamp) != 1)
  {
    tsamp = 1.08;
    multilog (log, LOG_ERR, "header had no TSAMP, assuming %f\n", tsamp);
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "parsed old tsamp=%f\n", tsamp);

  float new_tsamp = (tsamp * ctx->nfft_fwd) / ctx->nfft_bwd;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "calc'd new tsamp=%f\n", new_tsamp);

  uint64_t file_size = 0;
  uint64_t new_file_size = 0;
  if (ascii_header_get (client->header, "FILE_SIZE", "%"PRIu64, &file_size) != 1)
  {
    multilog (log, LOG_ERR, "header had no FILE_SIZE, ignoring\n");
  }
  else
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "old file_size=%"PRIu64"\n", file_size);
    new_file_size = (file_size * ctx->nfft_bwd) / ctx->nfft_fwd;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "calc'd new file_size=%"PRIu64"\n", new_file_size);
  }

  uint64_t bytes_per_second;
  uint64_t new_bytes_per_second = 0;
  if (ascii_header_get (client->header, "BYTES_PER_SECOND", "%"PRIu64, &bytes_per_second) != 1)
  {
    multilog (log, LOG_ERR, "header had no BYTES_PER_SECOND, ignoring\n");
  }
  else
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "old bytes_per_second=%"PRIu64"\n", bytes_per_second);
    new_bytes_per_second = (bytes_per_second * ctx->nfft_bwd) / ctx->nfft_fwd;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "calc'd new bytes_per_second=%"PRIu64"\n", new_bytes_per_second);
  }

  uint64_t obs_offset;
  uint64_t new_obs_offset = 0;
  if (ascii_header_get (client->header, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1)
  {
    multilog (log, LOG_ERR, "header had no OBS_OFFSET, ignoring\n");
  }
  else
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "old obs_offset=%"PRIu64"\n", obs_offset);
    new_obs_offset = (obs_offset * ctx->nfft_bwd) / ctx->nfft_fwd;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "calc'd new obs_offset=%"PRIu64"\n", new_obs_offset);
  }


  // get the header from the input data block
  uint64_t header_size = ipcbuf_get_bufsz (client->header_block);
  assert( header_size == ipcbuf_get_bufsz (ctx->hdu->header_block) );

  // get the next free header block on the out HDU
  header = ipcbuf_get_next_write (ctx->hdu->header_block);
  if (!header)  {
    multilog (log, LOG_ERR, "could not get next header block\n");
    return -1;
  }

  // copy the header from the in to the out
  memcpy ( header, client->header, header_size );

  if (ascii_header_set( header, "TSAMP", "%f", new_tsamp) < 0)
  {
    multilog (log, LOG_ERR, "failed to set TSAMP=%f in outgoing header\n", new_tsamp);
    return -1;
  }

  if (new_file_size)
  {
    if (ascii_header_set( header, "FILE_SIZE", "%"PRIu64, new_file_size) < 0)
    {
      multilog (log, LOG_ERR, "failed to set FILE_SIZE=%"PRIu64" in outgoing header\n", new_file_size);
      return -1;
    }
  }

  if (new_bytes_per_second)
  {
    if (ascii_header_set( header, "BYTES_PER_SECOND", "%"PRIu64, new_bytes_per_second) < 0)
    {
      multilog (log, LOG_ERR, "failed to set BYTES_PER_SECOND=%"PRIu64" in outgoing header\n", new_bytes_per_second);
      return -1;
    }
  }

  if (ascii_header_set( header, "OBS_OFFSET", "%"PRIu64, new_obs_offset) < 0)
  {
    multilog (log, LOG_ERR, "failed to set OBS_OFFSET=%"PRIu64" in outgoing header\n", new_obs_offset);
    return -1;
  }

  // mark the outgoing header as filled
  if (ipcbuf_mark_filled (ctx->hdu->header_block, header_size) < 0)  {
    multilog (log, LOG_ERR, "Could not mark filled Header Block\n");
    return -1;
  }

  if (ctx->verbose) 
    multilog (log, LOG_INFO, "HDU (key=%x) opened for writing\n", ctx->key);

  client->transfer_bytes = 0;
  client->optimal_bytes = 64*1024*1024;

  ctx->bytes_in = 0;
  ctx->bytes_out = 0;
  ctx->bytes_written = 0; 
  client->header_transfer = 0;

  // forward 64 point fft
  size_t in_fft_bytes = sizeof(float) * ctx->nfft_fwd * ctx->ndim;
  ctx->fft_in  = fftwf_malloc (in_fft_bytes);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "ctx->fft_in=%p size=%d\n", ctx->fft_in, in_fft_bytes);

  // output of forward fft
  ctx->fft_mid1 = fftwf_malloc (in_fft_bytes);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "ctx->fft_mid1=%p size=%d\n", ctx->fft_mid1, in_fft_bytes);

  // input to backward fft
  size_t out_fft_bytes = sizeof(float) * ctx->nfft_bwd * ctx->ndim;
  ctx->fft_mid2 = fftwf_malloc (out_fft_bytes);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "ctx->fft_mid2=%p size=%d\n", ctx->fft_mid2, out_fft_bytes);

  // output of the reverse fft
  ctx->fft_out = fftwf_malloc (out_fft_bytes);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "ctx->fft_out=%p size=%d\n", ctx->fft_out, out_fft_bytes);

  CHECK_ALIGN (ctx->fft_in);
  CHECK_ALIGN (ctx->fft_mid1);
  CHECK_ALIGN (ctx->fft_mid2);
  CHECK_ALIGN (ctx->fft_out);

  int fwd_direction_flags = FFTW_FORWARD;
  int flags = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;

  ctx->plan_fwd = fftwf_plan_dft_1d (ctx->nfft_fwd, (fftwf_complex*) ctx->fft_in, (fftwf_complex*) ctx->fft_mid1, fwd_direction_flags, flags);

  int bwd_direction_flags = FFTW_BACKWARD;
  ctx->plan_bwd = fftwf_plan_dft_1d (ctx->nfft_bwd, (fftwf_complex*) ctx->fft_mid2, (fftwf_complex*) ctx->fft_out, bwd_direction_flags, flags);

  return 0;
}


/*! Function that closes the data transfer */
int dbupdb_close (dada_client_t* client, uint64_t bytes_written)
{
  // the mopsr_dbupdb specific data
  mopsr_dbupdb_t* ctx = 0;

  // status and error logging facility
  multilog_t* log;

  assert (client != 0);

  ctx = (mopsr_dbupdb_t*) client->context;

  assert (ctx != 0);
  assert (ctx->hdu != 0);

  log = client->log;
  assert (log != 0);

  if (ctx->plan_fwd)
  {
    fftwf_destroy_plan(ctx->plan_fwd);
  }
  ctx->plan_fwd = 0;

  if (ctx->plan_bwd)
  {
    fftwf_destroy_plan(ctx->plan_bwd);
  }
  ctx->plan_bwd = 0;

  if (ctx->fft_in)
  {
    fftwf_free (ctx->fft_in);
  }
  ctx->fft_in = 0;

  if (ctx->fft_mid1)
  {
    fftwf_free (ctx->fft_mid1);
  }
  ctx->fft_mid1 = 0;

  if (ctx->fft_mid2)
  {
    fftwf_free (ctx->fft_mid2);
  }
  ctx->fft_mid2 = 0;

  if (ctx->fft_out)
  {
    fftwf_free (ctx->fft_out);
  }
  ctx->fft_out = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "bytes_in=%llu, bytes_out=%llu\n", ctx->bytes_in, ctx->bytes_out );

  if (ctx->block_open)
  {
    if (ipcio_close_block_write (ctx->hdu->data_block, ctx->bytes_written) < 0)
    {
      multilog (log, LOG_ERR, "dbupdb_close: ipcio_close_block_write failed\n");
      return -1;
    }
    ctx->block_open = 0;
    ctx->bytes_written = 0;
  }

  if (dada_hdu_unlock_write (ctx->hdu) < 0)
  {
    multilog (log, LOG_ERR, "dbupdb_close: cannot unlock DADA HDU (key=%x)\n", ctx->key);
    return -1;
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target via direct block IO*/
int64_t dbupdb_write_block (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t in_block_id)
{
  assert (client != 0);
  mopsr_dbupdb_t* ctx = (mopsr_dbupdb_t*) client->context;
  multilog_t * log = client->log;

  if (ctx->verbose) 
    multilog (log, LOG_INFO, "write_block: processing %llu bytes\n", in_data_size);

  // current DADA buffer block ID (unused)
  uint64_t out_block_id = 0;

  int64_t bytes_read = in_data_size;

  // number of bytes to be written to out DB
  uint64_t bytes_to_write = (in_data_size * ctx->nfft_bwd) / ctx->nfft_fwd;

  // input data pointer
  ctx->d = (int8_t *) in_data;  

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "block_write: opening block\n");

  // open output DB for writing
  if (!ctx->block_open)
  {
    ctx->block = (int8_t *) ipcio_open_block_write(ctx->hdu->data_block, &out_block_id);
    ctx->block_open = 1;
  }

  const uint64_t npart = in_data_size / (ctx->nchan * ctx->ndim * ctx->nant * ctx->nfft_fwd);

  if (in_data_size % (ctx->nchan * ctx->ndim * ctx->nant * ctx->nfft_fwd))
    multilog (log, LOG_ERR, "input block size [%"PRIu64"] % %d = %d, mismatch!!!!!\n", in_data_size, ctx->nchan * ctx->ndim * ctx->nant * ctx->nfft_fwd, in_data_size % (ctx->nchan * ctx->ndim * ctx->nant * ctx->nfft_fwd));
    
  // number of input elements between antenna
  const unsigned int ant_stride = ctx->ndim * ctx->npol;

  // number of input elements between channels 
  const unsigned int chan_stride = ctx->nant * ant_stride;

  // number of elements between successive complex values for same channel,ant
  const unsigned int in_stride = ctx->nchan * chan_stride;

  // number of input elements between parts
  const unsigned int in_part_stride  = in_stride * ctx->nfft_fwd;
  const unsigned int out_part_stride = in_stride * ctx->nfft_bwd;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write_block: npart=%"PRIu64" ant_stride=%u "
              " chan_stride=%u in_stride=%u in_part_stride=%u out_part_stride=%u\n",
              npart, ant_stride, chan_stride, in_stride, in_part_stride, out_part_stride);

  float * fft_ptr;
  unsigned int ipart, iant, ichan, ipt;
  int8_t * in;
  int8_t * out;
  unsigned int offset;

  size_t bytes_per_point   = ctx->ndim * sizeof(float);
  size_t fwd_bytes         = ctx->nfft_fwd * bytes_per_point;
  size_t half_bwd_bytes    = (ctx->nfft_bwd * bytes_per_point) / 2;
  size_t half_offset_bytes = (ctx->nfft_fwd - (ctx->nfft_bwd/2)) * bytes_per_point;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "write_block: half_bwd_bytes=%d half_offset_bytes=%d\n", half_bwd_bytes, half_offset_bytes);
  
  ctx->first = 1;

  unsigned unique_shifts = 32;
  float complex * facs = (float complex *) malloc (sizeof(float complex) * ctx->nchan * unique_shifts);
  float theta;
  float complex fac, val, res;
  unsigned i;
  float ratio = 2 * M_PI * (5.0 / 32.0);
  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    for (ipt=0; ipt < 32; ipt++)
    {
      i = (ichan * unique_shifts) + ipt;
      theta = (ichan + 20) * ratio * ipt;
      facs[i] = sin (theta) - cos(theta) * I;
    }
  }

  unsigned nfft = ctx->nfft_fwd;
  unsigned ndat = ctx->nchan * nfft;

  float * xpoints = (float *) malloc (sizeof(float) * ndat);
  float * detected = (float *) malloc (sizeof(float) * ndat);
  unsigned idat;
  unsigned half_fft, basechan, newchan;
  unsigned fft_offset;
  int bin;

  for (idat=0; idat < ndat; idat++)
    detected[idat] = 0;

  // unpack 8-bit data to float array for FFT
  for (ipart=0; ipart < npart; ipart++)
  {
    //multilog (log, LOG_INFO, "write_block: ipart=%d\n", ipart);
    for (iant=0; iant < ctx->nant; iant++)
    {
      for (ichan=0; ichan < ctx->nchan; ichan++)
      {
        in = (int8_t *) in_data + (ipart * in_part_stride) + (ichan * chan_stride) + (iant * ant_stride);
        fft_ptr = (float *) ctx->fft_in;

        for (ipt=0; ipt < ctx->nfft_fwd; ipt++)
        {
          val = (((float) in[0]) + 0.5) + (((float) in[1]) + 0.5) * I;
          i = (ichan * unique_shifts) + (ipt%32);
          res = val * facs[i];

          fft_ptr[0] = crealf(res);
          fft_ptr[1] = cimagf(res);

          fft_ptr += 2;
          in += in_stride;
        }

        // now whilst this is in cache, FFT (in-place)
        fftwf_execute_dft (ctx->plan_fwd, (fftwf_complex*) ctx->fft_in, (fftwf_complex*) ctx->fft_mid1);

#ifdef PGPLOT_DEBUG
        fft_ptr = (float *) ctx->fft_mid1;
        fft_offset = 0;
        half_fft = nfft / 2;

        // second half (flipped B)
        for (ipt=0; ipt < half_fft; ipt++)
        {
          basechan = ichan * nfft;
          newchan  = (ipt + half_fft);
          detected[basechan + newchan] += (fft_ptr[fft_offset] * fft_ptr[fft_offset]) + (fft_ptr[fft_offset+1] * fft_ptr[fft_offset+1]);
          fft_offset += 2;
        }
  
        if (nfft != ctx->nfft_fwd)
          fft_offset += (2 * (ctx->nfft_fwd - ctx->nfft_bwd));

        // first half (flipped A)
        for (ipt=half_fft; ipt < nfft; ipt++)
        {
          basechan = ichan * nfft;
          newchan  = (ipt - half_fft);
          detected[basechan + newchan] += (fft_ptr[fft_offset] * fft_ptr[fft_offset]) + (fft_ptr[fft_offset+1] * fft_ptr[fft_offset+1]);
          fft_offset += 2;
        }
#endif

        // memcpy B from input to ouput
        memcpy (ctx->fft_mid2, ctx->fft_mid1, half_bwd_bytes);

        // memcpy the inverse of A across
        memcpy (ctx->fft_mid2 + half_bwd_bytes, ctx->fft_mid1 + half_offset_bytes, half_bwd_bytes);

        // now whilst this is still in cache, bwd FFT! (in-place)
        fftwf_execute_dft (ctx->plan_bwd, (fftwf_complex*) ctx->fft_mid2, (fftwf_complex*) ctx->fft_out);

        // output base pointer
        out = (int8_t *) ctx->block + (ipart * out_part_stride) + (ichan * chan_stride) + (iant * ant_stride);
        fft_ptr = (float *) ctx->fft_out;

        // now whilst thisis still in cache, write to output array
        for (ipt=0; ipt < ctx->nfft_bwd; ipt++)
        {

          out[0] = (int8_t) floor((fft_ptr[0] / (float) ctx->nfft_bwd) - 0.5);
          out[1] = (int8_t) floor((fft_ptr[1] / (float) ctx->nfft_bwd) - 0.5);

          fft_ptr += 2;
          out += in_stride;
        }
        if (ichan == 1)
          ctx->first = 0;
      }
    }
  }

#if PGPLOT_DEBUG
  // now do a pgplot
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  multilog (log, LOG_INFO, "plot_packet: ndat=%d\n", ndat);

  // calculate limits
  for (idat=0; idat <ndat; idat++)
  {
    if (detected[idat] > ymax) ymax = detected[idat];
    if (detected[idat] < ymin) ymin = detected[idat];
    xpoints[idat] = idat;
  }

  float xmin = 0;
  float xmax = (float) ndat;
  xmax = 3 * nfft;
  ymin = 0;

  multilog (log, LOG_INFO, "plot_packet: xmin=%f, xmax=%f, ymin=%f, ymax=%f\n", xmin, xmax, ymin, ymax);

  cpgbbuf();
  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  cpglab("Freq", "Power", "Bandpass");
  cpgline(ndat, xpoints, detected);

  float line_x[2];
  float line_y[2];
  line_y[0] = ymin;
  line_y[1] = ymin + (ymax - ymin) / 2 ;

  cpgsls(2);
  for (ichan=0; ichan < ctx->nchan; ichan++)
  {
    line_x[0] = line_x[1] = (float) ichan * nfft;
    cpgline(2, line_x, line_y);
  }
  cpgsls(1);


  cpgebuf();

#endif

  //multilog (log, LOG_INFO, "write_block: in_data_size=%"PRIu64" bytes_to_write=%"PRIu64" bytes_read=%"PRIu64"\n", in_data_size, bytes_to_write, bytes_read);

  // close output DB for writing
  if (ctx->block_open)
  {
    ipcio_close_block_write(ctx->hdu->data_block, bytes_to_write);
    ctx->block_open = 0;
    ctx->block = 0;
  }

  ctx->bytes_in += bytes_read;
  ctx->bytes_out += bytes_to_write;

  free (detected);
  free (xpoints);

  //multilog (log, LOG_INFO, "write_block:  returning bytes_read=%"PRIu64"\n", bytes_read);

  return bytes_read;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbupdb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  fprintf(stderr, "dbupdb_write should be disabled!!!!!\n");

  return data_size;
}


int main (int argc, char **argv)
{
  /* DADA Data Block to Disk configuration */
  mopsr_dbupdb_t dbupdb = MOPSR_DBRESAMPDB_INIT;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Primary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  // bit promotion fator
  unsigned nfft_fwd = 128;
  unsigned nfft_bwd = 108;

  // number of transfers
  unsigned single_transfer = 0;

  // core to run on
  int core = -1;

  // input data block HDU key
  key_t in_key = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"c:dsv")) != -1)
  {
    switch (arg) 
    {
      
      case 'c':
        core = atoi(optarg);
        break;

      case 's':
        single_transfer = 1;
        break;

      case 'v':
        verbose++;
        break;
        
      default:
        usage ();
        return 0;
      
    }
  }

  char * device = "?";

  dbupdb.verbose = verbose;

  int num_args = argc-optind;
  unsigned i = 0;
   
  if (num_args != 2)
  {
    fprintf(stderr, "mopsr_dbupdb: must specify 2 datablocks\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  if (verbose > 1)
    fprintf (stderr, "parsing input key=%s\n", argv[optind]);
  if (sscanf (argv[optind], "%x", &in_key) != 1) {
    fprintf (stderr, "mopsr_dbupdb: could not parse in key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  if (verbose > 1)
    fprintf (stderr, "parsing output key=%s\n", argv[optind+1]);
  if (sscanf (argv[optind+1], "%x", &(dbupdb.key)) != 1) {
    fprintf (stderr, "mopsr_dbupdb: could not parse out key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }

  log = multilog_open ("mopsr_dbupdb", 0);

  multilog_add (log, stderr);

  if (verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // open connection to the in/read DB
  hdu = dada_hdu_create (log);

  dada_hdu_set_key (hdu, in_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", in_key);

  if (dada_hdu_lock_read (hdu) < 0)
    return EXIT_FAILURE;


  // open connection to the out/write DB
  dbupdb.hdu = dada_hdu_create (log);
  
  // set the DADA HDU key
  dada_hdu_set_key (dbupdb.hdu, dbupdb.key);
  
  // connect to the out HDU
  if (dada_hdu_connect (dbupdb.hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot connected to DADA HDU (key=%x)\n", dbupdb.key);
    return -1;
  } 

  //pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  //pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

  // output DB block size must be bit_p times the input DB block size
  int64_t in_block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu->data_block);
  int64_t out_block_size = ipcbuf_get_bufsz ((ipcbuf_t *) dbupdb.hdu->data_block);

  if (((in_block_size * nfft_bwd) / nfft_fwd) != out_block_size)
  {
    multilog (log, LOG_ERR, "output block size must be %d time the size of the input block size\n", (nfft_bwd / nfft_fwd));
    dada_hdu_disconnect (hdu);
    dada_hdu_disconnect (dbupdb.hdu);
    return EXIT_FAILURE;
  }

  if (in_block_size % 40 != 0)
  {
    multilog (log, LOG_ERR, "input block size must be a multile of NCHAN payload\n");
    dada_hdu_disconnect (hdu);
    dada_hdu_disconnect (dbupdb.hdu);
    return EXIT_FAILURE;
  }

  dbupdb.nfft_fwd = nfft_fwd;
  dbupdb.nfft_bwd = nfft_bwd;

  client = dada_client_create ();

  client->log = log;

  client->data_block   = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function  = dbupdb_open;
  client->io_function    = dbupdb_write;
  client->io_block_function = dbupdb_write_block;

  client->close_function = dbupdb_close;
  client->direction      = dada_client_reader;

  client->context = &dbupdb;
  client->quiet = (verbose > 0) ? 0 : 1;

#ifdef PGPLOT_DEBUG
  if (cpgopen("?") != 1)
  {
    multilog(log, LOG_INFO, "mopsr_dbplot: error opening plot device\n");
    exit(1);
  }
  cpgask(1);
#endif

  while (!client->quit)
  {
    if (verbose)
      multilog (log, LOG_INFO, "main: dada_client_read()\n");

    if (dada_client_read (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (verbose)
      multilog (log, LOG_INFO, "main: dada_hdu_unlock_read()\n");

    if (dada_hdu_unlock_read (hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      return EXIT_FAILURE;
    }

    if (single_transfer)
      client->quit = 1;

    if (!client->quit)
    {
      if (dada_hdu_lock_read (hdu) < 0)
      {
        multilog (log, LOG_ERR, "could not lock read on hdu\n");
        return EXIT_FAILURE;
      }
    }
  }

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

#ifdef PGPLOT_DEBUG
  cpgclos();
#endif
  return EXIT_SUCCESS;
}

