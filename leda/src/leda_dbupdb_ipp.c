/***************************************************************************
 *  
 *    Copyright (C) 2011 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "leda_def.h"
#include "dada_generator.h"

#include "ascii_header.h"
#include "daemon.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <math.h>
#include <byteswap.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>

#include <ipps.h>

#define IDLE 1
#define ACTIVE 2
#define QUIT 3

void usage()
{
  fprintf (stdout,
           "leda_dbupdb [options] in_key out_key\n"
           " -b num    bit promotion factor [default 2]\n"
           " -n nthr   use nthr threads to convert\n"
           " -s        1 transfer, then exit\n"
           " -S        1 observation with multiple transfers, then exit\n"
           " -t num    decimation factor [default 0]\n"
           " -z        use zero copy transfers\n"
           " -v        verbose mode\n");
}

#define UDP_DATA_64 UDP_DATA/8

typedef struct {
  //uint64_t header;
  uint64_t data[UDP_DATA_64];
} roach_pkt_t;

typedef union roach_data_block {
  uint64_t data[UDP_DATA_64];
  uint16_t dualpol[UDP_DATA/2];
  uint8_t smples[UDP_DATA];
} roach_blk_t;

typedef struct {

  // output DADA key
  key_t key;

  // output HDU
  dada_hdu_t * hdu;

  // bit promotion factor
  // arbitary bit promotion is not yet implemented
  // currently converts 4R+4C data into 8R+8C for 
  // efficient GPU reading
  unsigned bit_p;

  // number of bytes read
  uint64_t bytes_in;

  // number of bytes written
  uint64_t bytes_out;

  // verbose output
  int verbose;

  // number of bytes of consective pol
  int resolution;

  // pointer to cast to output memory
  uint8_t * out_mem;

  // required size of output memory
  uint64_t out_size;

  int8_t * block;

  unsigned block_open;

  uint64_t outdat;

  uint64_t bytes_written;

  unsigned ocnt;

  unsigned quit;

  pthread_cond_t * cond;

  pthread_mutex_t * mutex;

  unsigned nthreads;

  unsigned state;

  unsigned * thr_states;

  uint64_t * thr_start_packet;

  uint64_t * thr_end_packet;

  unsigned thread_count;

  uint64_t * d;

  unsigned nchan;

  unsigned npol;

  unsigned ndim;

} leda_dbupdb_t;

#define LEDA_DBUPDB_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

/*! Function that opens the data transfer target */
int dbupdb_open (dada_client_t* client)
{

  // the leda_dbupdb specific data
  leda_dbupdb_t* ctx = 0;

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

  ctx = (leda_dbupdb_t *) client->context;
  assert (ctx != 0);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbupdb_open()\n");

  // lock writer status on the out HDU
  if (dada_hdu_lock_write (ctx->hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot lock write DADA HDU (key=%x)\n", ctx->key);
    return -1;
  }

  if (ascii_header_get (client->header, "NBIT", "%d", &old_nbit) != 1)
  {
    old_nbit = 4; 
    multilog (log, LOG_WARNING, "header had no NBIT, assuming %d\n", old_nbit);
  }

  if (ascii_header_get (client->header, "NCHAN", "%d", &(ctx->nchan)) != 1)
  {
    ctx->nchan = 407;
    multilog (log, LOG_WARNING, "header had no NCHAN, assuming %d\n", ctx->nchan);
  }

  if (ascii_header_get (client->header, "NPOL", "%d", &(ctx->npol)) != 1)
  {
    ctx->npol = 2;
    multilog (log, LOG_WARNING, "header had no NPOL assuming %d\n", ctx->npol);
  }

  if (ascii_header_get (client->header, "NDIM", "%d", &(ctx->ndim)) != 1)
  {
    ctx->ndim = 2;
    multilog (log, LOG_WARNING, "header had no NDIM assuming %d\n", ctx->ndim);
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "parsed old NBIT=%d, NCHAN=%d NPOL=%d NDIM=%d\n",
                             old_nbit, ctx->nchan, ctx->npol, ctx->ndim);

  new_nbit = ctx->bit_p * old_nbit;

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

  return 0;
}

/*! Function that closes the data transfer */
int dbupdb_close (dada_client_t* client, uint64_t bytes_written)
{
  // the leda_dbupdb specific data
  leda_dbupdb_t* ctx = 0;

  // status and error logging facility
  multilog_t* log;

  assert (client != 0);

  ctx = (leda_dbupdb_t*) client->context;

  assert (ctx != 0);
  assert (ctx->hdu != 0);

  log = client->log;
  assert (log != 0);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "bytes_in=%"PRIu64", bytes_out=%"PRIu64"\n",
                    ctx->bytes_in, ctx->bytes_out );

  if (ctx->block_open)
  {
    if (ipcio_close_block_write (ctx->hdu->data_block, ctx->bytes_written) < 0)
    {
      multilog (log, LOG_ERR, "dbupdb_close: ipcio_close_block_write failed\n");
      return -1;
    }
    ctx->block_open = 0;
    ctx->outdat = 0;
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
  leda_dbupdb_t* ctx = (leda_dbupdb_t*) client->context;
  multilog_t * log = client->log;

  if (ctx->verbose) 
    multilog (log, LOG_INFO, "write_block: processing %"PRIu64" bytes\n", in_data_size);

  // current DADA buffer block ID (unused)
  uint64_t out_block_id = 0;

  int64_t bytes_read = in_data_size;

  // number of bytes to be written to out DB
  uint64_t bytes_to_write = in_data_size * ctx->bit_p;

  // input data pointer
  ctx->d = (uint64_t *) in_data;  

  if (ctx->verbose)
    multilog (log, LOG_INFO, "block_write: opening block\n");

  // open output DB for writing
  if (!ctx->block_open)
  {
    ctx->block = (int8_t *) ipcio_open_block_write(ctx->hdu->data_block, &out_block_id);
    ctx->block_open = 1;
  }

  // lock mutex
  pthread_mutex_lock (ctx->mutex);

  // prepare thread boundaries
  unsigned numpackets  = in_data_size / UDP_DATA;
  unsigned packets_per_thread = (unsigned) floor(numpackets / ctx->nthreads);
  unsigned i = 0;

  packets_per_thread -= (packets_per_thread % 4);

  for (i=0; i < ctx->nthreads; i++)
  {
    ctx->thr_start_packet[i] = i * packets_per_thread;
    ctx->thr_end_packet[i]   = ctx->thr_start_packet[i] + packets_per_thread;
    ctx->thr_states[i] = ACTIVE;
  }
  ctx->state = ACTIVE;

  if (numpackets % ctx->nthreads)
    ctx->thr_end_packet[ctx->nthreads - 1] = numpackets;

  ctx->d = (uint64_t *) in_data;

  // activate threads
  pthread_cond_broadcast (ctx->cond);
  pthread_mutex_unlock (ctx->mutex);

  // threads a working here...

  // wait for threads to finish
  pthread_mutex_lock (ctx->mutex);
  while (ctx->state == ACTIVE)
  {
    unsigned threads_finished = 0;
    while (!threads_finished)
    {
      threads_finished = 1;
      for (i=0; i<ctx->nthreads; i++)
      {
        if (ctx->thr_states[i] != IDLE)
          threads_finished = 0;
      }

      if (threads_finished)
        ctx->state = IDLE;
      else
        pthread_cond_wait (ctx->cond, ctx->mutex);
    }
  }
  pthread_mutex_unlock (ctx->mutex);

  // close output DB for writing
  if (ctx->block_open)
  {
    ipcio_close_block_write(ctx->hdu->data_block, bytes_to_write);
    ctx->block_open = 0;
    ctx->block = 0;
  }

  return bytes_read;

}
void * leda_dbudpdb_bitpromote_thread (void * arg)
{

  leda_dbupdb_t* ctx = (leda_dbupdb_t*) arg;

  // lock the mutex to 
  pthread_mutex_lock (ctx->mutex);
  const unsigned ithread = ctx->thread_count;
  ctx->thread_count++;

  if (ctx->verbose)
    fprintf(stderr, "[%d] bit promote thread starting\n", ithread);


  // buffer for doing 4->8 bit coversion in vector
  Ipp8u * reals = ippsMalloc_8u (UDP_DATA / 2);
  Ipp8u * imags = ippsMalloc_8u (UDP_DATA / 2);
  Ipp8u   mask  = 15;
  Ipp8u * in_dat = 0;
  Ipp8u * out_dat_1 = 0;        // for first time sample in packet
  Ipp8u * out_dat_2 = 0;        // for second time sample in packet

  uint64_t n_ands = 0;
  uint64_t n_shifts = 0;
  uint64_t n_assigns = 0;

  uint64_t npkt_processed = 0;
  uint64_t ipkt;
  unsigned int i = 0;
  const unsigned nant_per_packet = 4;
  const unsigned npacket_per_resolution = 4;
  unsigned out_time_sample_stride = 0;

  while (ctx->thr_states[ithread] != QUIT)
  {
    while (ctx->thr_states[ithread] == IDLE)
      pthread_cond_wait (ctx->cond, ctx->mutex);

    if (ctx->thr_states[ithread] == QUIT)
    {
      fprintf(stderr, "[%d] processed %"PRIu64" packets\n", ithread, npkt_processed);
      fprintf(stderr, "n_ands = %"PRIu64"\n", n_ands);
      fprintf(stderr, "n_shifts = %"PRIu64"\n", n_shifts);
      fprintf(stderr, "n_assigns = %"PRIu64"\n", n_assigns);
      pthread_mutex_unlock (ctx->mutex);
      ippsFree(reals);
      ippsFree(imags);
      pthread_exit (NULL);
    }
  
    out_time_sample_stride = npacket_per_resolution * ctx->nchan * 
                             nant_per_packet * ctx->npol * ctx->ndim; 

#define BOB 1

    if (ctx->verbose)
      fprintf(stderr, "[%d] processing packets %"PRIu64" - %"PRIu64"\n", 
                    ithread, ctx->thr_start_packet[ithread],  ctx->thr_end_packet[ithread]);

    pthread_mutex_unlock (ctx->mutex);

    in_dat = (Ipp8u *) ctx->d;
    in_dat += UDP_DATA * ctx->thr_start_packet[ithread];
  
    out_dat_1 = (Ipp8u *) ctx->block;
    out_dat_1 += (UDP_DATA * ctx->thr_start_packet[ithread] * 2);

    out_dat_2 = out_dat_1 + out_time_sample_stride;

    // First half of each packet contains 1 time sample for 407 chan, 4 Antenna, 2 pol, 2 dim, 4 bits / sample
    for (ipkt=ctx->thr_start_packet[ithread]; ipkt < ctx->thr_end_packet[ithread]; ipkt++)
    {
      if (ipkt > ctx->thr_start_packet[ithread] && ipkt % npacket_per_resolution == 0)
      {
        out_dat_1 += out_time_sample_stride;
        out_dat_2 += out_time_sample_stride;
      }

      // first half of packet
      ippsRShiftC_8u(in_dat, 4, reals, UDP_DATA/2);   // first shift right 4 bits left [REAL]
      n_shifts += UDP_DATA/2;
      ippsAndC_8u(in_dat, mask, imags, UDP_DATA/2);   // then mask 4 MSB's to produce [IMAG]
      n_ands += UDP_DATA/2;
      for (i=0; i<UDP_DATA/2; i++)
      {
        out_dat_1[2*i]   = reals[i];
        out_dat_1[2*i+1] = imags[i];
      }
      n_assigns += UDP_DATA;

      in_dat += UDP_DATA/2;

      // second half of packet
      ippsRShiftC_8u(in_dat, 4, reals, UDP_DATA/2);    // first shift right 4 bits left [REAL]
      n_shifts += UDP_DATA/2;
      ippsAndC_8u(in_dat, mask, imags, UDP_DATA/2);    // then mask 4 MSB's to produce [IMAG]
      n_ands += UDP_DATA/2;
      for (i=0; i<UDP_DATA/2; i++)
      {
        out_dat_2[2*i]   = reals[i];
        out_dat_2[2*i+1] = imags[i];
      }
      n_assigns += UDP_DATA;
      
      in_dat += UDP_DATA/2;
      out_dat_1 += UDP_DATA;
      out_dat_2 += UDP_DATA;

      npkt_processed++;
    }

    pthread_mutex_lock (ctx->mutex);
    ctx->thr_states[ithread] = IDLE;
    pthread_cond_broadcast (ctx->cond);
  }
  ippsFree(reals);
  ippsFree(imags);
  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbupdb_write (dada_client_t* client, void* data, uint64_t data_size)
{

  roach_pkt_t *roachpkt;
  roach_blk_t *roachblk;
  
  // the leda_dbupdb specific data
  leda_dbupdb_t* ctx = 0;
  
  // status and error logging facility
  multilog_t * log = 0;
  
  assert (client != 0);
  ctx = (leda_dbupdb_t*) client->context;
  
  assert (client->log != 0);
  log = client->log;

  assert (ctx != 0);
  assert (ctx->hdu != 0);
 
  if (ctx->verbose) 
    multilog (log, LOG_INFO, "write: processing %"PRIu64" bytes\n", data_size);

  // promote the data
  uint64_t indat = 0;
  uint64_t prom_data_size = data_size * ctx->bit_p;
  uint64_t fourwords=0;
  uint16_t dualpol;
  uint16_t *fourby16;  

  int sample=0;
  int word=0;
  int NPOL=2;
 
  uint8_t re;
  uint8_t im;
  uint8_t complexpol;
  

  // input data pointer
  uint64_t * d = (uint64_t *) data;  
  
  // allocate output memory with enough space to deal with all the promoted data
  ctx->out_mem = (uint8_t *) malloc(sizeof(uint8_t) * prom_data_size);
  
  uint8_t  read_index;
  uint8_t write_index;
  uint numpackets  = data_size / UDP_DATA; 
  uint antenna_repeat = 0;

  //fprintf(stderr, "numpackets=%d\n", numpackets);

  roachpkt = (roach_pkt_t *) &d;
  roachblk = (roach_blk_t *) roachpkt->data;

    /*
      packet contained 128 dual pol 4 bit samples
      from 1 antenna, for 1 channel before moving onto the next
      antenna. THIS WAS CHANGED FOR THE UPDATED ROACH DESIGN

      eg [chan0][ant1][pol0][re][im][pol1][re][im]x128
         [chan0][ant2][pol0][re]...etc

      was to rearranged to be:
      [chan0][ant1][pol0][re][im][pol1][re][im]
      [chan0][ant2][pol0][re][im][pol1][re][im] 
      
      ie, no 128 consequtive samples. -> check reading gulp for the gpu

      the main assumption is that the packets are structured in such a way
      that we are always starting on channel0.
    */


  if (ctx->verbose) 
    multilog (log, LOG_INFO, "processing packet\n");
  
  // process each packet in the datablock
  for (indat=0; indat < numpackets; indat++)
  {  
    /* 
       there are multiple time samples for each set of 
       channels, in this case, 20. (needs to be updated to a 
       variable - something like UDP_DATA / NUM_CHANS_PER_ROACH         
    */
    for (antenna_repeat=0;antenna_repeat<2;antenna_repeat++)
    {  
      // process each UDP_DATA_64 samples in the current packet
      for (word=0;word<(UDP_DATA_64/2);word++)
      {
        read_index = word+(antenna_repeat*UDP_DATA_64/2);        
        fourwords = bswap_64(roachblk->data[read_index]);

        fourby16 = (uint16_t *) &fourwords;
        for (sample=0;sample<4;sample++) 
        {
          dualpol = fourby16[sample];          
          write_index = read_index*2*NPOL;
          
          // are the pols getting written out in the correct order? (test this!)
          
          //pol0
          complexpol = (dualpol >> 8) & 0xff;
          im = complexpol & 0xf;
          re = (complexpol >> 4) & 0xf;
          ctx->out_mem[write_index] = re;
          ctx->out_mem[write_index+1] = im;
          
          //pol1
          complexpol = dualpol & 0xff;
          im = complexpol & 0xf;
          re = (complexpol >> 4) & 0xf;
          ctx->out_mem[write_index+2] = re;
          ctx->out_mem[write_index+3] = im;
        } // close sample loop
      } // close word loop      
    } // close antenna_repeat loop
    roachpkt++;
  } // close indat loop
  
  
  if (ctx->verbose) 
    multilog (log, LOG_INFO, "data block finished\n");

  // write data to the current queue
  // to turn into zero-copy should just be a change of this function + verification
  // that the data_block sizes etc are correct.
  
  //int64_t bytes_written = ipcio_write (ctx->hdu->data_block, data, prom_data_size);
  ipcio_write(ctx->hdu->data_block, (void *) ctx->out_mem, prom_data_size);
  
  ctx->bytes_in += data_size;
  ctx->bytes_out += prom_data_size;
  
  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbupdb_write: read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, prom_data_size);
  
  free(ctx->out_mem);

  if (ctx->verbose)
    multilog(log, LOG_INFO, "leaving dbupdb_write\n");

  return data_size;
}


int main (int argc, char **argv)
{
  /* DADA Data Block to Disk configuration */
  leda_dbupdb_t dbupdb = LEDA_DBUPDB_INIT;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Primary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  // decimation factor
  unsigned tdec = 0;

  // bit promotion fator
  unsigned bit_p = 2;

  // number of processing threads
  unsigned n_threads = 1;

  // number of transfers
  unsigned single_transfer = 0;

  // single transfer with multiple xfers
  unsigned quit_xfer = 0;

  // use zero copy transfers
  unsigned zero_copy = 0;

  // input data block HDU key
  key_t in_key = 0;

  // thread IDs
  pthread_t * ids = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"b:dn:sSt:vz")) != -1)
  {
    switch (arg) 
    {
      case 'b':
        if (!optarg)
        {
          fprintf (stderr, "leda_dbupdb: -b requires argument\n");
          usage();
          return EXIT_FAILURE;
        }
        if (sscanf (optarg, "%d", &bit_p) != 1)
        {
          fprintf (stderr, "leda_dbupdb: could not parse bit_p from %s\n",optarg);
          usage();
          return EXIT_FAILURE;
        }
        break;
      
      case 'd':
        daemon = 1;
        break;

      case 'n':
        n_threads= atoi(optarg);
        break;

      case 's':
        single_transfer = 1;
        break;

      case 'S':
        quit_xfer = 1;
        break;

      case 't':
        if (!optarg)
        { 
          fprintf (stderr, "leda_dbupdb: -t requires argument\n");
          usage();
          return EXIT_FAILURE;
        }
        if (sscanf (optarg, "%d", &tdec) != 1) 
        {
          fprintf (stderr ,"leda_dbupdb: could not parse tdec from %s\n", optarg);
          usage();
          return EXIT_FAILURE;
        } 
        
      case 'v':
        verbose++;
        break;
        
      case 'z':
        zero_copy = 1;
        break;
        
      default:
        usage ();
        return 0;
      
    }
  }

  dbupdb.verbose = verbose;
  dbupdb.bit_p = bit_p;

  int num_args = argc-optind;
  int i = 0;
      
  if (num_args != 2)
  {
    fprintf(stderr, "leda_dbupdb: must specify 2 datablocks\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  if (verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind]);
  if (sscanf (argv[optind], "%x", &in_key) != 1) {
    fprintf (stderr, "leda_dbupdb: could not parse in key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  if (verbose)
    fprintf (stderr, "parsing output key=%s\n", argv[optind+1]);
  if (sscanf (argv[optind+1], "%x", &(dbupdb.key)) != 1) {
    fprintf (stderr, "leda_dbupdb: could not parse out key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }

  log = multilog_open ("leda_dbupdb", 0);

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

  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

  // if zero copy then output DB block size must be bit_p times the input DB block size
  if (zero_copy)
  {
    dbupdb.nthreads = n_threads;

    int64_t in_block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu->data_block);
    int64_t out_block_size = ipcbuf_get_bufsz ((ipcbuf_t *) dbupdb.hdu->data_block);

    if (in_block_size * bit_p != out_block_size)
    {
      multilog (log, LOG_ERR, "output block size must be bit_p time the size of the input block size\n");
      dada_hdu_disconnect (hdu);
      dada_hdu_disconnect (dbupdb.hdu);
      return EXIT_FAILURE;
    }

    dbupdb.cond = &cond;
    dbupdb.mutex = &mutex;

    pthread_cond_init( dbupdb.cond, NULL);
    pthread_mutex_init( dbupdb.mutex, NULL);

    dbupdb.thr_states = (unsigned *) malloc (dbupdb.nthreads * sizeof(unsigned));
    dbupdb.thr_start_packet = (uint64_t *) malloc (dbupdb.nthreads * sizeof(uint64_t));
    dbupdb.thr_end_packet = (uint64_t *) malloc (dbupdb.nthreads * sizeof(uint64_t));

    ids = (pthread_t *) malloc (dbupdb.nthreads * sizeof(pthread_t));
    for (i=0; i< dbupdb.nthreads; i++)
    {
      dbupdb.thr_states[i] = IDLE;
      pthread_create (&(ids[i]), 0,  leda_dbudpdb_bitpromote_thread, (void *) &dbupdb);
    }
  }

  client = dada_client_create ();

  client->log = log;

  client->data_block   = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function  = dbupdb_open;
  client->io_function    = dbupdb_write;
  if (zero_copy)
    client->io_block_function = dbupdb_write_block;

  client->close_function = dbupdb_close;
  client->direction      = dada_client_reader;

  client->context = &dbupdb;
  client->quiet = (verbose > 0) ? 0 : 1;

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

    if (single_transfer || (quit_xfer && dbupdb.quit))
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

  if (zero_copy)
  {

    pthread_mutex_lock (dbupdb.mutex);
    while (dbupdb.state != IDLE)
      pthread_cond_wait (dbupdb.cond, dbupdb.mutex);

    for (i=0; i< dbupdb.nthreads; i++)
    {
      dbupdb.thr_states[i] = QUIT;
    }
    dbupdb.state = QUIT;

    pthread_cond_broadcast (dbupdb.cond);
    pthread_mutex_unlock (dbupdb.mutex);
    
    // join threads
    for (i=0; i<dbupdb.nthreads; i++)
      (void) pthread_join(ids[i], NULL);

    pthread_cond_destroy (dbupdb.cond);
    pthread_mutex_destroy (dbupdb.mutex);
    free(dbupdb.thr_states);
    free(dbupdb.thr_start_packet);
    free(dbupdb.thr_end_packet);
    free(ids);
  }

  return EXIT_SUCCESS;
}

