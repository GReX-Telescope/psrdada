/*
 * bpsr_udpheader
 *
 * Simply listens on a specified port for udp packets encoded
 * in the PARSEPC format
 *
 */

#include <sys/socket.h>
#include <math.h>

#include "config.h"
#include "bpsr_udpheader.h"
#include "sock.h"

void usage()
{
  fprintf (stdout,
	   "bpsr_udpheader [options]\n"
     " -h             print help text\n"
#ifdef HAVE_CUDA
     " -d device      use CUDA device for DMA zeroing of buffers\n"
#endif
	   " -i interface   ip/interface for inc. UDP packets [default all]\n"
	   " -p port        port on which to listen [default %d]\n"
     " -a acc_len     accumulation length of the iBob board [default %d]\n"
     " -x             assume cross polarisation BPSR\n"
     " -v             verbose messages\n"
     " -V             very verbose messages\n", 
     BPSR_DEFAULT_UDPDB_PORT, BPSR_DEFAULT_ACC_LEN);
}

void udpheader_init (udpheader_t * ctx)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "udpheader_init()\n");
  // initialize a socket large enough for vanilla or cross pol BPSR packets
  ctx->sock    = bpsr_init_sock();

  // initialize some stat structures
  ctx->packets = init_stats_t();
  ctx->bytes   = init_stats_t();

  ctx->sequence_incr = 512 * ctx->acc_len;
  ctx->spectra_per_second = (BPSR_IBOB_CLOCK * 1000000 / ctx->acc_len ) / BPSR_IBOB_NCHANNELS;
  ctx->bytes_per_second = ctx->spectra_per_second * ctx->packet_data_size;

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "udpheader_init seq_incr=%lu bytes_per_second=%lu\n", ctx->sequence_incr, ctx->bytes_per_second);

  ctx->block_size = ctx->packet_data_size * ctx->packets_per_block;
  multilog (ctx->log, LOG_INFO, "udpheader_init: block_size=%lu\n", ctx->block_size);

  // zero the buffers
#ifdef HAVE_CUDA
  if (ctx->device >= 0)
  {
    fprintf (stderr, "alloc GPU\n");
    cudaError_t err;
    dada_cuda_select_device (ctx->device);
    err = cudaStreamCreate (&(ctx->stream));
    ctx->gpu_buffer = dada_cuda_device_malloc (ctx->block_size);
    ctx->curr_buffer = dada_cuda_host_malloc (ctx->block_size);
    ctx->next_buffer = dada_cuda_host_malloc (ctx->block_size);
    ctx->zero_buffer = dada_cuda_host_malloc (ctx->block_size);
    cudaMemsetAsync(ctx->gpu_buffer, 0, ctx->block_size, ctx->stream);
    err = cudaMemcpyAsync (ctx->curr_buffer, ctx->gpu_buffer, 
                           ctx->block_size, cudaMemcpyDeviceToHost, 
                           ctx->stream);
    err = cudaMemcpyAsync (ctx->next_buffer, ctx->gpu_buffer, 
                           ctx->block_size, cudaMemcpyDeviceToHost, 
                           ctx->stream);
    err = cudaStreamSynchronize(ctx->stream);
  }
  else
#endif
  {
    ctx->curr_buffer = (char *) malloc(ctx->block_size);
    assert(ctx->curr_buffer != 0);
    ctx->next_buffer = (char *) malloc(ctx->block_size);
    assert(ctx->next_buffer != 0);
    ctx->zero_buffer = (char *) malloc(ctx->block_size);
    assert(ctx->zero_buffer != 0);

    bzero (ctx->curr_buffer, ctx->block_size);
    bzero (ctx->next_buffer, ctx->block_size);
  }
}


time_t udpheader_start_function (udpheader_t* ctx, time_t start_utc)
{
  multilog_t* log = ctx->log;

  /* Initialise variables */
  ctx->select_sleep = 0;
  ctx->prev_time = time(0);
  ctx->current_time = ctx->prev_time;
  ctx->curr_buffer_count = 0;
  ctx->next_buffer_count = 0;
  ctx->packets_late_this_sec = 0;
  ctx->packet_in_buffer = 0;

  // setup the expected sequence no to the initial value
  ctx->expected_sequence_no = 0;

  // create the UDP socket
  ctx->sock->fd = dada_udp_sock_in(ctx->log, ctx->interface, ctx->port, ctx->verbose);
  if (ctx->sock->fd < 0)
  {
    multilog (log, LOG_ERR, "failed to create udp socket\n");
    return 0; // n.b. this is an error value
  }

  // set the socket buffer size to 64 MB
  int sock_buf_size = 64*1024*1024;
  sock_buf_size = 128 * 1024;
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start: setting buffer size to %d\n", sock_buf_size);
  dada_udp_sock_set_buffer_size (ctx->log, ctx->sock->fd, ctx->verbose, sock_buf_size);

  // clear any packets buffered by the kernel
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start: clearing packets at socket\n");
  sock_nonblock(ctx->sock->fd);
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->packet_payload_size);
  return 0;
}

void* udpheader_read_function (udpheader_t* ctx, uint64_t* size)
{
  multilog_t * log = ctx->log;

  /* Flag to drop out of for loop */
  int quit = 0;

  /* Flag for timeout */
  int timeout_ocurred = 0;

  /* How much data has actaully been received */
  uint64_t data_received = 0;

  /* For select polling */
  struct timeval timeout;
  fd_set *rdsp = NULL;
  fd_set readset;

  /* Switch the buffers and their respective counters */
  char * tmp;
  tmp = ctx->curr_buffer;
  ctx->curr_buffer = ctx->next_buffer;
  ctx->next_buffer = ctx->zero_buffer;
  ctx->zero_buffer = tmp;

  ctx->curr_buffer_count = ctx->next_buffer_count;
  ctx->next_buffer_count = ctx->zero_buffer_count;
  ctx->zero_buffer_count = 0;

  // 0 the next buffer
#ifdef HAVE_CUDA
  if (ctx->device >= 0)
  {
    cudaError_t err;
    err = cudaStreamSynchronize(ctx->stream);
    err = cudaMemcpyAsync (ctx->zero_buffer, ctx->gpu_buffer,
                           ctx->block_size, cudaMemcpyDeviceToHost,
                           ctx->stream);
  }
  else
#endif
  {
    bzero (ctx->zero_buffer, ctx->block_size);
  }

  // Determine the sequence number boundaries for curr and next buffers
  ctx->min_sequence = ctx->expected_sequence_no;
  ctx->mid_sequence = ctx->min_sequence + ctx->packets_per_block;
  ctx->max_sequence = ctx->mid_sequence + ctx->packets_per_block;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "BLOCK [%lu - %lu - %lu]\n", ctx->min_sequence, ctx->mid_sequence, ctx->max_sequence);

  /* Determine the sequence number boundaries for curr and next buffers */
  uint64_t raw_sequence_no = 0;
  uint64_t sequence_no = 0;
  uint64_t prevnum = 0;

  /* Assume we will be able to return a full buffer */
  *size = ctx->block_size;

  // Continue to receive packets
  while (!quit) 
  {

    // If we had a packet in the socket buffer from a previous call to the buffer function
    while (!quit && !ctx->packet_in_buffer)
    {
      ctx->received = recvfrom (ctx->sock->fd, ctx->sock->buf, ctx->packet_payload_size, 0, NULL, NULL);
      if (ctx->received > 32)
      {
        ctx->packet_in_buffer = 1;
      }
      else if ((int)ctx->received == -1)
      {
        //ctx->nsleeps++;
      }
      else
      {
        multilog (log, LOG_ERR, "expected %lu bytes, received %ld\n", ctx->packet_payload_size, ctx->received);
        quit = 1;
        *size = DADA_ERROR_HARD;
        break;
      }
    }

    // If we did get a packet within the timeout, or one was in the buffer 
    if (!quit) 
    {
      // decode the packet's header and get the sequence number for this packet
      ctx->curr_sequence_no = decode_header(ctx->sock->buf) / ctx->sequence_incr;

      // if we are waiting for the first packet
      if ((ctx->expected_sequence_no == 0) && (data_received == 0))
      {
        multilog (log, LOG_INFO, "START : received packet %"PRIu64"\n", ctx->curr_sequence_no);
        ctx->expected_sequence_no = ctx->curr_sequence_no;
        ctx->min_sequence = ctx->expected_sequence_no;
        ctx->mid_sequence = ctx->min_sequence + ctx->packets_per_block;
        ctx->max_sequence = ctx->mid_sequence + ctx->packets_per_block;
        multilog (log, LOG_INFO, "START [%lu - %lu - %lu]\n", ctx->min_sequence, ctx->mid_sequence, ctx->max_sequence);
      }

      if (ctx->received != ctx->packet_payload_size)
      {
        multilog (log, LOG_ERR, "UDP packet size was incorrect (%d != %d)\n", ctx->received, ctx->packet_payload_size);
        *size = DADA_ERROR_HARD;
        break;
      }

      // Now try to slot the packet into the appropraite buffer */
      data_received += ctx->packet_data_size;

      /* Increment statistics */
      ctx->packets->received++;
      ctx->packets->received_per_sec++;
      ctx->bytes->received += BPSR_UDP_DATASIZE_BYTES;
      ctx->bytes->received_per_sec += BPSR_UDP_DATASIZE_BYTES;

      // assume the packet is processed
      ctx->packet_in_buffer = 0;

      // If the packet belongs in the curr_buffer */
      if ((ctx->curr_sequence_no >= ctx->min_sequence) &&
          (ctx->curr_sequence_no <  ctx->mid_sequence))
      {
        uint64_t buf_offset = (ctx->curr_sequence_no - ctx->min_sequence) * ctx->packet_data_size;
        //memcpy( (ctx->curr_buffer)+buf_offset, 
        //        (ctx->sock->buf)+ctx->packet_header_size,
        //        ctx->packet_data_size);
        ctx->curr_buffer_count++;
      }
      else if ((ctx->curr_sequence_no >= ctx->mid_sequence) &&
                 (ctx->curr_sequence_no <  ctx->max_sequence))
      {
        uint64_t buf_offset = (ctx->curr_sequence_no - ctx->mid_sequence) * ctx->packet_data_size;
        //memcpy( (ctx->curr_buffer)+buf_offset,
        //        (ctx->sock->buf)+ctx->packet_header_size,
        //        ctx->packet_data_size);
        ctx->next_buffer_count++;
      }
      // If this packet has arrived too late, it has already missed out
      else if (ctx->curr_sequence_no < ctx->min_sequence)
      {
        multilog (log, LOG_WARNING, "Packet arrived too soon, %"PRIu64" < %"PRIu64"\n",
                  ctx->curr_sequence_no, ctx->min_sequence);

      }
      // If a packet has arrived too soon, then we give up trying to fill the 
      // curr_buffer and return what we do have 
      else if (ctx->curr_sequence_no >= ctx->max_sequence)
      {
        float curr_percent = ((float) ctx->curr_buffer_count / (float) ctx->packets_per_block)*100;
        float next_percent = ((float) ctx->next_buffer_count / (float) ctx->packets_per_block)*100;
        multilog (log, LOG_WARNING, "%"PRIu64" > %"PRIu64"\n",ctx->curr_sequence_no,ctx->max_sequence);
        multilog (log, LOG_WARNING, "Not keeping up. curr_buffer %5.2f%, next_buffer %5.2f%\n",
                                    curr_percent, next_percent);
        ctx->packet_in_buffer = 1;
        quit = 1;
      }
      else
      {
        fprintf (stderr,"Sequence number invalid\n");
      }

      // If we have filled the current buffer, then we can stop
      if (ctx->curr_buffer_count == ctx->packets_per_block)
      {
        quit = 1;
      }
      else
      {
        assert(ctx->curr_buffer_count < ctx->packets_per_block);
      }

      // If the next buffer is at least half full
      if (ctx->next_buffer_count > (ctx->packets_per_block / 2))
      {
        float curr_percent = ((float) ctx->curr_buffer_count / (float) ctx->packets_per_block)*100;
        float next_percent = ((float) ctx->next_buffer_count / (float) ctx->packets_per_block)*100;

        multilog(log, LOG_WARNING, "Bailing curr_buf %5.2f%, next_buffer %5.2f%\n",curr_percent,next_percent); 
        quit = 1;
      }
    }
  } 

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "block capacity: curr=%"PRIu64" next=%"PRIu64" size=%"PRIu64"\n",
              ctx->curr_buffer_count, ctx->next_buffer_count, ctx->packets_per_block);

  // If we have received a packet during this function call
  if (data_received)
  {
    // If we have not received all the packets we expected
    if ((ctx->curr_buffer_count < ctx->packets_per_block) && (!timeout_ocurred))
    {
      multilog (log, LOG_WARNING, "Dropped %"PRIu64" packets\n",
               (ctx->packets_per_block - ctx->curr_buffer_count));

      ctx->packets->dropped += (ctx->packets_per_block - ctx->curr_buffer_count);
      ctx->packets->dropped_per_sec += (ctx->packets_per_block - ctx->curr_buffer_count);
      ctx->bytes->dropped += (ctx->packet_data_size * (ctx->packets_per_block - ctx->curr_buffer_count));
      ctx->bytes->dropped_per_sec += (ctx->packet_data_size * (ctx->packets_per_block - ctx->curr_buffer_count));
    }

    // If the timeout ocurred, this is most likely due to end of data
    if (timeout_ocurred)
    {
      *size = ctx->curr_buffer_count * ctx->packet_data_size;
      multilog (log, LOG_WARNING, "Suspected EOD received, returning "
                     "%"PRIi64" bytes\n",*size);
    }

    ctx->expected_sequence_no += ctx->packets_per_block;
  }
  else
    *size = 0;

  ctx->packets->received_per_sec = 0;
  ctx->packets->dropped_per_sec = 0;
  ctx->bytes->received_per_sec = 0;
  ctx->bytes->dropped_per_sec = 0;

  assert(ctx->curr_buffer != 0);
  return (void *) ctx->curr_buffer;

}

/*
 * Close the udp socket and file
 */

int udpheader_stop_function (udpheader_t* ctx)
{
  // print the packets capture performance
  float percent_dropped = 0;
  if (ctx->expected_sequence_no)
    percent_dropped = (float) ((double) ctx->packets->dropped / (double)ctx->expected_sequence_no)  ;

  fprintf(stderr, "Packets dropped %"PRIu64" / %"PRIu64" = %10.8f %%\n",
          ctx->packets->dropped, ctx->expected_sequence_no, 
          percent_dropped);
  
  if (ctx->sock)
    bpsr_free_sock (ctx->sock);
  ctx->sock = 0;

  if (ctx->packets)
    free (ctx->packets);
  ctx->packets = 0;

  if (ctx->bytes)
    free (ctx->bytes);
  ctx->bytes = 0;

  free(ctx->curr_buffer);
  free(ctx->next_buffer);

  return 0;

}


int main (int argc, char **argv)
{

  /* Interface on which to listen for udp packets */
  char * interface = "any";

  /* port on which to listen for incoming connections */
  int port = BPSR_DEFAULT_UDPDB_PORT;

  /* accumulation length of the iBob board */
  int acc_len = BPSR_DEFAULT_ACC_LEN;

  /* Flag set in verbose mode */
  char verbose = 0;

  int arg = 0;

  /* actual struct with info */
  udpheader_t udpheader;

  /* Pointer to array of "read" data */
  char *src;

  // full vs dual pol modes
  char cross_pol = 0;

#ifdef HAVE_CUDA
  int device = -1;
#endif

#ifdef HAVE_CUDA
  while ((arg=getopt(argc,argv,"d:i:p:a:vVxh")) != -1) {
#else
  while ((arg=getopt(argc,argv,":i:p:a:vVxh")) != -1) {
#endif
    switch (arg) {

#ifdef HAVE_CUDA
    case 'd':
      device = atoi(optarg);
      break;
#endif

    case 'i':
      if (optarg)
        interface = optarg;
      break;

    case 'p':
      port = atoi (optarg);
      break;

    case 'a':
      acc_len = atoi (optarg);
      break;

    case 'v':
      verbose=1;
      break;

    case 'V':
      verbose=2;
      break;

    case 'x':
      cross_pol = 1;
      break;

    case 'h':
      usage();
      return 0;
      
    default:
      usage ();
      return 0;
      
    }
  }

  assert ((BPSR_UDP_DATASIZE_BYTES + BPSR_UDP_COUNTER_BYTES) == BPSR_UDP_PAYLOAD_BYTES);

  multilog_t* log = multilog_open ("bpsr_udpheader", 0);
  multilog_add (log, stderr);
  multilog_serve (log, DADA_DEFAULT_PWC_LOG);

  udpheader.log = log;

  /* Setup context information */
  udpheader.verbose = verbose;
  udpheader.port = port;
  udpheader.interface = strdup(interface);
  udpheader.acc_len = acc_len;
  udpheader.packets_per_block = BPSR_NUM_UDP_PACKETS;
#ifdef HAVE_CUDA
  udpheader.device = device;
#endif
  if (cross_pol)
  {
    udpheader.packet_data_size    = BPSR_UDP_4POL_DATASIZE_BYTES;
    udpheader.packet_payload_size = BPSR_UDP_4POL_PAYLOAD_BYTES;
    udpheader.packet_header_size  = BPSR_UDP_4POL_HEADER_BYTES;
  }
  else
  {
    udpheader.packet_data_size    = BPSR_UDP_DATASIZE_BYTES;
    udpheader.packet_payload_size = BPSR_UDP_PAYLOAD_BYTES;
    udpheader.packet_header_size  = BPSR_UDP_COUNTER_BYTES;
  }

  // allocate memory 
  udpheader_init (&udpheader);

  udpheader.received = 0;
  udpheader.prev_time = time(0);
  udpheader.current_time = udpheader.prev_time;
  time_t utc = udpheader_start_function(&udpheader,0);

  if (utc == -1 ) {
    fprintf(stderr,"Error: udpheader_start_function failed\n");
    return EXIT_FAILURE;
  }

  int quit = 0;

  while (!quit) 
  {
    uint64_t bsize = udpheader.block_size;

    // TODO Add a quit control to the read function
    src = (char *) udpheader_read_function(&udpheader, &bsize);

    /* Quit if we dont get a packet for at least 1 second whilst recording */
    //if ((bsize <= 0) && (udpheader.state == RECORDING)) 
    //  quit = 1;

    if (udpheader.verbose == 2)
      fprintf(stdout,"udpheader_read_function: read %"PRIu64" bytes\n", bsize);
  }    

  if ( udpheader_stop_function(&udpheader) != 0)
    fprintf(stderr, "Error stopping acquisition");

  return EXIT_SUCCESS;

}


