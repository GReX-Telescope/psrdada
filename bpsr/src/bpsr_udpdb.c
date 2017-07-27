/***************************************************************************
 *  
 *    Copyright (C) 2010 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

/***************************************************************************
 *
 * bpsr_udpdb
 * 
 * Primary Write Client for BPSR backend
 *
 ****************************************************************************/

//#define _DEBUG 1

#include <math.h>

#include "bpsr_def.h"
#include "bpsr_udpdb.h"
#include "sock.h"

void usage()
{
  fprintf (stdout,
     "bpsr_udpdb [options]\n"
     " -c port        port to received pwcc commands on [default %d]\n"
     " -d             run as daemon\n"
     " -h             print help text\n"
     " -H filename    ascii header information in file\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
     " -k             hexidecimal shared memor key [default %x]\n"
     " -l port        multilog port to write logging output to [default %d]\n"
     " -m mode        1 for independent mode, 2 for controlled mode\n"
     " -p port        port on which to listen [default %d]\n"
     " -S num         file size in bytes\n"
     " -v             verbose messages\n"
     " -x             assume cross pol mode\n"
     " -1             one time only mode, exit after EOD is written\n",
     BPSR_DEFAULT_PWC_PORT, DADA_DEFAULT_BLOCK_KEY,
     BPSR_DEFAULT_PWC_LOGPORT, BPSR_DEFAULT_UDPDB_PORT);
}

/* allocate required memory structures */
int udpdb_init (udpdb_t * ctx)
{
  // initialize a socket large enough for vanilla or cross pol BPSR packets
  ctx->sock    = bpsr_init_sock();
  ctx->packets = init_stats_t();
  ctx->bytes   = init_stats_t();

  // setup the data buffer to be 32 MB
  ctx->curr_buffer = (char *) malloc(sizeof(char) * ctx->block_size);
  assert(ctx->curr_buffer != 0);
  ctx->next_buffer = (char *) malloc(sizeof(char) * ctx->block_size);
  assert(ctx->next_buffer != 0);
}

/* release memory */
int udpdb_destroy (udpdb_t * ctx)
{
  if (ctx->sock)
    bpsr_free_sock (ctx->sock);
  ctx->sock = 0;

  if (ctx->packets)
    free (ctx->packets);
  ctx->packets = 0;

  if (ctx->bytes)
    free (ctx->bytes);
  ctx->bytes = 0;
}


/* determine if the header is valid. Returns 1 if valid, 0 otherwise */
int udpdb_header_valid_function (dada_pwc_main_t* pwcm) 
{
  int utc_size = 64;
  char utc_buffer[utc_size];
  int valid = 1;

  // check if the UTC_START is set in the header
  if (ascii_header_get (pwcm->header, "UTC_START", "%s", utc_buffer) < 0) {
    valid = 0;
  }

  // check whether the UTC_START is set to UNKNOWN
  if (strcmp(utc_buffer,"UNKNOWN") == 0)
    valid = 0;

#ifdef _DEBUG
  multilog(pwcm->log, LOG_INFO, "Checking if header is valid: %d\n", valid);
#endif

  return valid;
}

/* error function */
int udpdb_error_function (dada_pwc_main_t* pwcm) {

  udpdb_t * ctx = (udpdb_t *) pwcm->context;

  /* If UTC_START has been received, the buffer function 
   * should be returning data */
  if (udpdb_header_valid_function (pwcm) == 1) 
  {
    ctx->error_seconds--;
    //multilog (pwcm->log, LOG_WARNING, "Error Function has %"PRIu64" error "
    //          "seconds remaining\n",ctx->error_seconds);

    if (ctx->error_seconds <= 0)
      return 2;
    else
      return 1;

  } else {
    return 0;
  }

}



time_t udpdb_start_function (dada_pwc_main_t* pwcm, time_t start_utc)
{
  // get our context, contains all required params
  udpdb_t * ctx = (udpdb_t *) pwcm->context;
  
  multilog_t * log = pwcm->log;

  // initialise / reset variables
  reset_stats_t (ctx->packets);
  reset_stats_t (ctx->bytes);

  ctx->error_seconds = 5;
  ctx->packet_in_buffer = 0;
  ctx->prev_time = time(0);
  ctx->current_time = ctx->prev_time;
  ctx->curr_buffer_count = 0;
  ctx->next_buffer_count = 0;
  
  // create the UDP socket
  ctx->sock->fd = dada_udp_sock_in(ctx->log, ctx->interface, ctx->port, ctx->verbose);
  if (ctx->sock->fd < 0)
  {
    multilog (log, LOG_ERR, "failed to create udp socket\n");
    return 0; // n.b. this is an error value
  }

  // set the socket buffer size to 64 MB
  int sock_buf_size = 64*1024*1024;
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start: setting buffer size to %d\n", sock_buf_size);
  dada_udp_sock_set_buffer_size (ctx->log, ctx->sock->fd, ctx->verbose, sock_buf_size);

  // clear any packets buffered by the kernel
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start: clearing packets at socket\n");
  sock_nonblock(ctx->sock->fd);
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->packet_payload_size);
  sock_block(ctx->sock->fd);

  // get the accumulation length from the header
  if (ascii_header_get (pwcm->header, "ACC_LEN", "%d", &(ctx->acc_len)) != 1)
  {
    multilog (log, LOG_WARNING, "Warning. ACC_LEN not set in header, Using "
              "%d as default\n", BPSR_DEFAULT_ACC_LEN);
    ascii_header_set (pwcm->header, "ACC_LEN", "%d", BPSR_DEFAULT_ACC_LEN);
    ctx->acc_len = BPSR_DEFAULT_ACC_LEN;
  }

  ctx->sequence_incr = 512 * ctx->acc_len;
  ctx->spectra_per_second = (BPSR_IBOB_CLOCK * 1000000 / ctx->acc_len ) / BPSR_IBOB_NCHANNELS;
  ctx->bytes_per_second = ctx->spectra_per_second * ctx->packet_data_size;

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start: bytes_per_second=%"PRIu64"\n", ctx->bytes_per_second);

  /* Check the whether the requried header fields have been set via the 
   * control interface. This is required when receiving data from the DFB3
   * board, but not required for data transfer */

  /* Set the current machines name in the header block as RECV_HOST */
  char myhostname[HOST_NAME_MAX] = "unknown";;
  gethostname(myhostname,HOST_NAME_MAX); 
  ascii_header_set (pwcm->header, "RECV_HOST", "%s", myhostname);

  // 0 both the curr and next buffers
  char zerodchar = 'c';
  memset(&zerodchar,0,sizeof(zerodchar));
  memset(ctx->curr_buffer,zerodchar,ctx->block_size);
  memset(ctx->next_buffer,zerodchar,ctx->block_size);

  // setup the expected sequence no to the initial value
  ctx->expected_sequence_no = 0;
  ctx->prev_seq = 0;

  time_t utc = 0;

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start: returning UTC=%d\n", utc);
  return utc;
}

void* udpdb_buffer_function (dada_pwc_main_t* pwcm, int64_t* size)
{
  
  udpdb_t * ctx = (udpdb_t *) pwcm->context;

  multilog_t * log = pwcm->log;

  // A char with all bits set to 0
  char zerodchar = 'c'; 
  memset(&zerodchar,0,sizeof(zerodchar));

  // Flag to drop out of for loop
  int quit = 0;

  // Flag for timeout
  int timeout_ocurred = 0;

  // How much data has actaully been received
  uint64_t data_received = 0;

  // Sequence number of current packet
  int ignore_packet = 0;

  // For select polling
  struct timeval timeout;
  fd_set *rdsp = NULL;
  fd_set readset;

  uint64_t prevnum = ctx->curr_sequence_no;

  // switch the next and current buffers
  char * tmp = ctx->curr_buffer;
  ctx->curr_buffer = ctx->next_buffer;
  ctx->next_buffer = tmp;
  
  // switch the buffer counters
  ctx->curr_buffer_count = ctx->next_buffer_count;
  ctx->next_buffer_count = 0;

  // 0 the next buffer 
  memset(ctx->next_buffer, zerodchar, ctx->block_size);

  // Determine the sequence number boundaries for curr and next buffers
  ctx->min_sequence = ctx->expected_sequence_no;
  ctx->mid_sequence = ctx->min_sequence + ctx->packets_per_block;
  ctx->max_sequence = ctx->mid_sequence + ctx->packets_per_block;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "seq range [%"PRIu64" - %"PRIu64" - %"PRIu64"] datasize=%"PRIu64"\n",
             ctx->min_sequence, ctx->mid_sequence, ctx->max_sequence, ctx->block_size);

  // Assume we will be able to return a full buffer
  *size = (int64_t) ctx->block_size;
  int64_t max_ignore = ctx->block_size;

  // Continue to receive packets
  while (!quit) 
  {
    // If we had a packet in the socket buffer from a previous call to the buffer function
    if (ctx->packet_in_buffer) 
    {
      ctx->packet_in_buffer = 0;
    } 
    // Else try to get a fresh packet
    else 
    {
      // 1.0 second timeout for select()
      timeout.tv_sec=0;
      timeout.tv_usec=1000000;

      FD_ZERO (&readset);
      FD_SET (ctx->sock->fd, &readset);
      rdsp = &readset;

      if ( select((ctx->sock->fd+1),rdsp,NULL,NULL,&timeout) == 0 ) 
      {
        if ((pwcm->pwc) && (pwcm->pwc->state == dada_pwc_recording))
        {
          multilog (log, LOG_WARNING, "UDP packet timeout: no packet "
                                      "received for 1 second\n");
        }
        quit = 1;
        ctx->received = 0;
        timeout_ocurred = 1;
        if (ctx->verbose)
          multilog (log, LOG_INFO, "buffer: timeout received!\n");
      } 
      else 
      {
        // Get a packet from the socket
        ctx->received = recvfrom (ctx->sock->fd, ctx->sock->buf, ctx->packet_payload_size, 0, NULL, NULL);
        ignore_packet = 0;
      }
    }

    // If we did get a packet within the timeout, or one was in the buffer
    if (!quit) {

      // Decode the packet's header
      ctx->curr_sequence_no = decode_header(ctx->sock->buf) / ctx->sequence_incr;

      // If we are waiting for the "first" packet
      if ((ctx->expected_sequence_no == 0) && (data_received == 0)) 
      {
#ifdef _DEBUG
        if ((ctx->curr_sequence_no < (ctx->prev_seq - 10)) && (ctx->prev_seq != 0)) {
          multilog(log, LOG_INFO, "packet num reset from %"PRIu64" to %"PRIu64"\n", ctx->prev_seq, ctx->curr_sequence_no);
        }
#endif
        ctx->prev_seq = ctx->curr_sequence_no;
 
        // Condition for detection of a restart
        if (ctx->curr_sequence_no < ctx->max_sequence)
        {
          multilog (log, LOG_INFO, "START: seq=%"PRIu64" raw=%"PRIu64"\n", ctx->curr_sequence_no, decode_header(ctx->sock->buf));

          if (ctx->verbose)
            multilog (log, LOG_INFO, "start on packet %"PRIu64"\n",ctx->curr_sequence_no);

          // Increment the buffer counts in case we have missed packets
          if (ctx->curr_sequence_no < ctx->mid_sequence)
            ctx->curr_buffer_count += ctx->curr_sequence_no;
          else 
          {
            ctx->curr_buffer_count += ctx->packets_per_block;
            ctx->next_buffer_count += ctx->curr_sequence_no - ctx->mid_sequence;
          }
        } 
        else 
        {
          ignore_packet = 1;
        }

        if (ctx->received != ctx->packet_payload_size) 
        {
          multilog (log, LOG_ERR, "UDP packet size was incorrect (%"PRIu64" != %d)\n", ctx->received, ctx->packet_payload_size);
          *size = DADA_ERROR_HARD;
          break;
        }
      }

      // If we are still waiting dor the start of data
      if (ignore_packet) {

        max_ignore -= ctx->packet_payload_size;
        if (max_ignore < 0) 
          quit = 1;

      } else {

        // If the packet we received was too small, pad it
        if (ctx->received < ctx->packet_payload_size) {
        
          uint64_t amount_to_pad = ctx->packet_payload_size - ctx->received;
          char * buffer_pos = (ctx->sock->buf) + ctx->received;
 
          // 0 the next buffer
          memset(buffer_pos, zerodchar, amount_to_pad);

          multilog (log, LOG_WARNING, "Short packet received, padded %"PRIu64
                                      " bytes\n", amount_to_pad);
        } 

        // If the packet we received was too long, warn about it
        if (ctx->received > ctx->packet_payload_size) 
        {
          multilog (log, LOG_WARNING, "Long packet received, truncated to %"PRIu64
                                      " bytes\n", ctx->packet_data_size);
        }

        // Now try to slot the packet into the appropraite buffer
        data_received += ctx->packet_data_size;

        // Increment statistics
        ctx->packets->received++;
        ctx->packets->received_per_sec++;
        ctx->bytes->received += ctx->packet_data_size;
        ctx->bytes->received_per_sec += ctx->packet_data_size;

        // If the packet belongs in the curr_buffer */
        if ((ctx->curr_sequence_no >= ctx->min_sequence) && 
            (ctx->curr_sequence_no <  ctx->mid_sequence)) 
        {

          uint64_t buf_offset = (ctx->curr_sequence_no - ctx->min_sequence) * ctx->packet_data_size;
          memcpy( (ctx->curr_buffer)+buf_offset, 
                  (ctx->sock->buf)+ctx->packet_header_size,
                  ctx->packet_data_size);
          ctx->curr_buffer_count++;
        } 
        else if ((ctx->curr_sequence_no >= ctx->mid_sequence) && 
                   (ctx->curr_sequence_no <  ctx->max_sequence)) 
        {
          uint64_t buf_offset = (ctx->curr_sequence_no - ctx->mid_sequence) * ctx->packet_data_size;
          memcpy( (ctx->curr_buffer)+buf_offset, 
                  (ctx->sock->buf)+ctx->packet_header_size,
                  ctx->packet_data_size);
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
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "curr: %"PRIu64", next: %"PRIu64", capacity: %"PRIu64"\n",ctx->curr_buffer_count, ctx->next_buffer_count, ctx->packets_per_block);

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

    /* If the timeout ocurred, this is most likely due to end of data */
    if (timeout_ocurred) 
    {
      *size = ctx->curr_buffer_count * ctx->packet_data_size;
      multilog (log, LOG_WARNING, "Suspected EOD received, returning "
                     "%"PRIi64" bytes\n",*size);
    }

    ctx->expected_sequence_no += ctx->packets_per_block;

  } else {

    /* If we have received no data, then return a size of 0 */
    *size = 0;

  }

  ctx->prev_time = ctx->current_time;
  ctx->current_time = time(0);
  
  if (ctx->prev_time != ctx->current_time) {

    if (ctx->verbose) {

      if (ignore_packet) {

        multilog(log, LOG_INFO, "Ignoring out of range sequence no: %"PRIu64" > %"PRIu64"\n",ctx->curr_sequence_no, (10 * ctx->spectra_per_second));

      } else {

        multilog(log, LOG_INFO, "Packets [%"PRIu64",%"PRIu64"] [%"PRIu64",%"PRIu64"]\n",
                 ctx->packets->dropped_per_sec,ctx->packets->dropped,
                 ctx->packets->received_per_sec,ctx->packets->received);

        multilog(log, LOG_INFO, "Bytes   [%"PRIu64",%"PRIu64"] [%"PRIu64",%"PRIu64"]\n",
                 ctx->bytes->dropped_per_sec,ctx->bytes->dropped,
                 ctx->bytes->received_per_sec,ctx->bytes->received);
      }

    } else {
      /*
      multilog(log, LOG_INFO, "Packet loss %5.3f % / %5.3f %, Data loss %5.3f %\n", 
               (100 * ((float) ctx->packets->dropped / (float) ctx->packets->received)), 
               (100 * ((float) ctx->packets->dropped_per_sec / (float) ctx->packets->received_per_sec)), 
               (100 * ((float) ctx->bytes->dropped / (float) ctx->bytes->received)));
       */
    }

    ctx->packets->received_per_sec = 0;
    ctx->packets->dropped_per_sec = 0;
    ctx->bytes->received_per_sec = 0;
    ctx->bytes->dropped_per_sec = 0;

  }

  assert(ctx->curr_buffer != 0);
  
  return (void *) ctx->curr_buffer;

}


int udpdb_stop_function (dada_pwc_main_t* pwcm)
{
  udpdb_t * ctx = (udpdb_t *) pwcm->context;

  float percent_dropped = 0;
  if (ctx->expected_sequence_no)
    percent_dropped = (float) ((double)ctx->packets->dropped / (double)ctx->expected_sequence_no) * 100;

  if (ctx->packets->dropped) 
  {
    multilog(pwcm->log, LOG_INFO, "packets dropped %"PRIu64" / %"PRIu64
             " = %8.6f %\n", ctx->packets->dropped, 
             ctx->expected_sequence_no, percent_dropped);
  }
  
  if (ctx->verbose) 
    fprintf(stderr, "stopping udp transfer\n");

  close(ctx->sock->fd);

  //TODO proper messaging and return values 
  return 0;

}


int main (int argc, char **argv)
{

  /* DADA Primary Write Client main loop  */
  dada_pwc_main_t* pwcm = 0;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Logger */
  multilog_t* log = 0;
  
  /* Interface on which to listen for udp packets */
  char * interface = "any";

  /* port on which to listen for udp packets */
  int port = BPSR_DEFAULT_UDPDB_PORT;

  /* pwcc command port */
  int c_port = BPSR_DEFAULT_PWC_PORT;

  /* multilog output port */
  int l_port = BPSR_DEFAULT_PWC_LOGPORT;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  int arg = 0;

  /* mode flag */
  int mode = 0; 

  /* mode flag */
  int onetime = 0; 

  /* max length of header file */
  unsigned long fSize=800000000;

  /* size of the header buffer */
  uint64_t header_size = 0;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  /* actual struct with info */
  udpdb_t udpdb;

  // packet type
  char cross_pol = 0;

  /* the filename from which the header will be read */
  char* header_file = 0;
  char* header_buf = 0;

  unsigned buffer_size = 64;
  static char* buffer = 0;
  char *src;

  while ((arg=getopt(argc,argv,"k:di:p:vm:S:H:n:1hc:l:x")) != -1) {
    switch (arg) {

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr,"bpsr_udpdb: could not parse key from %s\n",optarg);
        return -1;
      }
      break;
      
    case 'd':
      daemon = 1;
      break;

    case 'i':
      if (optarg)
        interface = optarg;
      break;

    case 'p':
      port = atoi (optarg);
      break;

    case '1':
      onetime = 1;
      break;

    case 'v':
      verbose=1;
      break;

    case 'm':
#ifdef _DEBUG
      fprintf(stderr,"Mode is %s \n",optarg);
#endif
      if (optarg) {
        mode = atoi (optarg);
        if (!(mode == 1)) {
          fprintf(stderr,"Specify a valid mode\n");
          usage();
        }
      }
      break;

    case 'S':
      if (optarg){
        fSize = atoi(optarg);
        fprintf(stderr,"File size will be %lu\n",fSize);
      }
      else usage();
      break;

    case 'H':
      if (optarg) {
        header_file = optarg;
      } else {
        fprintf(stderr,"Specify a header file\n");
        usage();
      }
      break;

    case 'c':
      if (optarg) {
        c_port = atoi(optarg);
        break;
      } else {
        usage();
        return EXIT_FAILURE;
      }

    case 'l':
      if (optarg) {
        l_port = atoi(optarg);
        break;
      } else {
        usage();
        return EXIT_FAILURE;
      }

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

  log = multilog_open ("bpsr_udpdb", 0);

  if (daemon) 
    be_a_daemon ();
  else
    multilog_add (log, stderr);

  multilog_serve (log, l_port);
  
  if (verbose) 
    fprintf (stderr, "Creating dada pwc main\n");

  pwcm = dada_pwc_main_create();

  /* Setup the global pointer to point to pwcm for the signal handler to
   * reference */

  pwcm->log = log;
  pwcm->context = &udpdb;
  pwcm->start_function  = udpdb_start_function;
  pwcm->buffer_function = udpdb_buffer_function;
  pwcm->stop_function   = udpdb_stop_function;
  pwcm->error_function  = udpdb_error_function;
  pwcm->header_valid_function  = udpdb_header_valid_function;
  pwcm->verbose = verbose;

  /* init iBob config */
  udpdb.log = log;
  udpdb.verbose = verbose;
  udpdb.interface = strdup(interface);
  udpdb.port = port;
  udpdb.acc_len = BPSR_DEFAULT_ACC_LEN;   // 25
  udpdb.mode = mode;
  udpdb.ibob_host = NULL;
  udpdb.ibob_port = 0;

  if (cross_pol)
  {
    udpdb.packet_data_size    = BPSR_UDP_4POL_DATASIZE_BYTES;
    udpdb.packet_payload_size = BPSR_UDP_4POL_PAYLOAD_BYTES;
    udpdb.packet_header_size  = BPSR_UDP_4POL_HEADER_BYTES;
  }
  else
  {
    udpdb.packet_data_size    = BPSR_UDP_DATASIZE_BYTES;
    udpdb.packet_payload_size = BPSR_UDP_PAYLOAD_BYTES;
    udpdb.packet_header_size  = BPSR_UDP_COUNTER_BYTES;
  }

  udpdb.packet_in_buffer = 0;
  udpdb.received = 0;
  udpdb.prev_time = time(0);
  udpdb.current_time = udpdb.prev_time;
  udpdb.error_seconds = 5;
    
  /* Connect to shared memory */
  hdu = dada_hdu_create (pwcm->log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_write (hdu) < 0)
    return EXIT_FAILURE;

  pwcm->data_block = hdu->data_block;
  pwcm->header_block = hdu->header_block;

  udpdb.block_size = ipcbuf_get_bufsz ((ipcbuf_t*) hdu->data_block);
  
  // ensure the data block size is a clean multiple of the packet data size
  if (udpdb.block_size % udpdb.packet_data_size)
  {
    multilog (pwcm->log, LOG_ERR, "data block size was not a clean multiple of the packet data size");
    if (dada_hdu_unlock_write (hdu) < 0)
      return EXIT_FAILURE;
    if (dada_hdu_disconnect (hdu) < 0)
      return EXIT_FAILURE;
    return EXIT_FAILURE;
  }

  udpdb.packets_per_block = udpdb.block_size / udpdb.packet_data_size;

  // allocate required memory structures now we know our block size
  udpdb_init (&udpdb);

  // We now need to setup the header information. for testing we can
  // accept local header info. In practise we will get this from
  // the control interface...

  if (mode == 1) {

    header_size = ipcbuf_get_bufsz (hdu->header_block);
    multilog (pwcm->log, LOG_INFO, "header block size = %llu\n", header_size);
    header_buf = ipcbuf_get_next_write (hdu->header_block);
    pwcm->header = header_buf;

    if (!header_buf)  {
      multilog (pwcm->log, LOG_ERR, "Could not get next header block\n");
      return EXIT_FAILURE;
    }

    // if header file is presented, use it. If not set command line attributes 
    if (header_file)  {

      if (verbose) fprintf(stderr,"read header file %s\n", header_file);
      if (fileread (header_file, header_buf, header_size) < 0)  {
        multilog (pwcm->log, LOG_ERR, "Could not read header from %s\n", 
                  header_file);
        return EXIT_FAILURE;
      }

      if (verbose) fprintf(stderr,"Retrieved header information\n");
      
    } else {

      if (verbose) fprintf(stderr,"Could not read header file\n");
      multilog(pwcm->log, LOG_ERR, "header file was not specified with -H\n");
      return EXIT_FAILURE;
    }

    // TODO Need to fill in the manual loop mode as we aren't being controlled
    
    if (!buffer)
      buffer = malloc (buffer_size);
    assert (buffer != 0);

    if (verbose) 
      fprintf(stderr, "running start function\n");
    time_t utc = udpdb_start_function(pwcm,0);

    if (utc == -1 ) {
      multilog(pwcm->log, LOG_ERR, "Could not run start function\n");
    }
    utc = time(0);
    strftime (buffer, buffer_size, DADA_TIMESTR, gmtime(&utc));

    /* write UTC_START to the header */
    if (ascii_header_set (header_buf, "UTC_START", "%s", buffer) < 0) {
      multilog (pwcm->log, LOG_ERR, "failed ascii_header_set UTC_START\n");
      return -1;
    }

    multilog (pwcm->log, LOG_INFO, "UTC_START %s written to header\n", buffer);

    /* donot set header parameters anymore - acqn. doesn't start */
    if (ipcbuf_mark_filled (hdu->header_block, header_size) < 0)  {
      multilog (pwcm->log, LOG_ERR, "Could not mark filled header block\n");
      return EXIT_FAILURE;
    }

    multilog (pwcm->log, LOG_INFO, "header marked filled\n");

    while (1) {

      int64_t bsize = udpdb.block_size;
      src = (char *) udpdb_buffer_function(pwcm, &bsize);

      /* write data to datablock */
      if (( ipcio_write(hdu->data_block, src, bsize) ) < bsize){
        multilog(pwcm->log, LOG_ERR, "Cannot write requested bytes to SHM\n");
        return EXIT_FAILURE;
      }
    }    

    if ( udpdb_stop_function(pwcm) != 0)
      fprintf(stderr, "Error stopping acquisition");

    if (dada_hdu_unlock_write (hdu) < 0)
      return EXIT_FAILURE;

    if (dada_hdu_disconnect (hdu) < 0)
      return EXIT_FAILURE;

  }
    /* we are controlled by PWC control interface */
  else {

    pwcm->header = hdu->header;

    if (verbose) fprintf (stderr, "Creating dada pwc control interface\n");
    pwcm->pwc = dada_pwc_create();

    pwcm->pwc->port = c_port;

    if (verbose) fprintf (stderr, "Creating dada server\n");
    if (dada_pwc_serve (pwcm->pwc) < 0) {
      fprintf (stderr, "dada_udpdb: could not start server\n");
      return EXIT_FAILURE;
    }
    if (verbose) fprintf (stderr, "Entering PWC main loop\n");
    if (dada_pwc_main (pwcm) < 0) {
      fprintf (stderr, "dada_udpdb: error in PWC main loop\n");
      return EXIT_FAILURE;
    }

    if (dada_hdu_unlock_write (hdu) < 0)
      return EXIT_FAILURE;
  
    if (dada_hdu_disconnect (hdu) < 0)
      return EXIT_FAILURE;

    if (verbose) fprintf (stderr, "Destroying pwc\n");
    dada_pwc_destroy (pwcm->pwc);

    if (verbose) fprintf (stderr, "Destroying pwc main\n");
    dada_pwc_main_destroy (pwcm);

  }

  udpdb_destroy(&udpdb);

  return EXIT_SUCCESS;

}


