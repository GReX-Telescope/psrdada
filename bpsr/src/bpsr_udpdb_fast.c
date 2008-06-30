/*
 * bpsr_udpdb
 *
 * 
 *
 */

#define ULTRA_PERFORMANCE 1

#include "bpsr_def.h"
#include "bpsr_udpdb_fast.h"
#include <math.h>

void usage()
{
  fprintf (stdout,
	   "bpsr_udpdb_fast [options]\n"
     " -h             print help text\n"
	   " -i interface   ip address to accept udp packets from [default %s]\n"
     " -a acc_len     accumulation length of the iBob board [default %d]\n"
	   " -p             port on which to listen [default %d]\n"
	   " -d             run as daemon\n"
     " -m mode        1 for independent mode, 2 for controlled mode\n"
     " -v             verbose messages\n"
     " -1             one time only mode, exit after EOD is written\n"
     " -H filename    ascii header information in file\n"
     " -c             don't verify udp headers against header block\n"
     " -S num         file size in bytes\n", 
     BPSR_UDP_INTERFACE, BPSR_DEFAULT_ACC_LEN, BPSR_DEFAULT_UDPDB_PORT);
}


/* Determine if the header is valid. Returns 1 if valid, 0 otherwise */
int udpdb_header_valid_function (dada_pwc_main_t* pwcm) {

  int utc_size = 1024;
  char utc_buffer[utc_size];
  int resolution;
  int valid = 1;

  /* Check if the UTC_START is set in the header*/
  if (ascii_header_get (pwcm->header, "UTC_START", "%s", utc_buffer) < 0) {
    valid = 0;
  }

  /* Check whether the UTC_START is set to UNKNOWN */
  if (strcmp(utc_buffer,"UNKNOWN") == 0)
    valid = 0;

  return valid;

}


time_t udpdb_start_function (dada_pwc_main_t* pwcm, time_t start_utc)
{

  /* get our context, contains all required params */
  udpdb_t* udpdb = (udpdb_t*)pwcm->context;
  
  multilog_t* log = pwcm->log;

  /* Initialise variables */
  udpdb->packets->received = 0;
  udpdb->packets->dropped = 0;
  udpdb->packets->received_per_sec = 0;
  udpdb->packets->dropped_per_sec = 0;

  udpdb->bytes->received = 0;
  udpdb->bytes->dropped = 0;
  udpdb->bytes->received_per_sec = 0;
  udpdb->bytes->dropped_per_sec = 0;

  udpdb->error_seconds = 10;
  udpdb->packet_in_buffer = 0;
  udpdb->prev_time = time(0);
  udpdb->current_time = udpdb->prev_time;
  udpdb->curr_buffer_count = 0;
  udpdb->next_buffer_count = 0;

  udpdb->sequence_incr = 512 * udpdb->acc_len;
  udpdb->spectra_per_second = (BPSR_IBOB_CLOCK * 1000000 / udpdb->acc_len ) / BPSR_IBOB_NCHANNELS;
  udpdb->bytes_per_second = udpdb->spectra_per_second * BPSR_UDP_DATASIZE_BYTES;
  
  /* Create a udp socket */
  udpdb->fd = bpsr_create_udp_socket(log, udpdb->interface, udpdb->port, udpdb->verbose);
  if (udpdb->fd < 0) {
    multilog (log, LOG_ERR, "Error, Failed to create udp socket\n");
    return 0; // n.b. this is an error value 
  }

  /* Check the whether the requried header fields have been set via the 
   * control interface. This is required when receiving data from the DFB3
   * board, but not required for data transfer */

  /* Set the current machines name in the header block as RECV_HOST */
  char myhostname[HOST_NAME_MAX] = "unknown";;
  gethostname(myhostname,HOST_NAME_MAX); 
  ascii_header_set (pwcm->header, "RECV_HOST", "%s", myhostname);
  
  /* setup the data buffer to be 32 MB */
  udpdb->datasize = BPSR_UDP_DATASIZE_BYTES * BPSR_NUM_UDP_PACKETS;
  udpdb->curr_buffer = (char *) malloc(sizeof(char) * udpdb->datasize);
  assert(udpdb->curr_buffer != 0);
  udpdb->next_buffer = (char *) malloc(sizeof(char) * udpdb->datasize);
  assert(udpdb->next_buffer != 0);


  /* 0 both the curr and next buffers */
  char zerodchar = 'c';
  memset(&zerodchar,0,sizeof(zerodchar));
  memset(udpdb->curr_buffer,zerodchar,udpdb->datasize);
  memset(udpdb->next_buffer,zerodchar,udpdb->datasize);

  /* Setup the socket buffer for receiveing UDP packets */
  udpdb->socket_buffer= (char *) malloc(sizeof(char) * BPSR_UDP_PAYLOAD_BYTES);
  assert(udpdb->socket_buffer != 0);

  /* setup the expected sequence no to the initial value */
  udpdb->expected_sequence_no = 0;

  /* Since udpdb cannot know the time sample of the first 
   * packet we always return 0 */
  return 0;
}

void* udpdb_buffer_function (dada_pwc_main_t* pwcm, uint64_t* size)
{
  
  /* get our context, contains all required params */
  udpdb_t* udpdb = (udpdb_t*)pwcm->context;

  /* logger */
  multilog_t* log = pwcm->log;

  // A char with all bits set to 0
  // char zerodchar = 'c'; 
  // memset(&zerodchar,0,sizeof(zerodchar));

  /* Flag to drop out of for loop */
  int quit = 0;

  /* Flag for timeout */
  int timeout_ocurred = 0;

  /* How much data has actaully been received */
  uint64_t data_received = 0;

  /* Sequence number of current packet */
  //uint64_t sequence_no = 0;
  //uint64_t raw_sequence_no = 0;
  int ignore_packet = 0;

  /* For select polling */
  struct timeval timeout;
  fd_set *rdsp = NULL;
  fd_set readset;

  int i = 0;
  int k = 0;

  /* Switch the next and current buffers and their respective counters */
  char *tmp;
  tmp = udpdb->curr_buffer;
  udpdb->curr_buffer = udpdb->next_buffer;
  udpdb->next_buffer = tmp;
  udpdb->curr_buffer_count = udpdb->next_buffer_count;
  udpdb->next_buffer_count = 0;


  /* 0 the next buffer */
  // memset(udpdb->next_buffer, zerodchar, udpdb->datasize);

  /* Determine the sequence number boundaries for curr and next buffers */
  udpdb->min_sequence = udpdb->expected_sequence_no;
  udpdb->mid_sequence = udpdb->min_sequence + BPSR_NUM_UDP_PACKETS;
  udpdb->max_sequence = udpdb->mid_sequence + BPSR_NUM_UDP_PACKETS;

#ifdef _DEBUG
  multilog(log, LOG_INFO, "[%"PRIu64" <-> %"PRIu64" <-> %"PRIu64"]\n",
           udpdb->min_sequence,udpdb->mid_sequence,udpdb->max_sequence);
#endif

  /* Assume we will be able to return a full buffer */
  *size = udpdb->datasize;
  int64_t max_ignore = udpdb->datasize;

  /* Continue to receive packets */
  while (!quit) {

    /* If we had a packet in the socket_buffer a previous call to the 
     * buffer function*/
    if (udpdb->packet_in_buffer) {

      udpdb->packet_in_buffer = 0;
                                                                                
    /* Else try to get a fresh packet */
    } else {
      
      /* 0.1 second timeout for select() */            
      timeout.tv_sec=0;
      timeout.tv_usec=100000;

      FD_ZERO (&readset);
      FD_SET (udpdb->fd, &readset);
      rdsp = &readset;

      if ( select((udpdb->fd+1),rdsp,NULL,NULL,&timeout) == 0 ) {
  
        if ((pwcm->pwc->state == dada_pwc_recording) &&
            (udpdb->expected_sequence_no != 0)) {
          multilog (log, LOG_WARNING, "UDP packet timeout: no packet "
                                      "received for 1 second\n");
        }
        quit = 1;
        udpdb->received = 0;
        timeout_ocurred = 1;

      } else {

        /* Get a packet from the socket */
        udpdb->received = sock_recv (udpdb->fd, udpdb->socket_buffer, BPSR_UDP_PAYLOAD_BYTES, 0);
        ignore_packet = 0;

      }
    }

    /* If we did get a packet within the timeout, or one was in the buffer */
    if (!quit) {

      /* Decode the packets apsr specific header */
      udpdb->curr_sequence_no = decode_header(udpdb->socket_buffer) / udpdb->sequence_incr;

      /* If we are waiting for the "first" packet */
      if (udpdb->expected_sequence_no == 0) {
  
        if (udpdb->curr_sequence_no < (10 * udpdb->spectra_per_second)) {

          multilog (log, LOG_INFO, "Start detected on packet %"PRIu64"\n",udpdb->curr_sequence_no);
          udpdb->expected_sequence_no = udpdb->curr_sequence_no + 1;
          fprintf(stderr, "Buffer size = %"PRIu64" bytes\n", udpdb->datasize);

        } else {
          ignore_packet = 1;
        }

      }

      /* If we are still waiting dor the start of data */
      if (ignore_packet) {

        max_ignore -= BPSR_UDP_PAYLOAD_BYTES;
        if (max_ignore < 0) 
          quit = 1;

      } else {

        /* Now try to slot the packet into the appropraite buffer */
        data_received += BPSR_UDP_DATASIZE_BYTES;

        /* Increment statistics */
        //udpdb->packets->received++;
        //udpdb->packets->received_per_sec++;
        //udpdb->bytes->received += BPSR_UDP_DATASIZE_BYTES;
        //udpdb->bytes->received_per_sec += BPSR_UDP_DATASIZE_BYTES;

        /* If the packet belongs in the curr_buffer */
        if ((udpdb->curr_sequence_no >= udpdb->min_sequence) && 
            (udpdb->curr_sequence_no <  udpdb->mid_sequence)) {

          memcpy( (udpdb->curr_buffer)+((udpdb->curr_sequence_no - udpdb->min_sequence) * BPSR_UDP_DATASIZE_BYTES), 
                  (udpdb->socket_buffer)+BPSR_UDP_COUNTER_BYTES,
                  BPSR_UDP_DATASIZE_BYTES);

          udpdb->curr_buffer_count++;

        } else if ((udpdb->curr_sequence_no >= udpdb->mid_sequence) && 
                   (udpdb->curr_sequence_no <  udpdb->max_sequence)) {

          memcpy( (udpdb->curr_buffer)+((udpdb->curr_sequence_no - udpdb->mid_sequence) * BPSR_UDP_DATASIZE_BYTES), 
                  (udpdb->socket_buffer)+BPSR_UDP_COUNTER_BYTES,
                  BPSR_UDP_DATASIZE_BYTES);

          udpdb->next_buffer_count++;

        /* If a packet has arrived too soon, then we give up trying to fill the 
           curr_buffer and return what we do have */
        } else if (udpdb->curr_sequence_no >= udpdb->max_sequence) {

          float curr_percent = ((float) udpdb->curr_buffer_count / (float) BPSR_NUM_UDP_PACKETS)*100;
          float next_percent = ((float) udpdb->next_buffer_count / (float) BPSR_NUM_UDP_PACKETS)*100;
  
          multilog (log, LOG_WARNING, "Not keeping up. curr_buffer %5.2f%, next_buffer %5.2f%\n",
                                      curr_percent, next_percent);

          udpdb->packet_in_buffer = 1;
          quit = 1;

        } else {
          fprintf (stderr,"Should never happen\n");
        }

        /* If we have filled the current buffer, then we can stop */
        if ((udpdb->curr_buffer_count == BPSR_NUM_UDP_PACKETS) ||
           (udpdb->next_buffer_count > (BPSR_NUM_UDP_PACKETS / 2))) {
          quit = 1;
        }

      }
    } 
  }

   /*multilog (log, LOG_INFO, "curr: %"PRIu64", next: %"PRIu64", capacity: %"PRIu64"\n",udpdb->curr_buffer_count, udpdb->next_buffer_count, BPSR_NUM_UDP_PACKETS); */

  /* If we have received a packet during this function call */
  if (data_received) {

    /* If the timeout ocurred, this is most likely due to end of data */
    if (timeout_ocurred) 
      *size = udpdb->curr_buffer_count * BPSR_UDP_DATASIZE_BYTES;

    udpdb->expected_sequence_no += BPSR_NUM_UDP_PACKETS;

  } else {

    /* If we have received no data, then return a size of 0 */
    *size = 0;

  }

  return (void *) udpdb->curr_buffer;

}


int udpdb_stop_function (dada_pwc_main_t* pwcm)
{

  /* get our context, contains all required params */
  udpdb_t* udpdb = (udpdb_t*)pwcm->context;
  int verbose = udpdb->verbose; /* saves some typing */

  float percent_dropped = 0;
  if (udpdb->expected_sequence_no) {
    percent_dropped = (float) ((double)udpdb->packets->dropped / (double)udpdb->expected_sequence_no) * 100;
  }

  multilog(pwcm->log, LOG_INFO, "packets dropped %"PRIu64" / %"PRIu64" = %10.8f %\n",
           udpdb->packets->dropped, udpdb->expected_sequence_no, percent_dropped);
  
  if (verbose) fprintf(stderr,"Stopping udp transfer\n");

  close(udpdb->fd);
  free(udpdb->curr_buffer);
  free(udpdb->next_buffer);
  free(udpdb->socket_buffer);

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
  char * interface = BPSR_UDP_INTERFACE;

  /* port on which to listen for udp packets */
  int port = BPSR_DEFAULT_UDPDB_PORT;

  /* accumulation length of the iBob board */
  int acc_len = BPSR_DEFAULT_ACC_LEN; 

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

  /* actual struct with info */
  udpdb_t udpdb;

  /* the filename from which the header will be read */
  char* header_file = 0;
  char* header_buf = 0;

  unsigned buffer_size = 64;
  static char* buffer = 0;
  char *src;

  while ((arg=getopt(argc,argv,"di:p:a:vm:S:H:n:1h")) != -1) {
    switch (arg) {
      
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

    case 'a':
      acc_len = atoi (optarg);
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

  multilog_serve (log, DADA_DEFAULT_PWC_LOG);
  
  if (verbose) fprintf (stderr, "Creating dada pwc main\n");
  pwcm = dada_pwc_main_create();

  /* Setup the global pointer to point to pwcm for the signal handler to
   * reference */

  pwcm->log = log;
  pwcm->context = &udpdb;
  pwcm->start_function  = udpdb_start_function;
  pwcm->buffer_function = udpdb_buffer_function;
  pwcm->stop_function   = udpdb_stop_function;
  pwcm->header_valid_function  = udpdb_header_valid_function;
  pwcm->verbose = verbose;

  /* init iBob config */
  udpdb.verbose = verbose;
  udpdb.interface = strdup(interface);
  udpdb.port = port;
  udpdb.acc_len = acc_len;

  /* init stats structs */
  stats_t packets = {0,0,0,0};
  stats_t bytes = {0,0,0,0};

  udpdb.packets = &packets;
  udpdb.bytes = &bytes;

  udpdb.packet_in_buffer = 0;
  udpdb.received = 0;
  udpdb.prev_time = time(0);
  udpdb.current_time = udpdb.prev_time;
  udpdb.error_seconds = 10;
  udpdb.statslog = multilog_open ("bpsr_udpdb_stats", 0);
  multilog_serve (udpdb.statslog, BPSR_DEFAULT_UDPDB_STATS);
    
  /* Connect to shared memory */
  hdu = dada_hdu_create (pwcm->log);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_write (hdu) < 0)
    return EXIT_FAILURE;

  pwcm->data_block = hdu->data_block;
  pwcm->header_block = hdu->header_block;


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

      /* get our context, contains all required params */
      udpdb_t* udpdb = (udpdb_t*)pwcm->context;

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

    time_t utc = udpdb_start_function(pwcm,0);

    if (utc == -1 ) {
      multilog(pwcm->log, LOG_ERR, "Could not run start function\n");
    }

    strftime (buffer, buffer_size, DADA_TIMESTR, gmtime(&utc));

    /* write UTC_START to the header */
    if (ascii_header_set (header_buf, "UTC_START", "%s", buffer) < 0) {
      multilog (pwcm->log, LOG_ERR, "failed ascii_header_set UTC_START\n");
      return -1;
    }

    /* donot set header parameters anymore - acqn. doesn't start */
    if (ipcbuf_mark_filled (hdu->header_block, header_size) < 0)  {
      multilog (pwcm->log, LOG_ERR, "Could not mark filled header block\n");
      return EXIT_FAILURE;
    }

    while (1) {

      uint64_t bsize = udpdb.datasize;
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

  return EXIT_SUCCESS;

}

