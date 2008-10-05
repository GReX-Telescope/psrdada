/*
 * bpsr_udpdb
 *
 */


#define _DEBUG 1

#include "bpsr_def.h"
#include "bpsr_udpdb.h"

#include "sock.h"
#include <math.h>

int from_ibob(int fd, char *response); 

void usage()
{
  fprintf (stdout,
	   "bpsr_udpdb [options] ibob_host ibob_port\n"
     " -h             print help text\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
	   " -p             port on which to listen [default %d]\n"
	   " -d             run as daemon\n"
     " -m mode        1 for independent mode, 2 for controlled mode\n"
     " -v             verbose messages\n"
     " -1             one time only mode, exit after EOD is written\n"
     " -H filename    ascii header information in file\n"
     " -c             don't verify udp headers against header block\n"
     " -S num         file size in bytes\n", 
     BPSR_DEFAULT_UDPDB_PORT);
}


/* Determine if the header is valid. Returns 1 if valid, 0 otherwise */
int udpdb_header_valid_function (dada_pwc_main_t* pwcm) {

  int utc_size = 1024;
  char utc_buffer[utc_size];
  int valid = 1;

  /* Check if the UTC_START is set in the header*/
  if (ascii_header_get (pwcm->header, "UTC_START", "%s", utc_buffer) < 0) {
    valid = 0;
  }

  /* Check whether the UTC_START is set to UNKNOWN */
  if (strcmp(utc_buffer,"UNKNOWN") == 0)
    valid = 0;

  multilog(pwcm->log, LOG_INFO, "Checking if header is valid: %d\n", valid);

  return valid;

}

/*
 * Error function
 */
int udpdb_error_function (dada_pwc_main_t* pwcm) {

  udpdb_t *udpdb = (udpdb_t*)pwcm->context;

  /* If we are still waiting for that first packet, we don't
   * really have an error state :), but the buffer function
   * will return 0 bytes */
  if (udpdb->expected_sequence_no > 0) {
    //sleep(1);
    udpdb->error_seconds--;
    multilog (pwcm->log, LOG_WARNING, "Error Function has %"PRIu64" error "
              "seconds remaining\n",udpdb->error_seconds);
  }

  if (udpdb->error_seconds <= 0)
    return 2;
  else
    return 1;
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
  
  /* Create a udp socket */
  udpdb->fd = bpsr_create_udp_socket(log, udpdb->interface, udpdb->port, udpdb->verbose);
  if (udpdb->fd < 0) {
    multilog (log, LOG_ERR, "Error, Failed to create udp socket\n");
    return 0; // n.b. this is an error value 
  }

  /* Get the accumulation length from the header */
  if (ascii_header_get (pwcm->header, "ACC_LEN", "%d", &(udpdb->acc_len)) != 1) {
    multilog (log, LOG_WARNING, "Warning. ACC_LEN not set in header, Using "
              "%d as default\n", BPSR_DEFAULT_ACC_LEN);
    ascii_header_set (pwcm->header, "ACC_LEN", "%d", BPSR_DEFAULT_ACC_LEN);
    udpdb->acc_len = BPSR_DEFAULT_ACC_LEN;
  }

  udpdb->sequence_incr = 512 * udpdb->acc_len;
  udpdb->spectra_per_second = (BPSR_IBOB_CLOCK * 1000000 / udpdb->acc_len ) / BPSR_IBOB_NCHANNELS;
  udpdb->bytes_per_second = udpdb->spectra_per_second * BPSR_UDP_DATASIZE_BYTES;

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
  udpdb->prev_seq = 0;

  /* Run the script to startup the ibob */
  char command[1024];

  multilog(log, LOG_INFO, "Running new supa function!\n");
  time_t utc = set_ibob_levels(pwcm);

  //if (udpdb->mode == 1) {
    //sprintf(command, "client_ibob_level_setter.pl -a %d %s", 
    //      udpdb->acc_len, myhostname);
  //} else {
    //sprintf(command, "client_ibob_level_setter.pl -s -a %d %s", 
    //      udpdb->acc_len, myhostname);
  //}

  //if (udpdb->verbose) 

  //long int utc_unix;
  //FILE * fptr = popen(command, "r");
  //fscanf(fptr, "%ld", &utc_unix);
  //pclose(fptr);

  //time_t utc = 0;
  //if (utc_unix > 0) 
  //  utc = (time_t) utc_unix;

  multilog(log, LOG_INFO, "start function returning UTC_START = %ld\n", utc);

  return utc;
}

void* udpdb_buffer_function (dada_pwc_main_t* pwcm, uint64_t* size)
{
  
  /* get our context, contains all required params */
  udpdb_t* udpdb = (udpdb_t*)pwcm->context;

  /* logger */
  multilog_t* log = pwcm->log;

  // A char with all bits set to 0
   char zerodchar = 'c'; 
   memset(&zerodchar,0,sizeof(zerodchar));

  /* Flag to drop out of for loop */
  int quit = 0;

  /* Flag for timeout */
  int timeout_ocurred = 0;

  /* How much data has actaully been received */
  uint64_t data_received = 0;

  /* Sequence number of current packet */
  int ignore_packet = 0;

  /* For select polling */
  struct timeval timeout;
  fd_set *rdsp = NULL;
  fd_set readset;

  uint64_t prevnum = udpdb->curr_sequence_no;

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

  //multilog(log, LOG_INFO, "[%"PRIu64" <-> %"PRIu64" <-> %"PRIu64"]\n",
  //         udpdb->min_sequence,udpdb->mid_sequence,udpdb->max_sequence);

  /* Assume we will be able to return a full buffer */
  *size = udpdb->datasize;
  int64_t max_ignore = udpdb->datasize;

  //multilog (log, LOG_INFO, "Before while\n");

  /* Continue to receive packets */
  while (!quit) {

    /* If we had a packet in the socket)buffer a previous call to the 
     * buffer function*/
    if (udpdb->packet_in_buffer) {

      udpdb->packet_in_buffer = 0;
                                                                                
    /* Else try to get a fresh packet */
    } else {
      
      /* 0.1 second timeout for select() */            
      timeout.tv_sec=0;
      timeout.tv_usec=1000000;

      FD_ZERO (&readset);
      FD_SET (udpdb->fd, &readset);
      rdsp = &readset;

      if ( select((udpdb->fd+1),rdsp,NULL,NULL,&timeout) == 0 ) {
    
        //multilog(log, LOG_INFO, "Select timeout\n");
  
        /*if ((pwcm->pwc->state == dada_pwc_recording) &&
            (udpdb->expected_sequence_no != 0)) {
          multilog (log, LOG_WARNING, "UDP packet timeout: no packet "
                                      "received for 1 second\n");
        }*/
        quit = 1;
        udpdb->received = 0;
        timeout_ocurred = 1;

      } else {
        //multilog(log, LOG_INFO, "Getting packet\n");

        /* Get a packet from the socket */
        udpdb->received = sock_recv (udpdb->fd, udpdb->socket_buffer, BPSR_UDP_PAYLOAD_BYTES, 0);
        ignore_packet = 0;

      }
    }

    //multilog(log, LOG_INFO, "After packet\n");

    /* If we did get a packet within the timeout, or one was in the buffer */
    if (!quit) {

      /* Decode the packets apsr specific header */
      udpdb->curr_sequence_no = decode_header(udpdb->socket_buffer) / udpdb->sequence_incr;

      //if (udpdb->curr_sequence_no != prevnum + 1) {
      //  multilog(log, LOG_WARNING, "%"PRIu64" != %"PRIu64" !!!\n", udpdb->curr_sequence_no, (prevnum + 1));
      //}
      //prevnum = udpdb->curr_sequence_no;

      /* If we are waiting for the "first" packet */
      if ((udpdb->expected_sequence_no == 0) && (data_received == 0)) {

        if ((udpdb->curr_sequence_no < (udpdb->prev_seq - 10)) && (udpdb->prev_seq != 0)) {
          multilog(log, LOG_INFO, "packet num reset from %"PRIu64" to %"PRIu64"\n", udpdb->prev_seq, udpdb->curr_sequence_no);
        }
        udpdb->prev_seq = udpdb->curr_sequence_no;
 
        /* Accept a "restart" which occurs within a 1 second window */ 
        if (udpdb->curr_sequence_no < (udpdb->spectra_per_second*5)) {

          multilog (log, LOG_INFO, "Start detected on packet %"PRIu64"\n",udpdb->curr_sequence_no);
          //udpdb->expected_sequence_no = udpdb->curr_sequence_no + 1;

          /* Increment the buffer counts with the missed packets */
          if (udpdb->curr_sequence_no < udpdb->mid_sequence)
            udpdb->curr_buffer_count += udpdb->curr_sequence_no;
          else {
            udpdb->curr_buffer_count += BPSR_NUM_UDP_PACKETS;
            udpdb->next_buffer_count += udpdb->curr_sequence_no - udpdb->mid_sequence;
          }

        } else {

          ignore_packet = 1;
        }

        if (udpdb->received != BPSR_UDP_PAYLOAD_BYTES) {
          multilog (log, LOG_ERR, "UDP packet size was incorrect (%"PRIu64" != %d)\n", udpdb->received, BPSR_UDP_PAYLOAD_BYTES);
          *size = DADA_ERROR_HARD;
          break;
        }
      }

      /* If we are still waiting dor the start of data */
      if (ignore_packet) {

        max_ignore -= BPSR_UDP_PAYLOAD_BYTES;
        if (max_ignore < 0) 
          quit = 1;

      } else {

        /* If the packet we received was too small, pad it */
        if (udpdb->received < BPSR_UDP_PAYLOAD_BYTES) {
        
          uint64_t amount_to_pad = BPSR_UDP_PAYLOAD_BYTES - udpdb->received;
          char * buffer_pos = (udpdb->socket_buffer) + udpdb->received;
 
          /* 0 the next buffer */
          memset(buffer_pos, zerodchar, amount_to_pad);

          multilog (log, LOG_WARNING, "Short packet received, padded %"PRIu64
                                      " bytes\n", amount_to_pad);
        } 

        /* If the packet we received was too long, warn about it */
        if (udpdb->received > BPSR_UDP_PAYLOAD_BYTES) {

          multilog (log, LOG_WARNING, "Long packet received, truncated to %"PRIu64
                                      " bytes\n", BPSR_UDP_DATASIZE_BYTES);
        }

        /* Now try to slot the packet into the appropraite buffer */
        data_received += BPSR_UDP_DATASIZE_BYTES;

        /* Increment statistics */
        udpdb->packets->received++;
        udpdb->packets->received_per_sec++;
        udpdb->bytes->received += BPSR_UDP_DATASIZE_BYTES;
        udpdb->bytes->received_per_sec += BPSR_UDP_DATASIZE_BYTES;

        /* If the packet belongs in the curr_buffer */
        if ((udpdb->curr_sequence_no >= udpdb->min_sequence) && 
            (udpdb->curr_sequence_no <  udpdb->mid_sequence)) {

          uint64_t buf_offset = (udpdb->curr_sequence_no - udpdb->min_sequence) * BPSR_UDP_DATASIZE_BYTES;

          memcpy( (udpdb->curr_buffer)+buf_offset, 
                  (udpdb->socket_buffer)+BPSR_UDP_COUNTER_BYTES,
                  BPSR_UDP_DATASIZE_BYTES);

          udpdb->curr_buffer_count++;

        } else if ((udpdb->curr_sequence_no >= udpdb->mid_sequence) && 
                   (udpdb->curr_sequence_no <  udpdb->max_sequence)) {

          uint64_t buf_offset = (udpdb->curr_sequence_no - udpdb->mid_sequence) * BPSR_UDP_DATASIZE_BYTES;

          memcpy( (udpdb->curr_buffer)+buf_offset, 
                  (udpdb->socket_buffer)+BPSR_UDP_COUNTER_BYTES,
                  BPSR_UDP_DATASIZE_BYTES);

          udpdb->next_buffer_count++;

        /* If this packet has arrived too late, it has already missed out */
        } else if (udpdb->curr_sequence_no < udpdb->min_sequence) {
 
          multilog (log, LOG_WARNING, "Packet arrived too soon, %"PRIu64" < %"PRIu64"\n",
                    udpdb->curr_sequence_no, udpdb->min_sequence);

        /* If a packet has arrived too soon, then we give up trying to fill the 
           curr_buffer and return what we do have */
        } else if (udpdb->curr_sequence_no >= udpdb->max_sequence) {

          float curr_percent = ((float) udpdb->curr_buffer_count / (float) BPSR_NUM_UDP_PACKETS)*100;
          float next_percent = ((float) udpdb->next_buffer_count / (float) BPSR_NUM_UDP_PACKETS)*100;
          //multilog (log, LOG_WARNING, "%"PRIu64" > %"PRIu64"\n",udpdb->curr_sequence_no,udpdb->max_sequence);
  
          multilog (log, LOG_WARNING, "Not keeping up. curr_buffer %5.2f%, next_buffer %5.2f%\n",
                                      curr_percent, next_percent);

          udpdb->packet_in_buffer = 1;
          quit = 1;

        } else {

          fprintf (stderr,"Sequence number invalid\n");

        }


        /* If we have filled the current buffer, then we can stop */
        if (udpdb->curr_buffer_count == BPSR_NUM_UDP_PACKETS) {
          quit = 1;
        } else {
          assert(udpdb->curr_buffer_count < BPSR_NUM_UDP_PACKETS);
        }

        /* If the next buffer is at least half full */
        if (udpdb->next_buffer_count > (BPSR_NUM_UDP_PACKETS / 2)) {
          float curr_percent = ((float) udpdb->curr_buffer_count / (float) BPSR_NUM_UDP_PACKETS)*100;
          float next_percent = ((float) udpdb->next_buffer_count / (float) BPSR_NUM_UDP_PACKETS)*100;

          multilog(log, LOG_WARNING, "Bailing curr_buf %5.2f%, next_buffer %5.2f%\n",curr_percent,next_percent);
          quit = 1;
        }
      }
    } 
  }

   /*multilog (log, LOG_INFO, "curr: %"PRIu64", next: %"PRIu64", capacity: %"PRIu64"\n",udpdb->curr_buffer_count, udpdb->next_buffer_count, BPSR_NUM_UDP_PACKETS); */

  /* If we have received a packet during this function call */
  if (data_received) {

    /* If we have not received all the packets we expected */
    if ((udpdb->curr_buffer_count < BPSR_NUM_UDP_PACKETS) && (!timeout_ocurred)) {

      multilog (log, LOG_WARNING, "Dropped %"PRIu64" packets\n",
               (BPSR_NUM_UDP_PACKETS - udpdb->curr_buffer_count));

      udpdb->packets->dropped += (BPSR_NUM_UDP_PACKETS - udpdb->curr_buffer_count);
      udpdb->packets->dropped_per_sec += (BPSR_NUM_UDP_PACKETS - udpdb->curr_buffer_count);
      udpdb->bytes->dropped += (BPSR_UDP_DATASIZE_BYTES * (BPSR_NUM_UDP_PACKETS - udpdb->curr_buffer_count));
      udpdb->bytes->dropped_per_sec += (BPSR_UDP_DATASIZE_BYTES * (BPSR_NUM_UDP_PACKETS - udpdb->curr_buffer_count));

    }

    /* If the timeout ocurred, this is most likely due to end of data */
    if (timeout_ocurred) {
      *size = udpdb->curr_buffer_count * BPSR_UDP_DATASIZE_BYTES;
      multilog (log, LOG_WARNING, "Suspected EOD received, returning "
                     "%"PRIu64" bytes\n",*size);
    }

    udpdb->expected_sequence_no += BPSR_NUM_UDP_PACKETS;

  } else {

    /* If we have received no data, then return a size of 0 */
    *size = 0;

  }

  udpdb->prev_time = udpdb->current_time;
  udpdb->current_time = time(0);
  
  if (udpdb->prev_time != udpdb->current_time) {

    if (udpdb->verbose) {

      if (ignore_packet) {

        multilog(log, LOG_INFO, "Ignoring out of range sequence no: %"PRIu64" > %"PRIu64"\n",udpdb->curr_sequence_no, (10 * udpdb->spectra_per_second));

      } else {

        multilog(log, LOG_INFO, "Packets [%"PRIu64",%"PRIu64"] [%"PRIu64",%"PRIu64"]\n",
                 udpdb->packets->dropped_per_sec,udpdb->packets->dropped,
                 udpdb->packets->received_per_sec,udpdb->packets->received);

        multilog(log, LOG_INFO, "Bytes   [%"PRIu64",%"PRIu64"] [%"PRIu64",%"PRIu64"]\n",
                 udpdb->bytes->dropped_per_sec,udpdb->bytes->dropped,
                 udpdb->bytes->received_per_sec,udpdb->bytes->received);
      }

    } else {
      /*
      multilog(log, LOG_INFO, "Packet loss %5.3f % / %5.3f %, Data loss %5.3f %\n", 
               (100 * ((float) udpdb->packets->dropped / (float) udpdb->packets->received)), 
               (100 * ((float) udpdb->packets->dropped_per_sec / (float) udpdb->packets->received_per_sec)), 
               (100 * ((float) udpdb->bytes->dropped / (float) udpdb->bytes->received)));
       */
    }

    udpdb->packets->received_per_sec = 0;
    udpdb->packets->dropped_per_sec = 0;
    udpdb->bytes->received_per_sec = 0;
    udpdb->bytes->dropped_per_sec = 0;

  }

  assert(udpdb->curr_buffer != 0);
  
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
  char * interface = "any";

  /* port on which to listen for udp packets */
  int port = BPSR_DEFAULT_UDPDB_PORT;

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

  char * ibob_host;
  int    ibob_port;

  while ((arg=getopt(argc,argv,"di:p:vm:S:H:n:1h")) != -1) {
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

  if ((argc - optind) != 2) {
    fprintf(stderr, "Error: host and port must be specified\n");
    usage();
    return EXIT_FAILURE;
  } else {
    ibob_host = argv[optind];
    ibob_port = atoi(argv[(optind+1)]);
  }

  log = multilog_open ("bpsr_udpdb", 0);

  if (daemon) 
    be_a_daemon ();
  else
    multilog_add (log, stderr);

  multilog_serve (log, DADA_DEFAULT_PWC_LOG);
  
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
  udpdb.verbose = verbose;
  udpdb.interface = strdup(interface);
  udpdb.port = port;
  udpdb.acc_len = BPSR_DEFAULT_ACC_LEN;   // 25
  udpdb.mode = mode;
  udpdb.ibob_host = strdup(ibob_host);
  udpdb.ibob_port = ibob_port;

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
    } else 

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

/* 
 * The following code handles ibob communication
 */

time_t set_ibob_levels(dada_pwc_main_t* pwcm) {

  udpdb_t* udpdb  = (udpdb_t*)pwcm->context;
  
  multilog_t* log = pwcm->log;

  int verbose = udpdb->verbose;
  verbose = 1;
  int init_scale = 131072;
  int n_attempts = 3;
  int arg = 0;
  
  char command[128];
  char lilreply[128];
  int fd = 0;
  int bpsr_interface_fd = 0;
  int n_atttempts = 3;
  int rval = 0;


  /* Not sure if this helps, but try to spread the access to ibobs out a little */
  sprintf(command, "/home/apsr/linux_64/bin/ibob_level_setter -n 0 %s %d\n",udpdb->ibob_host, udpdb->ibob_port);
  system(command);

  // Rearm the ibob
  srand ( udpdb->ibob_port );
  double randtime = (rand() / (RAND_MAX + 1.0) ) * 1000000;
  double notsorandtime = (udpdb->ibob_port - 2000) * 100000;

  struct timeval timeout;
  if (notsorandtime > 1000000) {
    timeout.tv_sec=1;
    timeout.tv_usec=(int) (notsorandtime-1000000);
  } else {
    timeout.tv_sec=0;
    timeout.tv_usec=(int) notsorandtime;
  }
  //timeout.tv_usec=(int) randtime;
  select(0,NULL,NULL,NULL,&timeout);

  /*
  fd = sock_open(udpdb->ibob_host, udpdb->ibob_port);
  if (fd) {
    sprintf(command, "regread reg_coeff_pol1\r\n");
    rval = sock_write(fd, command, strlen(command)-1);
    if (rval == 0) {
       multilog(log, LOG_ERR, "ibob write command failed %s\n",command);
       sock_close(fd);
    } else {
      // Try to clear the fucking thing 
      rval = from_ibob(fd, lilreply);
      multilog(log, LOG_INFO, "Read %s from ibob after initial test command\n", lilreply);
      sock_close(fd);
    }
  }*/

  
  while (n_attempts)
  {
    fd = sock_open(udpdb->ibob_host, udpdb->ibob_port);
    if (fd < 0)
    {
      multilog(log, LOG_WARNING, "Could not connect to ibob on %s:%d\n",
                       udpdb->ibob_host, udpdb->ibob_port); 
      n_attempts--;
      sleep(1);
    }
    else
    {
      multilog(log, LOG_INFO, "Connected to ibob on %s:%d\n",
                      udpdb->ibob_host, udpdb->ibob_port);
      n_attempts = 0;
    }
  }

  if ((!fd) && (!n_attempts))
    return -1;

  if (udpdb->mode != 1)
  {
    n_attempts = 3;

    while (n_attempts)
    {
      bpsr_interface_fd = sock_open("srv0", 57011);
      if (bpsr_interface_fd < 0) {
        multilog(log, LOG_WARNING, "Could not connect to bpsr_interface on\n",
                         "srv0", 57011); 
        n_attempts--;    
        sleep(1);
      
      } else {
        multilog(log, LOG_INFO, "Connected to tcs interface on srv0:%d\n",57011);
        n_attempts = 0;
      }                 
    } 
  }

  //sprintf(command, "/home/apsr/linux_64/bin/ibob_level_setter -n 2 %s %d",udpdb->ibob_host, udpdb->ibob_port);
  //system(command);

  // Write the initial coefficients
  long pol1_coeff = init_scale;
  long pol2_coeff = init_scale;
  rval = 0;

  char *pol1_bram = malloc(sizeof(char) * 163842);
  char *pol2_bram = malloc(sizeof(char) * 163842);

  long *pol1_vals = malloc(sizeof(long) * 512);
  long *pol2_vals = malloc(sizeof(long) * 512);

  long pol1_max_value;
  long pol2_max_value;

  int i=0;
  int bytes_read = 0;
  int bit_window = 0;
  n_attempts = 3;

  // Try to clear the fucking thing 
  rval = from_ibob(fd, pol1_bram);
  multilog(log, LOG_INFO, "Ibob had %s on the line before beginning\n", pol1_bram);

  while (n_attempts)
  {
    sprintf(command, "regwrite reg_coeff_pol1 %d\r\n",pol1_coeff);
    if (verbose) 
      multilog(log, LOG_INFO, "regwrite reg_coeff_pol1 %d\n",pol1_coeff);

    if (rval = sock_write(fd, command, strlen(command)-1) < 0 )
    {
      multilog(log, LOG_ERR, "ibob write command failed %s\n",command);
      sock_close(fd);
      return(-1);
    }

    sprintf(command, "regwrite reg_coeff_pol2 %d\r\n",pol2_coeff);
    if (verbose) 
        multilog(log, LOG_INFO, "regwrite reg_coeff_pol2 %d\n",pol2_coeff);

    if (rval = sock_write(fd, command, strlen(command)-1) < 0 )
    {
      multilog(log, LOG_ERR, "ibob write command failed %s\n",command);
      sock_close(fd);
      return(-1);
    }
    
    strcpy(command, "bramdump scope_output1/bram\r\n");
    if (verbose) 
      multilog(log, LOG_INFO, "bramdump scope_output1/bram\n");

    rval = sock_write(fd, command, strlen(command) -1);
    bytes_read = 0;
    rval = 1;
    while ((rval > 0) && (bytes_read < 163841))
    {
      rval = sock_tm_read(fd, (void *) (pol1_bram + bytes_read), 4096, 1.0);
      bytes_read += rval;
      //multilog(log, LOG_INFO, "read %d bytes, %d total\n",rval, bytes_read);
    }
    pol1_bram[163841] = '\0';

    strcpy(command, "bramdump scope_output3/bram\r\n");
    if (verbose) 
      multilog(log, LOG_INFO, "bramdump scope_output3/bram\n");

    rval = sock_write(fd, command, strlen(command) -1);
    bytes_read = 0;
    rval = 1;
    while ((rval > 0) && (bytes_read < 163841))
    {
      rval = sock_tm_read(fd, (void *) pol2_bram + bytes_read, 4096, 1.0);
      bytes_read += rval;
      //multilog(log, LOG_INFO, "read %d bytes, %d total\n",rval, bytes_read);
    }
    pol2_bram[163841] = '\0';

    if (verbose)
       multilog(log, LOG_INFO, "extracting counts\n");

    if (bytes_read)
    {
      extract_counts(pol1_bram, pol1_vals);
      extract_counts(pol2_bram, pol2_vals);

      if (verbose)
        multilog(log, LOG_INFO, "calculating maxes\n");

      pol1_max_value = calculate_max(pol1_vals, 8);
      pol2_max_value = calculate_max(pol2_vals, 8);

      if (verbose)
        multilog(log, LOG_INFO, "max vals = %d,%d\n",pol1_max_value, pol2_max_value);
    
      bit_window = find_bit_window(pol1_max_value, pol2_max_value, &pol1_coeff, &pol2_coeff);

      if (verbose) 
        multilog(log, LOG_INFO, "bit window %d, new scale factors %d,%d\n", bit_window,pol1_coeff, pol2_coeff);

      break;
    }
    else
    {
      n_attempts --;
      multilog(log, LOG_INFO, "Fail at setting levels... remaining attempts:%d\n", n_attempts);
    }
  }

  // Write the bit window choice to the ibob
  sprintf(command,"regwrite reg_output_bitselect %d\r\n",bit_window);
  if (verbose)
    multilog(log, LOG_INFO, "regwrite reg_output_bitselect %d\n",bit_window);

  rval = sock_write(fd, command, strlen(command) -1);

  // Tell the BPSR TCS interface script that we are ready
  if (udpdb->mode != 1) {
    sprintf(command, "READY\r\n");
    rval = sock_write(bpsr_interface_fd, command, strlen(command) -1);
    sock_close(bpsr_interface_fd);
  }

  timeout.tv_sec=0;
  timeout.tv_usec=400000;

  time_t curr = time(0);
  time_t prev = curr;

  while (curr== prev) {
    curr = time(0);
  }

  select(0,NULL,NULL,NULL,&timeout);

  if (verbose)
    multilog(log, LOG_INFO, "regwrite reg_arm 0\n");

  sprintf(command, "regwrite reg_arm 0\r\n");
  rval = sock_write(fd, command, strlen(command) -1);

  if (verbose)
    multilog(log, LOG_INFO, "regwrite reg_arm 1\n");

  sprintf(command, "regwrite reg_arm 1\r\n");
  rval = sock_write(fd, command, strlen(command) -1);

  if (verbose) 
    multilog(log, LOG_INFO, "closing ibob socket\n");

  sock_close(fd);
  
  return (curr + 1);

  /*sprintf(command, "/home/apsr/linux_64/bin/ibob_rearm_trigger %s %d",udpdb->ibob_host, udpdb->ibob_port);
  long int utc_unix;
  FILE * fptr = popen(command, "r");
  fscanf(fptr, "%ld", &utc_unix);
  pclose(fptr);
  return utc_unix;*/

}

void extract_counts(char * bram, long * vals) {

  const char *sep = "\r\n";
  char *line;
  char value[11];
  int i=0;

  line = strtok(bram, sep);
  strncpy(value, line+69, 11);
  vals[i] = atol(value);

  for (i=1; i<512; i++) {
    line = strtok(NULL, sep);
    strncpy(value, line+69, 11); 
    vals[i] = atol(value);
  }

}

long calculate_max(long * vals, long birdie_factor) {

  sort(vals,0,511);
  long median  = vals[255];
  long birdie_cutoff = birdie_factor * median;
  long max_value = 0;

  int i=0;
  int n_sum=0;
  double sum = 0;
  double sum_mean = 0;

  for (i=0; i<512; i++) {
    if (vals[i] < birdie_cutoff) {
      sum += (double) vals[i]; 
      n_sum++;
      if (vals[i] > max_value)
        max_value = vals[i];
    }
  }
  
  long mean_value = (long) ( sum / (double) n_sum);
  //fprintf(stderr, "max = %d, mean = %d\n",max_value, mean_value);

  return max_value;
}

void swap(long *a, long *b)
{
  long t=*a; *a=*b; *b=t;
}

void sort(long arr[], long beg, long end)
{
  if (end > beg + 1)
  {
    long piv = arr[beg], l = beg + 1, r = end;
    while (l < r)
    {
      if (arr[l] <= piv)
        l++;
      else
        swap(&arr[l], &arr[--r]);
    }
    swap(&arr[--l], &arr[beg]);
    sort(arr, beg, l);
    sort(arr, r, end);
  }
}

int find_bit_window(long pol1val, long pol2val, long *gain1, long* gain2) {

  long bitsel_min[4] = {0, 256, 65535, 16777216 };
  long bitsel_mid[4] = {64, 8192, 2097152, 536870912};
  long bitsel_max[4] = {255, 65535, 16777215, 4294967295};

  long val = ((long) (((double) pol1val + (double) pol2val) / 2.0));

  //Find which bit selection window we are currently sitting in
  int i=0;
  int current_window = 0;
  int desired_window = 0;

  for (i=0; i<4; i++) {

    //If average (max) value is in the lower half
    if ((val > bitsel_min[i]) && (val <= bitsel_mid[i])) {
      current_window = i;

      if (i == 0) {
        desired_window = 0;

      } else {
        desired_window = i-1;
      }
    }

    // If average (max)n value is in the upper half, simply raise to
    // the top of this window
    if ((val > bitsel_mid[i]) && (val <= bitsel_max[i])) {
      current_window = i;
      desired_window = i;
    }
  }

  if (desired_window == 3) {
    desired_window = 2;
  }

  if ((pol1val == 0) || (pol2val == 0)) {
    pol1val = 1;
    pol2val = 1;
  }

  long desired_val =  ((bitsel_max[desired_window]+1) / 2);

  //printf("desired val = %d\n",desired_val);
  
  double gain_factor1 = (double) desired_val / (double) pol1val;
  double gain_factor2 = (double) desired_val / (double) pol2val;

  gain_factor1 = sqrt(gain_factor1);
  gain_factor2 = sqrt(gain_factor2);

  gain_factor1 *= *gain1;
  gain_factor2 *= *gain2;

  *gain1 = (long) floor(gain_factor1);
  *gain2 = (long) floor(gain_factor2);

  if (*gain1 > 200000) {
    *gain1 = 200000;
  }
  if (*gain2 > 200000) {
    *gain2 = 200000;
  }
  return desired_window;
}

int from_ibob(int fd, char *response) {

  int bytes_read = 0;
  int rval = 1;
  while (rval > 0) {
    rval = sock_tm_read(fd, (void *) (response + bytes_read), 4096, 0.1);
    bytes_read += rval;
  }

  int i=0;
  for (i=0; i<bytes_read; i++) {
    if (response[i] == '\r')
       response[i] = ' ';
    if (response[i] == '\n')
       response[i] = ',';
  }

  response[bytes_read] = '\0';

  return bytes_read;
}


