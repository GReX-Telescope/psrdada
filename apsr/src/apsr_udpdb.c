#include "apsr_def.h"
#include "apsr_udpdb.h"

#include <math.h>

typedef struct {
  signed char r;
  signed char i;
} complex_char;

void usage()
{
  fprintf (stdout,
	   "apsr_udpdb [options]\n"
     " -h             print help text\n"
	   " -p             port on which to listen [default %d]\n"
	   " -d             run as daemon\n"
     " -m mode        1 for independent mode, 2 for controlled mode\n"
     " -v             verbose messages\n"
     " -1             one time only mode, exit after EOD is written\n"
     " -H filename    ascii header information in file\n"
     " -c             don't verify udp headers against header block\n"
     " -S num         file size in bytes\n",APSR_DEFAULT_UDPDB_PORT);
}


int64_t sock_recv (int fd, char* buffer, uint64_t size, int flags)
{
  int64_t received = 0;
  received = recvfrom (fd, buffer, size, 0, NULL, NULL);

  if (received < 0) {
    perror ("sock_recv recvfrom");
    return -1;
  }
  if (received == 0) {
    fprintf (stderr, "sock_recv received zero bytes\n");
  }

  return received;
}

int udpdb_error_function (dada_pwc_main_t* pwcm) {

  udpdb_t *udpdb = (udpdb_t*)pwcm->context;

  /* If we are still waiting for that first packet, we don't
   * really have an error state :), but the buffer function
   * will return 0 bytes */
  if (udpdb->expected_sequence_no > 0) {
    //sleep(1);
    udpdb->error_seconds--;
    //multilog (pwcm->log, LOG_WARNING, "Error Function has %"PRIu64" error "
    //          "seconds remaining\n",udpdb->error_seconds);
  }

  if (udpdb->error_seconds <= 0) 
    return 2;
  else
    return 1;
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

  /* Check whether the RESOLUTION is known */
  if (ascii_header_get (pwcm->header, "RESOLUTION", "%d", &resolution) != 1) {
    valid = 0;
  }

  return valid;

}


time_t udpdb_start_function (dada_pwc_main_t* pwcm, time_t start_utc)
{

  /* get our context, contains all required params */
  udpdb_t* udpdb = (udpdb_t*)pwcm->context;
  
  multilog_t* log = pwcm->log;

  /* Initialise variables */
  udpdb->packets_dropped_this_run = 0;
  udpdb->packets_received_this_run = 0;
  udpdb->bytes_received_this_run = 0;
  udpdb->error_seconds = 10;
  udpdb->dropped_packets_to_fill = 0;
  udpdb->packet_in_buffer = 0;
  udpdb->prev_time = time(0);
  udpdb->current_time = udpdb->prev_time;
  udpdb->curr_buffer_count = 0;
  udpdb->next_buffer_count = 0;
  udpdb->packets_late_this_sec = 0;
  udpdb->npol = 2;
  
  /* We should never need/want to do this...
  if (start_utc > now) {
    sleep_time = start_utc - now;
    multilog (log, LOG_INFO, "sleeping for %u sec\n", sleep_time);
    sleep (sleep_time);
  }*/

  /* Create a udp socket */
  if (create_udp_socket(pwcm) < 0) {
    multilog (log, LOG_ERR, "Error, Failed to create udp socket\n");
    return 0; // n.b. this is an error value 
  }

  /* Check the whether the requried header fields have been set via the 
   * control interface. This is required when receiving data from the DFB3
   * board, but not required for data transfer */
 
  if (ascii_header_get (pwcm->header, "NBAND", "%d", 
                        &(udpdb->expected_nbands)) != 1) {
    multilog (log, LOG_WARNING, "Warning. NBAND not set in header, "
                                "Using 1 as default\n");
    ascii_header_set (pwcm->header, "NBAND", "%d", 1);
  }
                                                                                
  if (ascii_header_get (pwcm->header, "NBIT", "%d",
                        &(udpdb->expected_header_bits)) != 1) {
    multilog (log, LOG_WARNING, "Warning. NBIT not set in header. "
                                "Using 8 as default.\n");
    ascii_header_set (pwcm->header, "NBIT", "%d", 8);
  }
                                                                                
  if (ascii_header_get (pwcm->header, "NCHAN", "%d",
                       &(udpdb->expected_nchannels)) != 1) {
    multilog (log, LOG_WARNING, "Warning. NCHAN not set in header "
                               "Using 1 as default.\n");
    ascii_header_set (pwcm->header, "NCHAN", "%d", 2);
  }

  /* Set the current machines name in the header block as RECV_HOST */
  char myhostname[HOST_NAME_MAX] = "unknown";;
  gethostname(myhostname,HOST_NAME_MAX); 
  ascii_header_set (pwcm->header, "RECV_HOST", "%s", myhostname);
  
  /* setup the data buffer for NUMUDPACKETS udp packets */
  udpdb->datasize = 32 * 1024 * 1024;
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
  udpdb->socket_buffer= (char *) malloc(sizeof(char) * UDPBUFFSIZE);
  assert(udpdb->socket_buffer != 0);

  /* setup the expected sequence no to the initial value */
  udpdb->expected_sequence_no = 0;

  /* set the packet_length to an expected value */
  udpdb->packet_length = 8972 - UDPHEADERSIZE;

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

  // We will make the size of the buffer returned equal to
  // the default block size (512K)

  /* Packets dropped in the last second */
  uint64_t packets_dropped_this_second = 0;

  /* How many packets will fit in each curr_buffer and next_buffer */
  uint64_t buffer_capacity = udpdb->datasize / udpdb->packet_length;

  header_struct header = { '0', '0', 0, '0', '0', '0',"0000", 0 };

  // A char with all bits set to 0
  char zerodchar = 'c'; 
  memset(&zerodchar,0,sizeof(zerodchar));

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
  memset(udpdb->next_buffer,zerodchar,udpdb->datasize);

  /* Determine the sequence number boundaries for curr and next buffers */
  uint64_t min_sequence = udpdb->expected_sequence_no;
  uint64_t mid_sequence = min_sequence + buffer_capacity;
  uint64_t max_sequence = mid_sequence + buffer_capacity;

  //multilog(log, LOG_INFO, "[%"PRIu64" <-> %"PRIu64" <-> %"PRIu64"]\n",min_sequence,mid_sequence,max_sequence); 

  /* Assume we will be able to return a full buffer */
  *size = buffer_capacity * udpdb->packet_length;

  /* Continue to receive packets */
  while (!quit) {

    /* If we had a packet in the socket)buffer a previous call to the 
     * buffer function*/
    if (udpdb->packet_in_buffer) {

      udpdb->packet_in_buffer = 0;
                                                                                
    /* Else try to get a fresh packet */
    } else {
                                                                                
      /* select to see if data has arrive. If recording, allow a 1 second
       * timeout, else 0.1 seconds to ensure buffer function is responsive */
      if (pwcm->pwc->state == dada_pwc_recording) {
        timeout.tv_sec=1;
        timeout.tv_usec=0;
      } else {
        timeout.tv_sec=0;
        timeout.tv_usec=100000;
      }

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
        udpdb->received = sock_recv (udpdb->fd, udpdb->socket_buffer,
                                     UDPBUFFSIZE, 0);
      }
    }

    /* If we did get a packet within the timeout, or one was in the buffer */
    if (!quit) {

      /* Decode the packets apsr specific header */
      decode_header(udpdb->socket_buffer, &header);

      /* When we have received the first packet */      
      if ((udpdb->expected_sequence_no == 0) && (data_received == 0)) {

        //multilog (log, LOG_INFO, "received = %d\n",udpdb->received);
        //multilog (log, LOG_INFO, "header.length = %d\n",header.length);
        //multilog (log, LOG_INFO, "header.source = %d\n",header.source);
        //multilog (log, LOG_INFO, "header.sequence = %d\n",header.sequence);
        //multilog (log, LOG_INFO, "header.bits = %d\n",header.bits);
        //multilog (log, LOG_INFO, "header.channels = %d\n",header.channels);
        //multilog (log, LOG_INFO, "header.bands = %d\n",header.bands);

        if (header.sequence != 0) {
          multilog (log, LOG_WARNING, "First packet received had sequence "
                                  "number %d\n",header.sequence);

          /* If the first packet is more than 1 gigabyte into the future,
           * then something is drastically wrong */
          if ((header.sequence * udpdb->received) > (1024*1024*1024)) {
            multilog (log, LOG_ERR, "First packet sequence was more than "
                                      "1 GB into the future. Stopping\n");
            *size = DADA_ERROR_HARD; 
          }

        } else {
          multilog (log, LOG_INFO, "First packet has been received\n");
        }

        /* If the UDP packet thinks its header is different to the prescribed
         * size, give up */
        if (header.length != UDPHEADERSIZE) {
          multilog (log, LOG_ERR, "Custom UDP header length incorrect. "
                   "Expected %d bytes, received %d bytes\n",UDPHEADERSIZE,
                   header.length);
          *size = DADA_ERROR_HARD;
          break;
        }

        /* define length of all future packets */
        udpdb->packet_length = udpdb->received - UDPHEADERSIZE;

        /* Set the number of packets a buffer can hold */
        buffer_capacity = udpdb->datasize / udpdb->packet_length;

	      multilog(log, LOG_INFO, "packet_length = %"PRIu64" bytes, udpdb->datasize = %"PRIu64", buffer_capacity = %"PRIu64"\n",udpdb->packet_length, udpdb->datasize, buffer_capacity);

        /* Adjust sequence numbers, now that we know packet_length*/
        min_sequence = udpdb->expected_sequence_no;
        mid_sequence = min_sequence + buffer_capacity;
        max_sequence = mid_sequence + buffer_capacity;

        /* multilog(log, LOG_INFO, "FIRST: [%"PRIu64" <-> %"PRIu64" <-> %"PRIu64"]\n",min_sequence,mid_sequence,max_sequence); */

        /* Set the amount a data we expect to return */
        *size = buffer_capacity * udpdb->packet_length;

        /* We define the polarization length to be the number of bytes
           in the packet, and set RESOLUTION header parameter accordingly */
        udpdb->npol = 2;
        if ( ascii_header_get (pwcm->header, "NPOL", "%d", &(udpdb->npol)) < 1) {
          multilog (log, LOG_WARNING, "Could not get NPOL header parameter\n");
        }

        header.pollength = (unsigned int) udpdb->packet_length;

        /* Set the resoultion header variable */
        if ( ascii_header_set (pwcm->header, "RESOLUTION", "%d",
                               header.pollength) < 0)
        {
                multilog (log, LOG_ERR, "Could not set RESOLUTION header parameter"
                    " to %d\n", header.pollength);
          *size = DADA_ERROR_HARD;
          break;
        }
      }

      /* If the packet we received was too small, pad it */
      if (udpdb->received < (udpdb->packet_length + UDPHEADERSIZE)) {

        uint64_t amount_to_pad = (udpdb->packet_length + UDPHEADERSIZE) - 
                                 udpdb->received;
        char * buffer_pos = (udpdb->socket_buffer) + udpdb->received;

        /* 0 the next buffer */
        memset(buffer_pos, zerodchar, amount_to_pad);

        multilog (log, LOG_WARNING, "Short packet received, padded %"PRIu64
                                 " bytes\n", amount_to_pad);
      }

      /* If the packet we received was too long, warn about it */
      if (udpdb->received > (udpdb->packet_length + UDPHEADERSIZE)) {
        multilog (log, LOG_WARNING, "Long packet received, truncated to %"PRIu64
                                 " bytes\n", udpdb->packet_length);
      }

      /* Now try to slot the packet into the appropraite buffer */
      data_received += (udpdb->received - UDPHEADERSIZE);

      /* Increment statistics */
      udpdb->packets_received_this_run++;
      udpdb->packets_received++;
      udpdb->bytes_received_this_run += (udpdb->received - UDPHEADERSIZE);
      udpdb->bytes_received += (udpdb->received - UDPHEADERSIZE);
 
      int crazymode = 0;

      if (crazymode) {

        uint64_t buf_offset = udpdb->curr_buffer_count * 
                              udpdb->packet_length;

        memcpy( (udpdb->curr_buffer)+buf_offset,
                (udpdb->socket_buffer)+UDPHEADERSIZE,
                udpdb->packet_length);

        udpdb->curr_buffer_count++;

      } else {

        /* If the packet belongs in the curr_buffer */
        if ((header.sequence >= min_sequence) && 
            (header.sequence <  mid_sequence)) {

           
          uint64_t buf_offset = (header.sequence - min_sequence) * 
                                 udpdb->packet_length;


          memcpy( (udpdb->curr_buffer)+buf_offset, 
                  (udpdb->socket_buffer)+UDPHEADERSIZE,
                  udpdb->packet_length);

          udpdb->curr_buffer_count++;


        } else if ((header.sequence >= mid_sequence) && 
                   (header.sequence <  max_sequence)) {
  
          uint64_t buf_offset = (header.sequence - mid_sequence) *
                                 udpdb->packet_length;

          memcpy( (udpdb->curr_buffer)+buf_offset, 
                  (udpdb->socket_buffer)+UDPHEADERSIZE,
                  udpdb->packet_length);

          udpdb->next_buffer_count++;

        /* If this packet has arrived too late, it has already missed out */
        } else if (header.sequence < min_sequence) {
 
          udpdb->packets_late_this_sec++; 

        /* If a packet has arrived too soon, then we give up trying to fill the 
           curr_buffer and return what we do have */
        } else if (header.sequence >= max_sequence) {

          float curr_percent = ((float) udpdb->curr_buffer_count / (float) buffer_capacity)*100;
          float next_percent = ((float) udpdb->next_buffer_count / (float) buffer_capacity)*100;
  
          multilog (log, LOG_WARNING, "Packet %d arrived too soon, expecting "
                                      "packets %"PRIu64" through %"PRIu64" [%8.5f,%8.5f]\n",
                                      header.sequence, min_sequence, 
                                      max_sequence, curr_percent, next_percent);

          udpdb->packet_in_buffer = 1;
          quit = 1;

        } else {

          /* SHOULD NEVER HAPPEN */    
          fprintf (stderr,"ERROR!!!\n");

        }

      /* TODO REMOVE!!! */
      } /* end crazymode */



      /* If we have filled the current buffer, then we can stop */
      if (udpdb->curr_buffer_count == buffer_capacity) {
        quit = 1;
      } else {
        assert(udpdb->curr_buffer_count < buffer_capacity);
      }

      /* If the next buffer is at least half full */
      if (udpdb->next_buffer_count > (buffer_capacity / 2)) {
        quit = 1;
      }
    }
  } 

  /* multilog (log, LOG_INFO, "curr: %"PRIu64", next: %"PRIu64", capacity: %"PRIu64"\n",udpdb->curr_buffer_count, udpdb->next_buffer_count, buffer_capacity); */

  /* If we have received a packet during this function call */
  if (data_received) {

    /* If we have not received all the packets we expected */
    if ((udpdb->curr_buffer_count < buffer_capacity) && (!timeout_ocurred)) {

      multilog (log, LOG_WARNING, "Dropped %"PRIu64" packets\n"
               , (buffer_capacity - udpdb->curr_buffer_count));

      udpdb->packets_dropped += buffer_capacity - udpdb->curr_buffer_count;
      udpdb->packets_dropped_this_run += buffer_capacity 
                                         - udpdb->curr_buffer_count;
    }

    /* If the timeout ocurred, this is most likely due to end of data */
    if (timeout_ocurred) {
      *size = udpdb->curr_buffer_count * udpdb->packet_length;
      multilog (log, LOG_WARNING, "Suspected EOD received, returning "
                     "%"PRIu64" bytes\n",*size);
    }

    udpdb->expected_sequence_no += buffer_capacity;

  } else {

    /* If we have received no data, then return a size of 0 */
    *size = 0;

  }

  /* Failure conditions */
  /*
  uint64_t drop_threshold = 0.75 * 
                           ipcbuf_get_nbufs((ipcbuf_t*)pwcm->data_block) *
                           ipcbuf_get_bufsz((ipcbuf_t*)pwcm->data_block);
          
  if (n_dropped * udpdb->packet_length > drop_threshold) {
    multilog (log, LOG_ERR, "Too many udp packets dropped. Threshold is"
              " %"PRIu64" bytes , but dropped %"PRIu64" bytes \n",
              drop_threshold, n_dropped*udpdb->packet_length); 
    *size = DADA_ERROR_HARD;
  }*/
                
  udpdb->prev_time = udpdb->current_time;
  udpdb->current_time = time(0);
  
  if (udpdb->prev_time != udpdb->current_time) {

    //multilog(log, LOG_INFO, "%d: %d bytes\n",header.sequence, udpdb->received);

    packets_dropped_this_second  = udpdb->packets_dropped - 
                                   udpdb->packets_dropped_last_sec;
    udpdb->packets_dropped_last_sec = udpdb->packets_dropped;

    uint64_t bytes_received_this_second = udpdb->bytes_received -
                                   udpdb->bytes_received_last_sec;
    udpdb->bytes_received_last_sec = udpdb->bytes_received;

    if (udpdb->verbose) { 

      multilog(udpdb->statslog, LOG_INFO,"npacks dropped: %"PRIu64", "
             "npackets: %"PRIu64", nbytes: %"PRIu64", nbytes_ps: %"PRIu64"\n",
             packets_dropped_this_second, udpdb->packets_received, 
             udpdb->bytes_received, bytes_received_this_second);

    } else {

      multilog(udpdb->statslog, LOG_INFO,"%"PRIu64"\n",
             packets_dropped_this_second);
    }

    if (udpdb->packets_late_this_sec) {
      multilog(log, LOG_WARNING,"%"PRIu64" packets arrvived late\n",
                                       packets_dropped_this_second);
      udpdb->packets_late_this_sec = 0;
    }
  
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
    percent_dropped = (float) ((double)udpdb->packets_dropped_this_run / (double)udpdb->expected_sequence_no) * 100;
  }

  multilog(pwcm->log, LOG_INFO, "packets dropped %"PRIu64" / %"PRIu64" = %10.8f %\n",
           udpdb->packets_dropped_this_run, udpdb->expected_sequence_no, percent_dropped);
  
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
  
  /* port on which to listen for incoming connections */
  int port = APSR_DEFAULT_UDPDB_PORT;

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

  while ((arg=getopt(argc,argv,"dp:vm:S:H:n:1h")) != -1) {
    switch (arg) {
      
    case 'd':
      daemon = 1;
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

  log = multilog_open ("apsr_udpdb", 0);

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
  pwcm->error_function  = udpdb_error_function;
  pwcm->header_valid_function  = udpdb_header_valid_function;
  pwcm->verbose = verbose;

  /* Setup context information */
  udpdb.verbose = verbose;
  udpdb.port = port;
  udpdb.packets_dropped = 0;
  udpdb.packets_dropped_this_run = 0;
  udpdb.packets_dropped_last_sec = 0;
  udpdb.packets_received = 0;
  udpdb.packets_received_this_run = 0;
  udpdb.packets_received_last_sec= 0;
  udpdb.bytes_received = 0;
  udpdb.bytes_received_last_sec = 0;
  udpdb.dropped_packets_to_fill = 0;
  udpdb.packet_in_buffer = 0;
  udpdb.received = 0;
  udpdb.prev_time = time(0);
  udpdb.current_time = udpdb.prev_time;
  udpdb.error_seconds = 10;
  udpdb.packet_length = 0;
  udpdb.statslog = multilog_open ("apsr_udpdb_stats", 0);
  multilog_serve (udpdb.statslog, APSR_DEFAULT_UDPDB_STATS);
    
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

      if (ascii_header_get (header_buf, "NBAND", "%d", 
                            &(udpdb->expected_nbands)) != 1) {
        multilog (pwcm->log, LOG_WARNING, "Warning. No NBAND in header, using "
                                          "1 as default\n");
        ascii_header_set (header_buf, "NBAND", "%d", 1);
      }

      if (ascii_header_get (header_buf, "NBIT", "%d", 
                            &(udpdb->expected_header_bits)) != 1) {
        multilog (pwcm->log, LOG_WARNING, "Warning. No NBIT in header, using "
                                          "8 as default\n");
        ascii_header_set (header_buf, "NBIT", "%d", 8);
      }

      if (ascii_header_get (header_buf, "NCHAN", "%d",
                            &(udpdb->expected_nchannels)) != 1) {
        multilog (pwcm->log, LOG_WARNING, "Warning. No NCHAN in header, "
                                          "using 1 as default\n");
        ascii_header_set (header_buf, "NCHAN", "%d", 1);
      }

      if (verbose) fprintf(stderr,"Retrieved header information\n");
      
    } 
    else {

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



int create_udp_socket(dada_pwc_main_t* pwcm) {

  udpdb_t* udpdb = pwcm->context;
  udpdb->fd = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);

  const int std_buffer_size = (128*1024) - 1;   // stanard kernel buffer of 128KB
  const int pref_buffer_size = 64*1024*1024;    // socket buffer of 64MB

  if (udpdb->fd < 0) {
    multilog(pwcm->log ,LOG_ERR, "Could not created UDP socket: %s\n",
             strerror(errno));
    return -1;
  }

  struct sockaddr_in udp_sock;
  bzero(&(udp_sock.sin_zero), 8);                     // clear the struct
  udp_sock.sin_family = AF_INET;                      // internet/IP
  udp_sock.sin_port = htons(udpdb->port);             // set the port number
  udp_sock.sin_addr.s_addr = htonl(INADDR_ANY);       // receive from any ip
                                                                                                                
  if (bind(udpdb->fd, (struct sockaddr *)&udp_sock, sizeof(udp_sock)) == -1) {
    multilog (pwcm->log, LOG_ERR, "Error binding UDP socket: %s\n",
              strerror(errno));
    return -1;
  }

  // try setting the buffer to the maximum, warn if we cant
  int len = 0;
  int value = pref_buffer_size;
  int retval = 0;
  len = sizeof(value);
  retval = setsockopt(udpdb->fd, SOL_SOCKET, SO_RCVBUF, &value, len);
  if (retval != 0) {
    perror("setsockopt SO_RCVBUF");
    return -1;
  }

  // now check if it worked....
  len = sizeof(value);
  value = 0;
  retval = getsockopt(udpdb->fd, SOL_SOCKET, SO_RCVBUF, &value, 
                      (socklen_t *) &len);
  if (retval != 0) {
    perror("getsockopt SO_RCVBUF");
    return -1;
  }

  // If we could not set the buffer to the desired size, warn...
  if (value/2 != pref_buffer_size) {
    multilog (pwcm->log, LOG_WARNING, "Warning. Failed to set udp socket's "
              "buffer size to: %d, falling back to default size: %d\n",
              pref_buffer_size, std_buffer_size);

    len = sizeof(value);
    value = std_buffer_size;
    retval = setsockopt(udpdb->fd, SOL_SOCKET, SO_RCVBUF, &value, len);
    if (retval != 0) {
      perror("setsockopt SO_RCVBUF");
      return -1;
    }

    // Now double check that the buffer size is at least correct here
    len = sizeof(value);
    value = 0;
    retval = getsockopt(udpdb->fd, SOL_SOCKET, SO_RCVBUF, &value, 
                        (socklen_t *) &len);
    if (retval != 0) {
      perror("getsockopt SO_RCVBUF");
      return -1;
    }
                                                                                                                
    // If we could not set the buffer to the desired size, warn...
    if (value/2 != std_buffer_size) {
      multilog (pwcm->log, LOG_WARNING, "Warning. Failed to set udp socket's "
                "buffer size to: %d\n", std_buffer_size);
    }
  }
  return 0;
}

void print_udpbuffer(char * buffer, int buffersize) {

  int i = 0;
  fprintf(stdout,"udp_packet_buffer[");
  for (i=0; i < buffersize; i++) {
    fprintf(stdout,"%d",buffer[i]);
  }
  fprintf(stdout,"]\n");
        
}

void check_udpdata(char * buffer, int buffersize, int value) {
                                                                                
  int i = 0;
  char c;
  c = (char) value;
  
  for (i=0; i < buffersize; i++) {
    if (c != buffer[i]){ 
      fprintf(stderr,"%d: [%d] != [%d]\n",i,buffer[i],c);
      i = buffersize;
    }
  }
                                                                                
}

void check_header(header_struct header, udpdb_t *udpdb, multilog_t* log) {

  if (header.bits != udpdb->expected_header_bits) {
    multilog (log, LOG_WARNING, "Packet %"PRIu64": expected %d "
              "nbits per sample, received %d\n", header.sequence, 
              udpdb->expected_header_bits, header.bits);
  }

  if (header.channels != udpdb->expected_nchannels) {
    multilog (log, LOG_WARNING, "Packet %"PRIu64": expected %d "
              "channels, received %d\n", header.sequence,
              udpdb->expected_nchannels, header.channels);
  }
                                                                            
  if (header.bands != udpdb->expected_nbands) {
    multilog (log, LOG_WARNING, "Packet %"PRIu64": expected %d "
              "bands, received %d\n", header.sequence, udpdb->expected_nbands,
              header.bands);
  }
                                                                            
  if (header.length != UDPHEADERSIZE) {
    multilog (log, LOG_WARNING, "Packet %"PRIu64": expected a header "
              "of length %d, received %d\n", header.sequence, UDPHEADERSIZE,
              header.length);
  }
}


