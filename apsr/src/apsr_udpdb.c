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
     " -S num         file size in bytes\n",DADA_DEFAULT_UDPDB_PORT);
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
  if (udpdb->expected_sequence_no > 1) {
    sleep(1);
    udpdb->error_seconds--;
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
  udpdb->packets_dropped_this_run = 0;
  udpdb->packets_received_this_run = 0;
  udpdb->bytes_received_this_run = 0;
  udpdb->error_seconds = 10;
  udpdb->dropped_packets_to_fill = 0;
  udpdb->buffer_full = 0;
  udpdb->prev_time = time(0);
  udpdb->current_time = udpdb->prev_time;
  
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
 
  if (udpdb->check_udp_headers) { 

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
                                                                                
    if (ascii_header_get (pwcm->header, "NCHANNEL", "%d",
                          &(udpdb->expected_nchannels)) != 1) {
      multilog (log, LOG_WARNING, "Warning. NCHANNEL not set in header "
                                  "Using 2 as default.\n");
      ascii_header_set (pwcm->header, "NCHANNEL", "%d", 2);
    }
  }
  
  /* setup the data buffer for NUMUDPACKETS udp packets */
  udpdb->datasize = 64 * 1024 * 1024;
  udpdb->data = (char *) malloc(sizeof(char) * udpdb->datasize);
  assert(udpdb->data != 0);

  udpdb->buffer= (char *) malloc(sizeof(char) * UDPBUFFSIZE);
  assert(udpdb->buffer != 0);

  /* setup the expected sequence no to the initial value */
  udpdb->expected_sequence_no = 1;

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

  /* Current packet rate per second */
  uint64_t packets_per_second = 0;

  /* Current byte rate per secon */
  uint64_t bytes_per_second = 0;

  /* Useful data received */
  uint64_t obs_data_received = 0;
  
  uint64_t i = 0;
  uint64_t buffer_counter = 0;      // How full the data 
  header_struct header = { '0', '0', 0, '0', '0', '0',"0000", 0 };

  // A char with all bits set to 0
  char zerodchar = 'c'; 
  memset(&zerodchar,0,sizeof(zerodchar));

  int verbose = udpdb->verbose;
  int quit = 0;
  struct timeval timeout;
  struct timeval* timeoutp;
  fd_set *rdsp = NULL;
  fd_set readset;

  /* Check if no empty buffers */
  //int bufs_clear = semctl (((ipcbuf_t *)(pwcm->data_block))->semid, 5, GETVAL);
  //if (bufs_clear == 0) {
  //  if (pwcm->pwc->state == dada_pwc_recording)
  //    multilog (log, LOG_WARNING, "Warning. no clear blocks available\n");
  //}

  /* keep receiving packets until the data buffer will not take another
     full packet */
  for (i=0; (!(quit) && ((buffer_counter+UDPBUFFSIZE) < udpdb->datasize));i++) {

    /* if we have missed udp packet(s), we need to send zero'd packets */
    if (udpdb->dropped_packets_to_fill > 0) {

      /* Fill as many empty packets as we can*/
      int nfills = (int) floor(
                    (float) (udpdb->datasize - 
                   (buffer_counter + udpdb->packet_length)) / 
                      ((float) (udpdb->packet_length)));

      nfills = (nfills > 1) ? nfills : 1; // Max comparre
      nfills = (nfills < udpdb->dropped_packets_to_fill ? nfills : 
               udpdb->dropped_packets_to_fill);
            
      obs_data_received = udpdb->packet_length*nfills;
      memset((udpdb->data)+buffer_counter,zerodchar,obs_data_received);
      buffer_counter += obs_data_received;
      udpdb->dropped_packets_to_fill -= nfills;
      udpdb->expected_sequence_no += nfills;

    } else {
  
      /* If we had a packet in the buffer from a dropped packet seqeunce */ 
      if (udpdb->buffer_full) {
        udpdb->buffer_full = 0;

      /* else get a fresh packet */
      } else {

        /* select to see if data has arrive */

        /* whilst, waiting for the first packet, have an incredibly short time
         * timeout, so that the pwc_main isn't "locked" by waiting */      
        /* In recording state, we allow a 1 second timeout */
        if (pwcm->pwc->state == dada_pwc_recording) {
          timeout.tv_sec=1;
          timeout.tv_usec=0;
        } else {
          timeout.tv_sec=0;
          timeout.tv_usec=100000;
        }

        timeoutp = &timeout;

        FD_ZERO (&readset);
        FD_SET (udpdb->fd, &readset);
        rdsp = &readset;

        int rval = select((udpdb->fd+1),rdsp,NULL,NULL,timeoutp);

        if (rval == 0) {
          if (pwcm->pwc->state == dada_pwc_recording) {
            multilog (log, LOG_ERR, "Error. UDP packet timeout: no packet"
                      " received for 1 second\n");
          } 

          quit = 1;
          udpdb->received = 0;

        } else {

          udpdb->received = sock_recv (udpdb->fd,udpdb->buffer,UDPBUFFSIZE,0);

          /* The length of all packets is defined to be the length of the 
           * first packet... (excluding header which is hardcoded*/
          if (udpdb->expected_sequence_no == 1) 
            udpdb->packet_length = udpdb->received - UDPHEADERSIZE;
    
          /*
          if (verbose) fprintf(stderr,"udpdb_buffer_function (%"PRIu64") udp "
                             "pack received, %"PRIu64"\n",i,udpdb->received);
          */
        }
      }

      
      if (udpdb->received != (UDPHEADERSIZE + udpdb->packet_length)) {

        /* If we are checking all the header values, then we should be using 
         * the DFB3, hence a small packet (which isn't the last packet) must
         * be erroneous. hence pad the packet with 0's...*/
        if (udpdb->check_udp_headers) {

          if (!quit) {
            multilog (log, LOG_ERR, "Error. Received %"PRIu64" bytes from udp "
                      "socket. Expected %d. Padding missing bytes with 0's\n",
                      udpdb->received, (UDPHEADERSIZE + udpdb->packet_length));

            char zerodchar = 'c';
            memset(&zerodchar,0,sizeof(zerodchar));
            memset((udpdb->buffer+udpdb->received),zerodchar,
                   (UDPHEADERSIZE + udpdb->packet_length)-udpdb->received);

            /* now adjust the "received value to what it should have been */
            udpdb->received = (UDPHEADERSIZE + udpdb->packet_length);
          }

        } else {

          multilog (log, LOG_WARNING, "Warning. Received %"PRIu64" bytes "
                    "from udp socket. Expected %d. Assuming its a deliberately "
                    "shortened packet\n",udpdb->received,
                    (UDPHEADERSIZE + udpdb->packet_length));

        }
      }

      /* If we have received any useful data */
      if (udpdb->received > UDPHEADERSIZE) { 
        obs_data_received = udpdb->received - UDPHEADERSIZE;

        decode_header(udpdb->buffer, &header);
        header.pollength = ((udpdb->received - header.length) * 4) / (header.bits*2);

#ifdef _DEBUG
        print_header(&header);
        print_udpbuffer(udpdb->buffer+header.length,received-header.length);
#endif
      
        // The sequence number is meant to only revert once every 6.3 days
        // just log a warning in case this is unexpected 
        if (udpdb->expected_sequence_no > header.sequence) {
           multilog (log, LOG_WARNING, "Warning. packet sequence number "
                     "has reverted from %d to %d\n", 
                     udpdb->expected_sequence_no, header.sequence);
           udpdb->expected_sequence_no = header.sequence;
        }

        /* We have lost 1 or more packets */
        if (udpdb->expected_sequence_no < header.sequence) {

          udpdb->packets_dropped += (header.sequence - 
                                     udpdb->expected_sequence_no);
          udpdb->packets_dropped_this_run += (header.sequence -
                                              udpdb->expected_sequence_no);

          /* we will now need to send zero'd packets in place of lost ones */
          assert(udpdb->dropped_packets_to_fill == 0);
          udpdb->dropped_packets_to_fill = (header.sequence - 
                                          udpdb->expected_sequence_no);

          /* the buffer contains a packet, but must be sent after zeros */
          udpdb->buffer_full = 1; 

          multilog (log, LOG_WARNING, "Warning. %d packets dropped (%d "
                    "to %d)\n",(header.sequence - udpdb->expected_sequence_no),
                    udpdb->expected_sequence_no,header.sequence);

          // We now need to pad the appropriate amount of 0's into the 
          // data block. This depends on the number of packets actually lost
          // and the size of the packets

        } else {

          memcpy((udpdb->data)+buffer_counter, (udpdb->buffer)+UDPHEADERSIZE,
                 obs_data_received);

          buffer_counter += obs_data_received;

          /* Increment statistics for reporting */
          udpdb->packets_received++;
          udpdb->packets_received_this_run++;

          udpdb->bytes_received += obs_data_received;
          udpdb->bytes_received_this_run += obs_data_received;
    
          udpdb->expected_sequence_no++;      // Setup next sequence

          if (udpdb->check_udp_headers) 
            check_header(header, udpdb, pwcm->log);

        }
      } else {
        //fprintf(stderr,"Received a packet of less than %d bytes\n",UDPHEADERSIZE);
      }
    }
  }
  
  udpdb->prev_time = udpdb->current_time;
  udpdb->current_time = time(0);
  
  if (udpdb->prev_time != udpdb->current_time) {

          
    packets_per_second = udpdb->packets_received - 
                         udpdb->packets_received_last_sec;

    /*
    if (verbose) 
      fprintf(stderr,"packets:  %"PRIu64"  %"PRIu64" %"PRIu64"\n",
              udpdb->packets_received, udpdb->packets_received_last_sec, 
              packets_per_second);
              */

    bytes_per_second = udpdb->bytes_received - udpdb->bytes_received_last_sec;

    udpdb->packets_received_last_sec = udpdb->packets_received;
    udpdb->bytes_received_last_sec = udpdb->bytes_received;
    
    multilog(udpdb->statslog, LOG_INFO, "%7.0fM, %"PRIu64", %7.0f GB, "
                    "%6.0f MB/s, %"PRIu64", %"PRIu64"\n",
             ((float) (udpdb->packets_received_this_run/1000000)),
             packets_per_second,
             ((float) udpdb->bytes_received_this_run/(1024*1024*1024)),
             ((float) bytes_per_second / (1024*1024)),
             udpdb->packets_dropped_this_run,udpdb->packets_dropped);

  }
  
  *size = buffer_counter;

  /* if (verbose) fprintf(stderr,"returning %"PRIu64" bytes\n",*size);  */

  assert(udpdb->data != 0);
  
  return (void *) udpdb->data;

}


int udpdb_stop_function (dada_pwc_main_t* pwcm)
{

/* get our context, contains all required params */
  udpdb_t* udpdb = (udpdb_t*)pwcm->context;
  int verbose = udpdb->verbose; /* saves some typing */
  
  if (verbose) fprintf(stderr,"Stopping udp transfer\n");

  close(udpdb->fd);
  free(udpdb->data);
  free(udpdb->buffer);

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
  int port = DADA_DEFAULT_UDPDB_PORT;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  int arg = 0;

  /* mode flag */
  int mode = 0; 

  /* mode flag */
  int onetime = 0; 

  /* header provided flag, default to check */
  int check_udp_headers = 1; 
  
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

  while ((arg=getopt(argc,argv,"dp:vm:cS:H:n:1h")) != -1) {
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

    case 'c':
      check_udp_headers = 0;
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

  log = multilog_open ("apsr_udpdb", daemon);

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
  pwcm->verbose = verbose;

  /* Setup context information */
  udpdb.verbose = verbose;
  udpdb.port = port;
  udpdb.check_udp_headers = check_udp_headers;
  udpdb.packets_dropped = 0;
  udpdb.packets_dropped_this_run = 0;
  udpdb.packets_received = 0;
  udpdb.packets_received_this_run = 0;
  udpdb.packets_received_last_sec= 0;
  udpdb.bytes_received = 0;
  udpdb.bytes_received_last_sec = 0;
  udpdb.dropped_packets_to_fill = 0;
  udpdb.buffer_full = 0;
  udpdb.received = 0;
  udpdb.prev_time = time(0);
  udpdb.current_time = udpdb.prev_time;
  udpdb.error_seconds = 10;
  udpdb.packet_length = 0;
  udpdb.statslog = multilog_open ("apsr_udpdb_stats", 0);
  multilog_serve (udpdb.statslog, DADA_DEFAULT_UDPDB_STATS);
    
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

      if (ascii_header_get (header_buf, "NCHANNEL", "%d",
                            &(udpdb->expected_nchannels)) != 1) {
        multilog (pwcm->log, LOG_WARNING, "Warning. No NCHANNEL in header, "
                                          "using 2 as default\n");
        ascii_header_set (header_buf, "NCHANNEL", "%d", 2);
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
    multilog (log, LOG_WARNING, "Warning. Packet %"PRIu64": expected %d "
              "nbits per sample, received %d\n", header.sequence, 
              udpdb->expected_header_bits, header.bits);
  }

  if (header.channels != udpdb->expected_nchannels) {
    multilog (log, LOG_WARNING, "Warning. Packet %"PRIu64": expected %d "
              "channels, received %d\n", header.sequence,
              udpdb->expected_nchannels, header.channels);
  }
                                                                            
  if (header.bands != udpdb->expected_nbands) {
    multilog (log, LOG_WARNING, "Warning. Packet %"PRIu64": expected %d "
              "bands, received %d\n", header.sequence, udpdb->expected_nbands,
              header.bands);
  }
                                                                            
  if (header.length != UDPHEADERSIZE) {
    multilog (log, LOG_WARNING, "Warning. Packet %"PRIu64": expected a header "
              "of length %d, received %d\n", header.sequence, UDPHEADERSIZE,
              header.length);
  }
}


