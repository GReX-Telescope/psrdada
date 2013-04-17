/***************************************************************************
 *  
 *    Copyright (C) 2012 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

/***************************************************************************
 *
 * hispec_udpdb
 * 
 * Primary Write Client for HISPEC backend
 *
 ****************************************************************************/

// #define _DEBUG 1

#include <math.h>

#include "hispec_def.h"
#include "hispec_udpdb.h"
#include "sock.h"

// Global quit flag for use with signal()
int quit_threads = 0;

void usage()
{
  fprintf (stdout,
	   "hispec_udpdb [options]\n"
     " -h             print help text\n"
     " -i address     ip address for UDP packets [default all]\n"
	   " -p port        port on which to listen [default %d]\n"
	   " -d             run as daemon\n"
	   " -k             hexidecimal shared memor key [default %x]\n"
     " -m mode        1 for independent mode, 2 for controlled mode\n"
     " -v             verbose messages\n"
     " -H filename    ascii header information in file\n"
     " -l port        multilog port to write logging output to [default %d]\n"
     " -c port        port to received pwcc commands on [default %d]\n",
     HISPEC_DEFAULT_UDPDB_PORT, DADA_DEFAULT_BLOCK_KEY,
     HISPEC_DEFAULT_PWC_LOGPORT, HISPEC_DEFAULT_PWC_PORT);
}


// Determine if the header is valid. Returns 1 if valid, 0 otherwise
int udpdb_header_valid_function (dada_pwc_main_t* pwcm)
{
  int utc_size = 64;
  char utc_buffer[utc_size];
  int valid = 1;

  // Check if the UTC_START is set in the header
  if (ascii_header_get (pwcm->header, "UTC_START", "%s", utc_buffer) < 0) {
    valid = 0;
  }

  // Check whether the UTC_START is set to UNKNOWN 
  if (strcmp(utc_buffer,"UNKNOWN") == 0)
    valid = 0;

#ifdef _DEBUG
  multilog(pwcm->log, LOG_INFO, "Checking if header is valid: %d\n", valid);
#endif

  return valid;
}

// Error function
int udpdb_error_function (dada_pwc_main_t* pwcm) 
{
  hispec_udpdb_t * udpdb = (hispec_udpdb_t*) pwcm->context;

  // If UTC_START has been received, the buffer function 
  // should be returning data
  if (udpdb_header_valid_function (pwcm) == 1) 
  {
    udpdb->error_seconds--;
    if (udpdb->error_seconds <= 0)
      return 2;
    else
      return 1;
  }
  else
    return 0;
}

//
// Initialise resources
//
int hispec_udpdb_init (hispec_udpdb_t* udpdb)
{
  fprintf (stderr, "hispec_udpdb_init()\n");

  // setup the data optimal transfer size
  udpdb->datasize = HISPEC_UDP_DATASIZE_BYTES * HISPEC_NUM_UDP_PACKETS;
  fprintf (stderr, "hispec_udpdb_init: datasize=%"PRIu64"\n", udpdb->datasize);

  // initialize first buffer
  if (udpdb->curr_buffer)
    free (udpdb->curr_buffer);
  udpdb->curr_buffer = (char *) malloc(sizeof(char) * udpdb->datasize);
  assert(udpdb->curr_buffer != 0);

  // initialize second buffer
  if (udpdb->next_buffer)
    free (udpdb->next_buffer);
  udpdb->next_buffer = (char *) malloc(sizeof(char) * udpdb->datasize);
  assert(udpdb->next_buffer != 0);

  // initialize socket buffer for receiveing 1 UDP packet
  if (udpdb->socket_buffer)
    free (udpdb->socket_buffer);
  udpdb->socket_buffer= (char *) malloc(sizeof(char) * HISPEC_UDP_PAYLOAD_BYTES);
  assert(udpdb->socket_buffer != 0);

  // 0 both the curr and next buffers
  char zerodchar = 'c';
  memset(&zerodchar,0,sizeof(zerodchar));
  memset(udpdb->curr_buffer,zerodchar,udpdb->datasize);
  memset(udpdb->next_buffer,zerodchar,udpdb->datasize);

  return 0;
}


//
// Start function, called at start of data
//
time_t udpdb_start_function (dada_pwc_main_t* pwcm, time_t start_utc)
{
  // get our context, contains all required params
  hispec_udpdb_t* udpdb = (hispec_udpdb_t*)pwcm->context;
  
  multilog_t* log = pwcm->log;

  // Initialise variables
  udpdb->packets->received = 0;
  udpdb->packets->dropped = 0;
  udpdb->packets->received_per_sec = 0;
  udpdb->packets->dropped_per_sec = 0;

  udpdb->bytes->received = 0;
  udpdb->bytes->dropped = 0;
  udpdb->bytes->received_per_sec = 0;
  udpdb->bytes->dropped_per_sec = 0;

  udpdb->error_seconds = 5;
  udpdb->packet_in_buffer = 0;
  udpdb->prev_time = time(0);
  udpdb->current_time = udpdb->prev_time;
  udpdb->curr_buffer_count = 0;
  udpdb->next_buffer_count = 0;
  
  // setup the expected sequence no to the initial value
  udpdb->sequence_incr = 512 * 25;
  udpdb->expected_sequence_no = 0;
  udpdb->prev_seq = 1;

  // Set the current machines name in the header block as RECV_HOST 
  char myhostname[HOST_NAME_MAX] = "unknown";;
  gethostname(myhostname,HOST_NAME_MAX); 
  ascii_header_set (pwcm->header, "RECV_HOST", "%s", myhostname);
  
  // create a udp socket for recv
  multilog (log, LOG_INFO, "udpdb_start_function: listening on %s:%d\n", udpdb->interface, udpdb->port);
  udpdb->fd = dada_udp_sock_in (log, udpdb->interface, udpdb->port, udpdb->verbose);
  if (udpdb->fd < 0)
  {
    multilog (log, LOG_ERR, "udpdb_start_function: failed to create udp socket\n");
    return 0; // n.b. this is an error value 
  }

  // try to increase the UDP socket buffer size to 64 MB (may likely fail)
  int rval = dada_udp_sock_set_buffer_size(log, udpdb->fd, udpdb->verbose, 67108864);
  if (rval < 0)
    multilog (log, LOG_WARNING, "failed in increase UDP socket buffer size to 64 MB\n");

  time_t utc = 0;
  return utc;
}

void* udpdb_buffer_function (dada_pwc_main_t* pwcm, int64_t* size)
{
  
  hispec_udpdb_t* udpdb = (hispec_udpdb_t*)pwcm->context;

  multilog_t* log = pwcm->log;

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

  uint64_t prevnum = udpdb->curr_sequence_no;

  // switch the next and current buffers
  char * tmp = udpdb->curr_buffer;
  udpdb->curr_buffer = udpdb->next_buffer;
  udpdb->next_buffer = tmp;
  
  // switch the buffer counters
  udpdb->curr_buffer_count = udpdb->next_buffer_count;
  udpdb->next_buffer_count = 0;

  // 0 the next buffer 
  memset(udpdb->next_buffer, zerodchar, udpdb->datasize);

  // Determine the sequence number boundaries for curr and next buffers
  udpdb->min_sequence = udpdb->expected_sequence_no;
  udpdb->mid_sequence = udpdb->min_sequence + HISPEC_NUM_UDP_PACKETS;
  udpdb->max_sequence = udpdb->mid_sequence + HISPEC_NUM_UDP_PACKETS;

  if (udpdb->verbose)
    multilog (log, LOG_INFO, "seq range [%"PRIu64" - %"PRIu64" - %"PRIu64"] datasize=%"PRIu64"\n",
             udpdb->min_sequence, udpdb->mid_sequence, udpdb->max_sequence, udpdb->datasize);

  // Assume we will be able to return a full buffer
  *size = (int64_t) udpdb->datasize;
  int64_t max_ignore = udpdb->datasize;

  // contains decoded header paramters
  hispec_udp_header_t pkt_hdr;

  // Continue to receive packets
  while (!quit) 
  {
    // If we had a packet in the socket buffer from a previous call to the buffer function
    if (udpdb->packet_in_buffer) 
    {
      udpdb->packet_in_buffer = 0;
    } 
    // Else try to get a fresh packet
    else 
    {

      // 1.0 second timeout for select()
      timeout.tv_sec=0;
      timeout.tv_usec=1000000;

      FD_ZERO (&readset);
      FD_SET (udpdb->fd, &readset);
      rdsp = &readset;

      if ( select((udpdb->fd+1),rdsp,NULL,NULL,&timeout) == 0 ) 
      {
        if ((pwcm->pwc) && (pwcm->pwc->state == dada_pwc_recording))
        {
          multilog (log, LOG_WARNING, "UDP packet timeout: no packet "
                                      "received for 1 second\n");
        }
        quit = 1;
        udpdb->received = 0;
        timeout_ocurred = 1;
      } 
      else 
      {
        // Get a packet from the socket
        udpdb->received = recvfrom (udpdb->fd, udpdb->socket_buffer, HISPEC_UDP_PAYLOAD_BYTES, 0, NULL, NULL);
        ignore_packet = 0;
      }
    }

    // If we did get a packet within the timeout, or one was in the buffer
    if (!quit) {

      decode_header((unsigned char *) udpdb->socket_buffer, &pkt_hdr);

      // Decode the packet's header
#ifdef TEST_MODE
      udpdb->curr_sequence_no = (int) pkt_hdr.pkt_cnt / udpdb->sequence_incr;
#else
      udpdb->curr_sequence_no = (uint64_t) pkt_hdr.pkt_cnt;
#endif

      if (udpdb->verbose > 1)
        multilog (log, LOG_INFO, "PKT: version=%d beam_id=%d, seq_no=%"PRIu64", diode_state=%d freq_state=%d\n", 
                  pkt_hdr.version, pkt_hdr.beam_id, udpdb->curr_sequence_no, pkt_hdr.diode_state, pkt_hdr.freq_state);

      //multilog (log, LOG_INFO, "seq=%"PRIu64"\n", udpdb->curr_sequence_no);

      // If we are waiting for the "first" packet
      if ((udpdb->expected_sequence_no == 0) && (data_received == 0)) 
      {

        //multilog (log, LOG_INFO, "waiting for start: seq=%"PRIu64"\n", udpdb->curr_sequence_no);

#ifdef _DEBUG
        if ((udpdb->curr_sequence_no < (udpdb->prev_seq - 10)) && (udpdb->prev_seq != 0)) {
          multilog(log, LOG_INFO, "packet num reset from %"PRIu64" to %"PRIu64"\n", udpdb->prev_seq, udpdb->curr_sequence_no);
        }
#endif
        udpdb->prev_seq = udpdb->curr_sequence_no;
 
        // Condition for detection of a restart
        if (udpdb->curr_sequence_no < udpdb->max_sequence)
        {
          if (udpdb->verbose)
            multilog (log, LOG_INFO, "start on packet %"PRIu64"\n",udpdb->curr_sequence_no);

          // Increment the buffer counts in case we have missed packets
          if (udpdb->curr_sequence_no < udpdb->mid_sequence)
            udpdb->curr_buffer_count += udpdb->curr_sequence_no;
          else 
          {
            udpdb->curr_buffer_count += HISPEC_NUM_UDP_PACKETS;
            udpdb->next_buffer_count += udpdb->curr_sequence_no - udpdb->mid_sequence;
          }
        } 
        else 
        {
          ignore_packet = 1;
        }

        if (udpdb->received != HISPEC_UDP_PAYLOAD_BYTES) {
          multilog (log, LOG_ERR, "UDP packet size was incorrect (%"PRIu64" != %d)\n", udpdb->received, HISPEC_UDP_PAYLOAD_BYTES);
          *size = DADA_ERROR_HARD;
          break;
        }
      }

      // If we are still waiting dor the start of data
      if (ignore_packet) {

        max_ignore -= HISPEC_UDP_PAYLOAD_BYTES;
        if (max_ignore < 0) 
          quit = 1;

      } else {

        // If the packet we received was too small, pad it
        if (udpdb->received < HISPEC_UDP_PAYLOAD_BYTES) {
        
          uint64_t amount_to_pad = HISPEC_UDP_PAYLOAD_BYTES - udpdb->received;
          char * buffer_pos = (udpdb->socket_buffer) + udpdb->received;
 
          // 0 the next buffer
          memset(buffer_pos, zerodchar, amount_to_pad);

          multilog (log, LOG_WARNING, "Short packet received, padded %"PRIu64
                                      " bytes\n", amount_to_pad);
        } 

        // If the packet we received was too long, warn about it
        if (udpdb->received > HISPEC_UDP_PAYLOAD_BYTES) 
        {
          multilog (log, LOG_WARNING, "Long packet received, truncated to %"PRIu64
                                      " bytes\n", HISPEC_UDP_DATASIZE_BYTES);
        }

        // Now try to slot the packet into the appropraite buffer
        data_received += HISPEC_UDP_DATASIZE_BYTES;

        // Increment statistics
        udpdb->packets->received++;
        udpdb->packets->received_per_sec++;
        udpdb->bytes->received += HISPEC_UDP_DATASIZE_BYTES;
        udpdb->bytes->received_per_sec += HISPEC_UDP_DATASIZE_BYTES;

        // If the packet belongs in the curr_buffer */
        if ((udpdb->curr_sequence_no >= udpdb->min_sequence) && 
            (udpdb->curr_sequence_no <  udpdb->mid_sequence)) 
        {

          uint64_t buf_offset = (udpdb->curr_sequence_no - udpdb->min_sequence) * HISPEC_UDP_DATASIZE_BYTES;
          memcpy( (udpdb->curr_buffer)+buf_offset, 
                  (udpdb->socket_buffer)+HISPEC_UDP_COUNTER_BYTES,
                  HISPEC_UDP_DATASIZE_BYTES);
          udpdb->curr_buffer_count++;
        } 
        else if ((udpdb->curr_sequence_no >= udpdb->mid_sequence) && 
                   (udpdb->curr_sequence_no <  udpdb->max_sequence)) 
        {
          uint64_t buf_offset = (udpdb->curr_sequence_no - udpdb->mid_sequence) * HISPEC_UDP_DATASIZE_BYTES;
          memcpy( (udpdb->curr_buffer)+buf_offset, 
                  (udpdb->socket_buffer)+HISPEC_UDP_COUNTER_BYTES,
                  HISPEC_UDP_DATASIZE_BYTES);
          udpdb->next_buffer_count++;
        } 
        // If this packet has arrived too late, it has already missed out
        else if (udpdb->curr_sequence_no < udpdb->min_sequence) 
        {
          multilog (log, LOG_WARNING, "Packet arrived too soon, %"PRIu64" < %"PRIu64"\n",
                    udpdb->curr_sequence_no, udpdb->min_sequence);

        } 
        // If a packet has arrived too soon, then we give up trying to fill the 
        // curr_buffer and return what we do have 
        else if (udpdb->curr_sequence_no >= udpdb->max_sequence) 
        {
          float curr_percent = ((float) udpdb->curr_buffer_count / (float) HISPEC_NUM_UDP_PACKETS)*100;
          float next_percent = ((float) udpdb->next_buffer_count / (float) HISPEC_NUM_UDP_PACKETS)*100;
          multilog (log, LOG_WARNING, "%"PRIu64" > %"PRIu64"\n",udpdb->curr_sequence_no,udpdb->max_sequence);
          multilog (log, LOG_WARNING, "Not keeping up. curr_buffer %5.2f%, next_buffer %5.2f%\n",
                                      curr_percent, next_percent);
          udpdb->packet_in_buffer = 1;
          quit = 1;
        } 
        else 
        {
          fprintf (stderr,"Sequence number invalid\n");
        }

        // If we have filled the current buffer, then we can stop
        if (udpdb->curr_buffer_count == HISPEC_NUM_UDP_PACKETS) 
        {
          quit = 1;
        } 
        else 
        {
          assert(udpdb->curr_buffer_count < HISPEC_NUM_UDP_PACKETS);
        }

        // If the next buffer is at least half full
        if (udpdb->next_buffer_count > (HISPEC_NUM_UDP_PACKETS / 2)) 
        {
          float curr_percent = ((float) udpdb->curr_buffer_count / (float) HISPEC_NUM_UDP_PACKETS)*100;
          float next_percent = ((float) udpdb->next_buffer_count / (float) HISPEC_NUM_UDP_PACKETS)*100;

          multilog(log, LOG_WARNING, "Bailing curr_buf %5.2f%, next_buffer %5.2f%\n",curr_percent,next_percent);
          quit = 1;
        }
      }
    } 
  }

  if (udpdb->verbose)
    multilog (log, LOG_INFO, "curr: %"PRIu64", next: %"PRIu64", capacity: %"PRIu64"\n",udpdb->curr_buffer_count, udpdb->next_buffer_count, HISPEC_NUM_UDP_PACKETS);

  // If we have received a packet during this function call
  if (data_received) 
  {
    // If we have not received all the packets we expected
    if ((udpdb->curr_buffer_count < HISPEC_NUM_UDP_PACKETS) && (!timeout_ocurred)) 
    {
      multilog (log, LOG_WARNING, "Dropped %"PRIu64" packets\n",
               (HISPEC_NUM_UDP_PACKETS - udpdb->curr_buffer_count));

      udpdb->packets->dropped += (HISPEC_NUM_UDP_PACKETS - udpdb->curr_buffer_count);
      udpdb->packets->dropped_per_sec += (HISPEC_NUM_UDP_PACKETS - udpdb->curr_buffer_count);
      udpdb->bytes->dropped += (HISPEC_UDP_DATASIZE_BYTES * (HISPEC_NUM_UDP_PACKETS - udpdb->curr_buffer_count));
      udpdb->bytes->dropped_per_sec += (HISPEC_UDP_DATASIZE_BYTES * (HISPEC_NUM_UDP_PACKETS - udpdb->curr_buffer_count));

    }

    /* If the timeout ocurred, this is most likely due to end of data */
    if (timeout_ocurred) 
    {
      *size = udpdb->curr_buffer_count * HISPEC_UDP_DATASIZE_BYTES;
      multilog (log, LOG_WARNING, "Suspected EOD received, returning "
                     "%"PRIi64" bytes\n",*size);
    }

    udpdb->expected_sequence_no += HISPEC_NUM_UDP_PACKETS;

  } else {

    /* If we have received no data, then return a size of 0 */
    *size = 0;

  }

  udpdb->prev_time = udpdb->current_time;
  udpdb->current_time = time(0);
  
  if (udpdb->prev_time != udpdb->current_time) {

    if (udpdb->verbose) {

      if (ignore_packet) {

        multilog(log, LOG_INFO, "Ignoring out of range sequence no: %"PRIu64"\n",udpdb->curr_sequence_no);

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

//
// called at the end of an observation
//
int udpdb_stop_function (dada_pwc_main_t* pwcm)
{
  hispec_udpdb_t* udpdb = (hispec_udpdb_t*) pwcm->context;

  float percent_dropped = 0;
  if (udpdb->expected_sequence_no)
    percent_dropped = (float) ((double)udpdb->packets->dropped / (double)udpdb->expected_sequence_no) * 100;

  if (udpdb->packets->dropped) 
    multilog(pwcm->log, LOG_INFO, "packets dropped %"PRIu64" / %"PRIu64
             " = %8.6f %\n", udpdb->packets->dropped, 
             udpdb->expected_sequence_no, percent_dropped);
  
  if (udpdb->verbose) 
    fprintf(stderr, "udpdb_stop_function: closing UDP socket\n");

  if (udpdb->fd)
    close (udpdb->fd);
  udpdb->fd = 0;

  return 0;
}

int hispec_udpdb_destroy (hispec_udpdb_t * udpdb)
{
  if (udpdb->curr_buffer)
    free(udpdb->curr_buffer);
  udpdb->curr_buffer = 0;

  if (udpdb->next_buffer)
    free(udpdb->next_buffer);
  udpdb->next_buffer = 0;

  if (udpdb->socket_buffer)
    free(udpdb->socket_buffer);
  udpdb->socket_buffer = 0;
}


int main (int argc, char **argv)
{
  // DADA Primary Write Client
  dada_pwc_main_t* pwcm = 0;

  // DADA Header plus Data Unit
  dada_hdu_t* hdu = 0;

  // DADA Logger
  multilog_t* log = 0;
  
  // IP address on which to listen for udp packets
  char * interface = "any";

  // port on which to listen for udp packets
  int port = HISPEC_DEFAULT_UDPDB_PORT;

  // pwcc control / command port
  int c_port = HISPEC_DEFAULT_PWC_PORT;

  // multilog output port
  int l_port = HISPEC_DEFAULT_PWC_LOGPORT;

  // daemon mode flag
  char daemon = 0;

  // verbose mode flag
  char verbose = 0;

  // control / independent mode flag
  int mode = 0;

  // size of the header buffer
  uint64_t header_size = 0;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;

  // actual struct with info
  hispec_udpdb_t udpdb;

  /* the filename from which the header will be read */
  char* header_file = 0;
  char* header_buf = 0;

  char *src;

  while ((arg=getopt(argc,argv,"c:dhH:k:di:p:vm:n:c:l:")) != -1)
  {
    switch (arg)
    {
      case 'c':
        if (optarg)
        {
          c_port = atoi(optarg);
          break;
        } 
        else 
        {
          usage();
          return EXIT_FAILURE;
        }

      case 'd':
        daemon = 1;
        break;

      case 'h':
        usage();
        return EXIT_SUCCESS;



    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr,"hispec_udpdb: could not parse key from %s\n",optarg);
        return -1;
      }
      break;
      
    case 'i':
      if (optarg)
        interface = optarg;
      break;

    case 'p':
      port = atoi (optarg);
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

    case 'H':
      if (optarg) {
        header_file = optarg;
      } else {
        fprintf(stderr,"Specify a header file\n");
        usage();
      }
      break;

    case 'l':
      if (optarg) {
        l_port = atoi(optarg);
        break;
      } else {
        usage();
        return EXIT_FAILURE;
      }

    default:
      usage ();
      return EXIT_FAILURE;
      
    }
  }

  log = multilog_open ("hispec_udpdb", 0);

  if (daemon) 
    be_a_daemon ();
  else
    multilog_add (log, stderr);

  multilog_serve (log, l_port);
  
  if (verbose) 
    multilog (log, LOG_INFO, "creating dada_pwc_main\n");
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
  udpdb.mode = mode;
  udpdb.ibob_host = NULL;
  udpdb.ibob_port = 0;

  /* init stats structs */
  stats_t packets = {0,0,0,0};
  stats_t bytes = {0,0,0,0};

  udpdb.curr_buffer = 0;
  udpdb.next_buffer = 0;
  udpdb.socket_buffer = 0;

  udpdb.packets = &packets;
  udpdb.bytes = &bytes;

  udpdb.packet_in_buffer = 0;
  udpdb.received = 0;
  udpdb.prev_time = time(0);
  udpdb.current_time = udpdb.prev_time;
  udpdb.error_seconds = 5;

  if (verbose) 
    multilog (log, LOG_INFO, "initialize resources\n");
  if (hispec_udpdb_init (&udpdb) < 0)
  {
    multilog (log, LOG_ERR, "main: hispec_udpdb_init failed\n");
    return EXIT_FAILURE;
  }
    
  // Connect to shared memory
  if (verbose) 
    multilog (log, LOG_INFO, "connecting to shared memory\n");
  hdu = dada_hdu_create (pwcm->log);
  if (!hdu)
  {
    multilog (log, LOG_ERR, "failed to create HDU struct\n");
    hispec_udpdb_destroy (&udpdb);
    return EXIT_FAILURE;
  }

  // set the hexidecimal identifier
  dada_hdu_set_key(hdu, dada_key);

  // connect to the HDU
  if (dada_hdu_connect (hdu) < 0)
  {
    multilog (log, LOG_ERR, "failed to connect to the HDU\n");
    hispec_udpdb_destroy (&udpdb);
    return EXIT_FAILURE;
  }

  if (dada_hdu_lock_write (hdu) < 0)
  {
    multilog (log, LOG_ERR, "failed to lock write on the data block\n");
    hispec_udpdb_destroy (&udpdb);
    return EXIT_FAILURE;
  }

  pwcm->data_block = hdu->data_block;
  pwcm->header_block = hdu->header_block;

  // We now need to setup the header information. for testing we can
  // accept local header info. In practise we will get this from
  // the control interface...
  if (mode == 1) 
  {
    char buffer[64];

    header_size = ipcbuf_get_bufsz (hdu->header_block);
    if (verbose)
      multilog (pwcm->log, LOG_INFO, "header block size = %llu\n", header_size);
    header_buf = ipcbuf_get_next_write (hdu->header_block);
    pwcm->header = header_buf;

    if (!header_buf)
    {
      multilog (pwcm->log, LOG_ERR, "Could not get next header block\n");
      return EXIT_FAILURE;
    }

    if (!header_file)
    {
      multilog (pwcm->log, LOG_ERR, "header file was not specified with -H\n");
      return EXIT_FAILURE;
    }

    if (verbose)
      multilog (pwcm->log, LOG_INFO, "read header file %s\n", header_file);

    if (fileread (header_file, header_buf, header_size) < 0)
    {
      multilog (pwcm->log, LOG_ERR, "could not read header from %s\n", header_file);
      return EXIT_FAILURE;
    }

    if (verbose) 
      multilog (pwcm->log, LOG_INFO, "retrieved header information\n");

    if (verbose) 
      multilog (pwcm->log, LOG_INFO, "running start function\n");
    time_t utc = udpdb_start_function(pwcm,0);

    if (utc == -1)
      multilog(pwcm->log, LOG_ERR, "Could not run start function\n");

    if (utc == 0)
    {
      multilog(pwcm->log, LOG_WARNING, "Overrideing UTC_START to current UTC\n");
      utc = time(0);
    }

    // just get the current UTC and set it as the UTC_START
    strftime (buffer, 64, DADA_TIMESTR, gmtime(&utc));

    // write UTC_START to the header
    if (ascii_header_set (header_buf, "UTC_START", "%s", buffer) < 0)
    {
      multilog (pwcm->log, LOG_ERR, "failed ascii_header_set UTC_START\n");
      return EXIT_FAILURE;
    }

    multilog (pwcm->log, LOG_INFO, "UTC_START %s written to header\n", buffer);

    // donot set header parameters anymore - acqn. doesn't start
    if (ipcbuf_mark_filled (hdu->header_block, header_size) < 0)
    {
      multilog (pwcm->log, LOG_ERR, "Could not mark filled header block\n");
      return EXIT_FAILURE;
    }

    if (verbose)
      multilog (pwcm->log, LOG_INFO, "header marked filled\n");

    // main loop, we just keep calling the buffer function until CTRL+C generates
    // a SIGINT or the packets timeout
    int64_t bsize = udpdb.datasize;
    int64_t total_bytes_received = 0;
    while (!quit_threads) 
    {
      // read the requested number of bytes from the data block
      src = (char *) udpdb_buffer_function(pwcm, &bsize);
      total_bytes_received += bsize;

      // check the number of bytes read
      if (udpdb.datasize != bsize)
      {
        multilog(pwcm->log, LOG_WARNING, "buffer function returned fewer bytes than requested, exiting\n");
        if (total_bytes_received)
        {
          multilog(pwcm->log, LOG_WARNING, "suspected UDP timeout, exiting\n");
          quit_threads = 1;
        }
      }

      // write data to datablock
      if (( ipcio_write(hdu->data_block, src, bsize) ) < bsize)
      {
        multilog(pwcm->log, LOG_ERR, "main: cannot write requested bytes to SHM\n");
        quit_threads = 1;
      }
    }

    if ( udpdb_stop_function(pwcm) != 0)
      fprintf(stderr, "Error stopping acquisition");

    if (dada_hdu_unlock_write (hdu) < 0)
      return EXIT_FAILURE;

    if (dada_hdu_disconnect (hdu) < 0)
      return EXIT_FAILURE;

  }
  // we are controlled by PWC control interface
  else 
  {
    pwcm->header = hdu->header;

    if (verbose) 
      fprintf (stderr, "main: creating dada pwc control interface\n");
    pwcm->pwc = dada_pwc_create();
    pwcm->pwc->port = c_port;

    if (verbose)
      fprintf (stderr, "main: creating dada server\n");
    if (dada_pwc_serve (pwcm->pwc) < 0) 
    {
      fprintf (stderr, "main: could not start pwc server\n");
      return EXIT_FAILURE;
    }

    if (verbose) 
      fprintf (stderr, "main: entering PWC main loop\n");
    if (dada_pwc_main (pwcm) < 0)
    {
      fprintf (stderr, "main: error in PWC main loop\n");
      return EXIT_FAILURE;
    }

    if (dada_hdu_unlock_write (hdu) < 0)
      return EXIT_FAILURE;
  
    if (dada_hdu_disconnect (hdu) < 0)
      return EXIT_FAILURE;

    if (verbose) 
      fprintf (stderr, "main: destroying pwc\n");
    dada_pwc_destroy (pwcm->pwc);

    if (verbose) 
      fprintf (stderr, "main: destroying pwc main\n");
    dada_pwc_main_destroy (pwcm);
  }

  // release resources
  hispec_udpdb_destroy (&udpdb);

  return EXIT_SUCCESS;

}


