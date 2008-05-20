/*
 * bpsr_udpheader
 *
 * Simply listens on a specified port for udp packets encoded
 * in the PARSEPC format
 *
 */

#include "bpsr_udpheader.h"

#include <sys/socket.h>
#include <math.h>

typedef struct {
  signed char r;
  signed char i;
} complex_char;

void usage()
{
  fprintf (stdout,
	   "bpsr_udpheader [options]\n"
     " -h             print help text\n"
	   " -p port        port on which to listen [default %d]\n"
     " -v             verbose messages\n"
     " -V             very verbose messages\n", BPSR_DEFAULT_UDPDB_PORT);
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

time_t udpheader_start_function (udpheader_t* udpheader, time_t start_utc)
{

  /* Initialise variables */
  udpheader->dropped_packets_to_fill = 0;
  udpheader->packet_in_buffer = 0;
  udpheader->prev_time = time(0);
  udpheader->current_time = udpheader->prev_time;
  udpheader->curr_buffer_count = 0;
  udpheader->next_buffer_count = 0;
  udpheader->packets_late_this_sec = 0;

  /* UDP SETUP */
  /* Create a udp socket */

  if (udpheader->verbose == 2) 
    fprintf(stderr, "udpheader_start_function: creating udp socket\n");        
          
  if (create_udp_socket(udpheader) < 0) {
    fprintf(stderr,"udpheader_start_function: Error, Failed to create udp socket\n");
    return 0; // n.b. this is an error value 
  }
  if (udpheader->verbose == 2) 
    fprintf(stderr, "udpheader_start_function: udp socket created\n");        

  /* Set the current machines name in the header block as RECV_HOST */
  char myhostname[HOST_NAME_MAX] = "unknown";;
  gethostname(myhostname,HOST_NAME_MAX); 

  /* setup the data buffer for NUMUDPACKETS udp packets */
  udpheader->datasize = 32000000;
  udpheader->curr_buffer = (char *) malloc(sizeof(char) * udpheader->datasize);
  assert(udpheader->curr_buffer != 0);
  udpheader->next_buffer = (char *) malloc(sizeof(char) * udpheader->datasize);
  assert(udpheader->next_buffer != 0);

  if (udpheader->verbose == 2) 
    fprintf(stderr, "udpheader_start_function: setting internal buffers to %"PRIu64" bytes\n",
                    udpheader->datasize);

  /* 0 both the curr and next buffers */
  char zerodchar = 'c';
  memset(&zerodchar,0,sizeof(zerodchar));
  memset(udpheader->curr_buffer,zerodchar,udpheader->datasize);
  memset(udpheader->next_buffer,zerodchar,udpheader->datasize);

  /* Setup the socket buffer for receiveing UDP packets */
  udpheader->socket_buffer= (char *) malloc(sizeof(char) * BPSR_UDP_PAYLOAD_BYTES);
  assert(udpheader->socket_buffer != 0);

  /* setup the expected sequence no to the initial value */
  udpheader->expected_sequence_no = 0;

  /* set the packet_length to an expected value */
  udpheader->packet_length = BPSR_UDP_DATASIZE_BYTES;

  return 0;
}

void* udpheader_read_function (udpheader_t* udpheader, uint64_t* size)
{
  
  /* Packets dropped in the last second */
  uint64_t packets_dropped_this_second = 0;
  uint64_t packets_received_this_second = 0;
  uint64_t bytes_received_this_second = 0;

  /* How many packets will fit in each curr_buffer and next_buffer */
  uint64_t buffer_capacity = udpheader->datasize / udpheader->packet_length;

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

  /* Switch the next and current buffers and their respective counters */
  char *tmp;
  tmp = udpheader->curr_buffer;
  udpheader->curr_buffer = udpheader->next_buffer;
  udpheader->next_buffer = tmp;
  udpheader->curr_buffer_count = udpheader->next_buffer_count;
  udpheader->next_buffer_count = 0;

  /* 0 the next buffer */
  memset(udpheader->next_buffer,zerodchar,udpheader->datasize);

  /* Determine the sequence number boundaries for curr and next buffers */
  uint64_t min_sequence = udpheader->expected_sequence_no;
  uint64_t mid_sequence = min_sequence + buffer_capacity;
  uint64_t max_sequence = mid_sequence + buffer_capacity;

  uint64_t sequence_no = 0;

  if (udpheader->verbose == 2) 
    fprintf(stderr, "[%"PRIu64" <-> %"PRIu64" <-> %"PRIu64"]\n",min_sequence,
                    mid_sequence,max_sequence); 

  /* Assume we will be able to return a full buffer */
  *size = buffer_capacity * udpheader->packet_length;

  /* Continue to receive packets */
  while (!quit) {

    /* If we had a packet in the socket_buffer a previous call to the 
     * buffer function*/
    if (udpheader->packet_in_buffer) {

      udpheader->packet_in_buffer = 0;
                                                                                
    /* Else try to get a fresh packet */
    } else {
                                                                                
      /* select to see if data has arrive. If recording, allow a 1 second
       * timeout, else 0.1 seconds to ensure buffer function is responsive */
      if (udpheader->state == RECORDING) {
        timeout.tv_sec=1;
        timeout.tv_usec=0;
      } else {
        timeout.tv_sec=60;
        timeout.tv_usec=0;
      }

      FD_ZERO (&readset);
      FD_SET (udpheader->udpfd, &readset);
      rdsp = &readset;

      if ( select((udpheader->udpfd+1),rdsp,NULL,NULL,&timeout) == 0 ) {

        if ((udpheader->state == RECORDING) &&
            (udpheader->expected_sequence_no != 0)) {
          fprintf(stderr, "Warning: UDP packet timeout: no packet "
                          "received for 1 second\n");
        }
        quit = 1;
        udpheader->received = 0;
        timeout_ocurred = 1;

      } else {

        /* Get a packet from the socket */
        udpheader->received = sock_recv (udpheader->udpfd, udpheader->socket_buffer,
                                         BPSR_UDP_PAYLOAD_BYTES, 0);

      }

    }

    /* If we did get a packet within the timeout, or one was in the buffer */
    if (!quit) {

      /* Decode the packets apsr specific header */
      sequence_no = decode_header(udpheader->socket_buffer);

      sequence_no /= 12800;

      if(udpheader->verbose >= 2) 
        fprintf(stderr, "%"PRIu64"\n", sequence_no);

      /* When we have received the first packet */      
      if ((udpheader->expected_sequence_no == 0) && (data_received == 0)) {

        udpheader->state = RECORDING;
        udpheader->expected_sequence_no = sequence_no;

        if (sequence_no != 0) {
          fprintf(stderr,"Warning: First packet received had sequence "
                         "number %"PRIu64"\n",sequence_no);
        } else {
          fprintf(stderr,"First packet received\n");
        }

        if (udpheader->verbose >= 1) {
          fprintf(stderr, "Packet Information\n========================================\n");
          fprintf(stderr, "packet: total\t= header + real data\n");
          fprintf(stderr, "packet: %"PRIu64"\t= %d\t + %"PRIu64"\n", udpheader->received, BPSR_UDP_COUNTER_BYTES, BPSR_UDP_DATASIZE_BYTES);
          fprintf(stderr, "buffer: %"PRIu64" packets = %"PRIu64" bytes\n",buffer_capacity, (buffer_capacity*udpheader->packet_length));
          fprintf(stderr, "========================================\n");
        }

      }

      /* If the packet we received was too small, pad it */
      if (udpheader->received < BPSR_UDP_PAYLOAD_BYTES) {

        uint64_t amount_to_pad = udpheader->received - BPSR_UDP_COUNTER_BYTES;
        char * buffer_pos = (udpheader->socket_buffer) + udpheader->received;

        /* 0 the next buffer */
        memset(buffer_pos, zerodchar, amount_to_pad);

        fprintf(stderr,"Warning: Short packet received, padded %"PRIu64" bytes\n", amount_to_pad);

      }

      /* If the packet we received was too long, warn about it */
      if (udpheader->received > BPSR_UDP_PAYLOAD_BYTES) {
        fprintf(stderr,"Warning: Long packet received, truncated to %"PRIu64
                       " bytes", udpheader->packet_length);
      }

      /* Now try to slot the packet into the appropraite buffer */
      data_received += BPSR_UDP_DATASIZE_BYTES;

      /* Increment statistics */
      udpheader->packets_received++;
      udpheader->bytes_received += BPSR_UDP_DATASIZE_BYTES;
 
      /* If the packet belongs in the curr_buffer */
      if ((sequence_no >= min_sequence) && 
          (sequence_no <  mid_sequence)) {

        uint64_t buf_offset = (sequence_no - min_sequence) * udpheader->packet_length;
  
        memcpy( (udpheader->curr_buffer)+buf_offset, 
                (udpheader->socket_buffer)+BPSR_UDP_COUNTER_BYTES,
                udpheader->packet_length);

        udpheader->curr_buffer_count++;


      } else if ((sequence_no >= mid_sequence) && 
                 (sequence_no <  max_sequence)) {

        uint64_t buf_offset = (sequence_no - mid_sequence) * udpheader->packet_length;

        memcpy( (udpheader->curr_buffer)+buf_offset, 
                (udpheader->socket_buffer)+BPSR_UDP_COUNTER_BYTES,
                udpheader->packet_length);

        udpheader->next_buffer_count++;

      /* If this packet has arrived too late, it has already missed out */
      } else if (sequence_no < min_sequence) {
 
        udpheader->packets_late_this_sec++; 

      /* If a packet has arrived too soon, then we give up trying to fill the 
         curr_buffer and return what we do have */
      } else if (sequence_no >= max_sequence) {

        fprintf(stderr, "Warning: Packet %"PRIu64" arrived too soon, expecting "
                        "packets %"PRIu64" through %"PRIu64"\n",
                        sequence_no, min_sequence, max_sequence);

        udpheader->packet_in_buffer = 0;
        quit = 1;

      } else {

        /* SHOULD NEVER HAPPEN */    
        fprintf (stderr,"ERROR header sequence number invalid\n");

      }

      /* If we have filled the current buffer, then we can stop */
      if (udpheader->curr_buffer_count == buffer_capacity) {
        quit = 1;
      } else {
        assert(udpheader->curr_buffer_count < buffer_capacity);
      }

      /* If the next buffer is at least half full */
      if (udpheader->next_buffer_count > (buffer_capacity / 2)) {
        quit = 1;
      }
    }

  } 

  /* If we have received a packet during this function call */
  if (data_received) {

    /* If we have not received all the packets we expected */
    if ((udpheader->curr_buffer_count < buffer_capacity) && (!timeout_ocurred)) {

      fprintf(stderr,"Warning: Dropped %"PRIu64" packets\n", 
              (buffer_capacity - udpheader->curr_buffer_count));

      udpheader->packets_dropped += buffer_capacity - udpheader->curr_buffer_count;
    }

    /* If the timeout ocurred, this is most likely due to end of data */
    if (timeout_ocurred) {
      *size = udpheader->curr_buffer_count * udpheader->packet_length;
      fprintf(stderr,"Warning: Suspected EOD received, returning %"PRIu64
                      " bytes\n",*size);
    }

    udpheader->expected_sequence_no += buffer_capacity;

  } else {

    /* If we have received no data, then return a size of 0 */
    *size = 0;

  }

  udpheader->prev_time = udpheader->current_time;
  udpheader->current_time = time(0);
  
  if (udpheader->prev_time != udpheader->current_time) {

    packets_dropped_this_second  = udpheader->packets_dropped - 
                                   udpheader->packets_dropped_last_sec;
    udpheader->packets_dropped_last_sec = udpheader->packets_dropped;

    packets_received_this_second  = udpheader->packets_received -
                                   udpheader->packets_received_last_sec;
    udpheader->packets_received_last_sec = udpheader->packets_received;

    bytes_received_this_second = udpheader->bytes_received -
                                   udpheader->bytes_received_last_sec;
    udpheader->bytes_received_last_sec = udpheader->bytes_received;

    if (udpheader->verbose >= 1) {
     
      fprintf(stderr, "Pkts dropped: %"PRIu64"/%"PRIu64"\t Pkts received: %"PRIu64"/%"PRIu64"\t Bytes received: %"PRIu64"/%"PRIu64"\n",
                      packets_dropped_this_second,udpheader->packets_dropped,
                      packets_received_this_second, udpheader->packets_received,
                      bytes_received_this_second, udpheader->bytes_received);
    }

    if (udpheader->packets_late_this_sec) {
      fprintf(stderr, "%"PRIu64" packets arrvived late\n", 
                      packets_dropped_this_second);
      udpheader->packets_late_this_sec = 0;
    }
  
  }

  assert(udpheader->curr_buffer != 0);
  
  return (void *) udpheader->curr_buffer;

}

/*
 * Close the udp socket and file
 */

int udpheader_stop_function (udpheader_t* udpheader)
{

  /* get our context, contains all required params */
  int verbose = udpheader->verbose; /* saves some typing */

  float percent_dropped = 0;

  if (udpheader->expected_sequence_no) {
    percent_dropped = (float) ((double)udpheader->packets_dropped / (double)udpheader->expected_sequence_no);
  }

  fprintf(stderr, "Packets dropped %"PRIu64" / %"PRIu64" = %10.8f %\n",
          udpheader->packets_dropped, udpheader->expected_sequence_no, 
          percent_dropped);
  
  close(udpheader->udpfd);
  free(udpheader->curr_buffer);
  free(udpheader->next_buffer);
  free(udpheader->socket_buffer);

  return 0;

}


int main (int argc, char **argv)
{

  /* port on which to listen for incoming connections */
  int port = BPSR_DEFAULT_UDPDB_PORT;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  int arg = 0;

  /* actual struct with info */
  udpheader_t udpheader;

  /* Pointer to array of "read" data */
  char *src;

  while ((arg=getopt(argc,argv,"p:vVh")) != -1) {
    switch (arg) {

    case 'p':
      port = atoi (optarg);
      break;

    case 'v':
      verbose=1;
      break;

    case 'V':
      verbose=2;
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

  /* Setup context information */
  udpheader.verbose = verbose;
  udpheader.port = port;
  udpheader.packets_dropped = 0;
  udpheader.packets_dropped_last_sec = 0;
  udpheader.packets_received = 0;
  udpheader.packets_received_last_sec= 0;
  udpheader.bytes_received = 0;
  udpheader.bytes_received_last_sec = 0;
  udpheader.dropped_packets_to_fill = 0;
  udpheader.packet_in_buffer = 0;
  udpheader.received = 0;
  udpheader.prev_time = time(0);
  udpheader.current_time = udpheader.prev_time;
  udpheader.error_seconds = 10;
  udpheader.packet_length = 0;
  udpheader.state = NOTRECORDING;
    
  time_t utc = udpheader_start_function(&udpheader,0);

  if (utc == -1 ) {
    fprintf(stderr,"Error: udpheader_start_function failed\n");
    return EXIT_FAILURE;
  }

  int quit = 0;
  int header_written = 0;

  while (!quit) {

    uint64_t bsize = udpheader.datasize;

   /* TODO Add a quit control to the read function */
    src = (char *) udpheader_read_function(&udpheader, &bsize);

    /* Quit if we dont get a packet for at least 1 second whilst recording */
    if ((bsize <= 0) && (udpheader.state == RECORDING)) 
      quit = 1;

    if (udpheader.verbose == 2)
      fprintf(stdout,"udpheader_read_function: read %"PRIu64" bytes\n", bsize);
  }    

  if ( udpheader_stop_function(&udpheader) != 0)
    fprintf(stderr, "Error stopping acquisition");

  return EXIT_SUCCESS;

}



int create_udp_socket(udpheader_t* udpheader) {

  udpheader->udpfd = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);

  const int std_buffer_size = (128*1024) - 1;   // stanard socket buffer 128KB
  const int pref_buffer_size = 64*1024*1024;    // preffered socket buffer 64MB

  if (udpheader->udpfd < 0) {
    fprintf(stderr,"Could not created UDP socket: %s\n", strerror(errno));
    return -1;
  }

  struct sockaddr_in udp_sock;
  struct in_addr *addr;

  bzero(&(udp_sock.sin_zero), 8);                     // clear the struct
  udp_sock.sin_family = AF_INET;                      // internet/IP
  udp_sock.sin_port = htons(udpheader->port);           // set the port number
  udp_sock.sin_addr.s_addr = inet_addr("10.0.0.4");

  fprintf(stderr, "listening on 10.0.0.4:%d\n",udpheader->port);

  if (bind(udpheader->udpfd, (struct sockaddr *)&udp_sock, sizeof(udp_sock)) == -1) {
    fprintf(stderr,"Error binding UDP socket: %s\n", strerror(errno));
    return -1;
  }

  // try setting the buffer to the maximum, warn if we cant
  int len = 0;
  int value = pref_buffer_size;
  int retval = 0;
  len = sizeof(value);
  retval = setsockopt(udpheader->udpfd, SOL_SOCKET, SO_RCVBUF, &value, len);
  if (retval != 0) {
    perror("setsockopt SO_RCVBUF");
    return -1;
  }

  // now check if it worked....
  len = sizeof(value);
  value = 0;
  retval = getsockopt(udpheader->udpfd, SOL_SOCKET, SO_RCVBUF, &value, 
                      (socklen_t *) &len);
  if (retval != 0) {
    perror("getsockopt SO_RCVBUF");
    return -1;
  }

  // If we could not set the buffer to the desired size, warn...
  if (value/2 != pref_buffer_size) {
    fprintf(stderr,"Warning. Failed to set udp socket's "
              "buffer size to: %d, falling back to default size: %d\n",
              pref_buffer_size, std_buffer_size);

    len = sizeof(value);
    value = std_buffer_size;
    retval = setsockopt(udpheader->udpfd, SOL_SOCKET, SO_RCVBUF, &value, len);
    if (retval != 0) {
      perror("setsockopt SO_RCVBUF");
      return -1;
    }

    // Now double check that the buffer size is at least correct here
    len = sizeof(value);
    value = 0;
    retval = getsockopt(udpheader->udpfd, SOL_SOCKET, SO_RCVBUF, &value, 
                        (socklen_t *) &len);
    if (retval != 0) {
      perror("getsockopt SO_RCVBUF");
      return -1;
    }
                                                                                                                
    // If we could not set the buffer to the desired size, warn...
    if (value/2 != std_buffer_size) {
      fprintf(stderr, "Warning. Failed to set udp socket's "
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


