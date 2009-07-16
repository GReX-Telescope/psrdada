/*
 * caspsr_udpheader. Reads UDP packets and checks the header for correctness
 */

#include "caspsr_udpheader.h"
#include "dada_generator.h"

#include <sys/socket.h>
#include "sock.h"
#include <math.h>

void usage()
{
  fprintf (stdout,
	   "caspsr_udpheader [options]\n"
     " -h             print help text\n"
	   " -i interface   ip/interface for inc. UDP packets [default all]\n"
	   " -p port        port on which to listen [default %d]\n"
     " -v             verbose messages\n"
     " -V             very verbose messages\n", CASPSR_DEFAULT_UDPDB_PORT);
}


time_t udpheader_start_function (udpheader_t* ctx, time_t start_utc)
{

  multilog_t* log = ctx->log;

  ctx->packets = init_stats_t();
  ctx->bytes = init_stats_t();
  ctx->curr = init_caspsr_buffer(CASPSR_UDP_DATA * CASPSR_UDP_NPACKS);
  ctx->next = init_caspsr_buffer(CASPSR_UDP_DATA * CASPSR_UDP_NPACKS);
  ctx->sock = init_socket_buffer(CASPSR_UDP_PAYLOAD);
  ctx->header = init_caspsr_header();

  zero_caspsr_buffer(ctx->curr);
  zero_caspsr_buffer(ctx->next);

  ctx->prev_time = time(0);
  ctx->current_time = ctx->prev_time;

  ctx->udpfd = dada_udp_sock_in(log, ctx->interface, ctx->port, ctx->verbose);
  if (ctx->udpfd < 0) {
    multilog (log, LOG_ERR, "Error, Failed to create udp socket\n");
    return 0;
  }

  /* set the socket size to 64 MB */
  dada_udp_sock_set_buffer_size (log, ctx->udpfd, ctx->verbose);

  /* Set the current machines name in the header block as RECV_HOST */
  char myhostname[HOST_NAME_MAX] = "unknown";;
  gethostname(myhostname,HOST_NAME_MAX); 

  /* setup the next_seq to the initial value */
  ctx->next_seq = 0;

  return 0;
}

void* udpheader_read_function (udpheader_t* ctx, uint64_t* size)
{

  ctx->got_enough = 0;

  /* Flag for timeout */
  int timeout_ocurred = 0;

  /* How much data has actaully been received */
  uint64_t data_received = 0;

  /* Switch the next and current buffers and their respective counters */
  ctx->temp = ctx->curr;
  ctx->curr = ctx->next;
  ctx->next = ctx->temp;

  /* update the counters for the next buffer */
  ctx->next->count = 0;
  ctx->next->min = ctx->curr->max + 1;
  ctx->next->max = ctx->next->min + CASPSR_UDP_NPACKS;

  zero_caspsr_buffer(ctx->next);

  /* Determine the sequence number boundaries for curr and next buffers */
  int errsv;

  /* Assume we will be able to return a full buffer */
  *size = (CASPSR_UDP_DATA * CASPSR_UDP_NPACKS);

  /* Continue to receive packets */
  while (!ctx->got_enough) {

    /* If there is a packet in the buffer from the previous call */
    if (ctx->sock->have_packet) {

      ctx->sock->have_packet = 0;

    /* Get a packet from the socket */
    } else {

      ctx->sock->have_packet = 0;
      ctx->timer = 100;

      while (!ctx->sock->have_packet) {

        /* Get a packet from the (non-blocking) socket */
        ctx->sock->got = dada_sock_recv (ctx->udpfd, ctx->sock->buffer, 
                                         CASPSR_UDP_PAYLOAD, 0);

        /* If we successfully got a packet */
        if (ctx->sock->got == CASPSR_UDP_PAYLOAD) {

          ctx->sock->have_packet = 1;

        /* If no packet at the socket */
        } else if (ctx->sock->got == -1) {

          errsv = errno;

          if (errsv == EAGAIN) {

            /* Busy sleep for the half the time between packets 
             * at 204800 packets/sec ths is around 5/2 usec */
            ctx->timeout.tv_sec = 0;
            ctx->timeout.tv_usec = 3;
            select(0, NULL, NULL, NULL, &(ctx->timeout));
            ctx->timer--;

          } else {

            multilog (ctx->log, LOG_ERR, "recvfrom failed: %s\n", strerror(errsv));
            ctx->got_enough = 1;

          }

        } else {

          multilog (ctx->log, LOG_ERR, "Received %d bytes, expected %d\n", ctx->sock->got, CASPSR_UDP_PAYLOAD);

        }

        if (ctx->timer == 0) {
          multilog (ctx->log, LOG_ERR, "Timeout on receiving a packet\n");
          ctx->sock->have_packet = 0;
          ctx->got_enough = 1;
        }
      }
    }

    /* check that the packet is of the correct size */
    if (ctx->sock->have_packet && ctx->sock->got != CASPSR_UDP_PAYLOAD) {
      multilog (ctx->log, LOG_WARNING, "Received a packet of unexpected size\n");
      ctx->got_enough = 1;
    }

    /* If we did get a packet within the timeout */
    if (!ctx->got_enough && ctx->sock->have_packet) {

      /* Decode the packets apsr specific header */
      caspsr_decode_header(ctx->header, ctx->sock->buffer);

      //ctx->header->seq_no /= 1024;
      ctx->header->seq_no /= 2048;

      /* If we are waiting for the first packet */
      if ((ctx->next_seq == 0) && (ctx->curr->count == 0)) {
        fprintf(stderr,"START : received packet %"PRIu64"\n", ctx->header->seq_no);
        ctx->next_seq = ctx->header->seq_no;

        /* update the min/max sequence numbers for the receiving buffers */
        ctx->curr->min = ctx->header->seq_no;
        ctx->curr->max = ctx->curr->min + CASPSR_UDP_NPACKS;
        ctx->next->min = ctx->curr->max + 1;
        ctx->next->max = ctx->next->min + CASPSR_UDP_NPACKS;

      } 

      /* Increment statistics */
      ctx->packets->received++;
      ctx->packets->received_per_sec++;
      ctx->bytes->received += CASPSR_UDP_DATA;
      ctx->bytes->received_per_sec += CASPSR_UDP_DATA;

      /* If we have filled the current buffer, then we can stop */
      if (ctx->curr->count >= CASPSR_UDP_NPACKS)
        ctx->got_enough = 1;

      //fprintf(stderr, "curr. count=%"PRIu64", size=%"PRIu64", min=%"PRIu64", max=%"PRIu64"\n",
      //                ctx->curr->count,  ctx->curr->size,  ctx->curr->min,  ctx->curr->max);

      /* we are going to process the packet we have */
      ctx->sock->have_packet = 0;
        
      if (ctx->header->seq_no < ctx->curr->min) {
        multilog (ctx->log, LOG_WARNING, "Packet underflow %"PRIu64" < min (%"PRIu64")\n",
                    ctx->header->seq_no, ctx->curr->min); 

      } else if (ctx->header->seq_no <= ctx->curr->max) {
        memcpy( ctx->curr->buffer + (ctx->header->seq_no - ctx->curr->min) *  CASPSR_UDP_DATA, 
                ctx->sock->buffer + 16,
                CASPSR_UDP_DATA);
        ctx->curr->count++;

      } else if (ctx->header->seq_no <= ctx->next->max) {

        memcpy( ctx->next->buffer + (ctx->header->seq_no - ctx->next->min) *  CASPSR_UDP_DATA, 
                ctx->sock->buffer + 16,
               CASPSR_UDP_DATA);
        ctx->next->count++;

        if (ctx->header->seq_no > ctx->next->max-(CASPSR_UDP_NPACKS/2)) {

          ctx->got_enough = 1;
          multilog (ctx->log, LOG_WARNING, "1: Not keeping up. curr %5.2f%, next %5.2f%\n",
                  ((float) ctx->curr->count / (float) CASPSR_UDP_NPACKS)*100,
                   ((float) ctx->next->count / (float) CASPSR_UDP_NPACKS)*100);

        }

      /*} else if (((float) ctx->next->count) > ((float)(ctx->next->max - ctx->next->min) / 2.0)) {

        multilog (ctx->log, LOG_WARNING, "1: Not keeping up. next->count %"PRIu64" > "
                 "next->(max-min)=%"PRIu64" /2 \n",
                 ctx->next->count,
                 (ctx->next->max - ctx->next->min));

        multilog (ctx->log, LOG_WARNING, "2: Not keeping up. curr %5.2f%, next %5.2f%\n",
                 ((float) ctx->curr->count / (float) CASPSR_UDP_NPACKS)*100,
                 ((float) ctx->next->count / (float) CASPSR_UDP_NPACKS)*100);

        ctx->sock->have_packet = 1;
        ctx->got_enough = 1;
        */
      } else {

        multilog (ctx->log, LOG_WARNING, "2: Not keeping up. curr %5.2f%, next %5.2f%\n",
                 ((float) ctx->curr->count / (float) CASPSR_UDP_NPACKS)*100,
                 ((float) ctx->next->count / (float) CASPSR_UDP_NPACKS)*100);

        ctx->sock->have_packet = 1;
        ctx->got_enough = 1;

      }
    }
  } 

  /* Some checks before returning */
  if (ctx->curr->count) {

    if ((ctx->curr->count < CASPSR_UDP_NPACKS) && !(ctx->timer == 0)) {
      multilog (ctx->log, LOG_WARNING, "Dropped %"PRIu64" packets\n",
               (CASPSR_UDP_NPACKS - ctx->curr->count));

      ctx->packets->dropped += (CASPSR_UDP_NPACKS - ctx->curr->count);
      ctx->packets->dropped_per_sec += (CASPSR_UDP_NPACKS - ctx->curr->count);
      ctx->bytes->dropped += (CASPSR_UDP_DATA * (CASPSR_UDP_NPACKS - ctx->curr->count));
      ctx->bytes->dropped_per_sec += (CASPSR_UDP_DATA * (CASPSR_UDP_NPACKS - ctx->curr->count));
    
    }

    /* If the timeout ocurred, this is most likely due to end of data */
    if (ctx->timer == 0) {
      *size = ctx->curr->count * CASPSR_UDP_DATA;
      fprintf(stderr,"Warning: Suspected EOD received, returning %"PRIu64
                    " bytes\n",*size);
    }

  } else {

    *size = 0;
  }

  /* temporal statistics */
  ctx->next_seq += CASPSR_UDP_NPACKS;
  ctx->prev_time = ctx->current_time;
  ctx->current_time = time(0);
  
  if (ctx->prev_time != ctx->current_time) {

    if (ctx->verbose > 0) {
      multilog (ctx->log, LOG_INFO, "MB/s=%f, kP/s=%f\n", 
                                    (float) ctx->bytes->received_per_sec/(1024.0*1024.0),
                                    (float) ctx->packets->received_per_sec/1000);
    }

    ctx->packets->received_per_sec = 0;
    ctx->bytes->received_per_sec = 0;


  }

  assert(ctx->curr->buffer != 0);
  
  return (void *) ctx->curr->buffer;

}

/*
 * Close the udp socket and file
 */

int udpheader_stop_function (udpheader_t* ctx)
{

  /* get our context, contains all required params */
  float percent_dropped = 0;

  if (ctx->next_seq) {
    percent_dropped = (float) ((double)ctx->packets->dropped / (double)ctx->next_seq);
  }

  fprintf(stderr, "Packets dropped %"PRIu64" / %"PRIu64" = %10.8f %%\n",
          ctx->packets->dropped, ctx->next_seq, percent_dropped);
  
  close(ctx->udpfd);
  free_caspsr_buffer(ctx->curr);
  free_caspsr_buffer(ctx->next);
  free_socket_buffer(ctx->sock);

  return 0;

}


int main (int argc, char **argv)
{

  /* Interface on which to listen for udp packets */
  char * interface = "any";

  /* port on which to listen for incoming connections */
  int port = CASPSR_DEFAULT_UDPDB_PORT;

  /* Flag set in verbose mode */
  char verbose = 0;

  int arg = 0;

  /* actual struct with info */
  udpheader_t udpheader;

  /* Pointer to array of "read" data */
  char *src;

  while ((arg=getopt(argc,argv,"i:p:vVh")) != -1) {
    switch (arg) {

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

  multilog_t* log = multilog_open ("caspsr_udpheader", 0);
  multilog_add (log, stderr);
  multilog_serve (log, DADA_DEFAULT_PWC_LOG);

  udpheader.log = log;

  /* Setup context information */
  udpheader.verbose = verbose;
  udpheader.port = port;
  udpheader.interface = strdup(interface);
  udpheader.state = NOTRECORDING;
    
  time_t utc = udpheader_start_function(&udpheader,0);

  if (utc == -1 ) {
    fprintf(stderr,"Error: udpheader_start_function failed\n");
    return EXIT_FAILURE;
  }

  int quit = 0;

  while (!quit) {

    uint64_t bsize = (CASPSR_UDP_DATA * CASPSR_UDP_NPACKS);

    src = (char *) udpheader_read_function(&udpheader, &bsize);

    if (udpheader.verbose == 2)
      fprintf(stdout,"udpheader_read_function: read %"PRIu64" bytes\n", bsize);
  }    

  if ( udpheader_stop_function(&udpheader) != 0)
    fprintf(stderr, "Error stopping acquisition");

  return EXIT_SUCCESS;

}


