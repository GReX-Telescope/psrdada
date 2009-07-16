/*
 * caspsr_udpdb. Reads UDP packets and checks the header for correctness
 */

#include "caspsr_udpdb.h"
#include "dada_generator.h"

#include <sys/socket.h>
#include "sock.h"
#include <math.h>

#define _DEBUG 1

void usage()
{
  fprintf (stdout,
	   "caspsr_udpdb [options]\n"
     " -h             print help text\n"
	   " -i iface       interface for UDP packets [default all interfaces]\n"
	   " -p port        port to open for UDP packest [default %d]\n"
     " -c port        port to open for PWCC commands [defalt %d]\n"
     " -l port        port to make multilog output available on [defalt %d]\n"
     " -k key         hexidecimal shared memory key [default %x]\n"
     " -H file        load header from file, run without PWCC\n"
     " -d             run as a daemon\n"
     " -v             verbose messages\n",
     CASPSR_DEFAULT_UDPDB_PORT, CASPSR_DEFAULT_PWC_PORT, 
     CASPSR_DEFAULT_PWC_LOGPORT, DADA_DEFAULT_BLOCK_KEY);
}

/* Determine if the header is valid. Returns 1 if valid, 0 otherwise */
int udpdb_header_valid_function (dada_pwc_main_t* pwcm) {

  unsigned utc_size = 64;
  char utc_buffer[utc_size];
  int valid = 1;

  /* Check if the UTC_START is set in the header*/
  if (ascii_header_get (pwcm->header, "UTC_START", "%s", utc_buffer) < 0) {
    valid = 0;
  }

  /* Check whether the UTC_START is set to UNKNOWN */
  if (strcmp(utc_buffer,"UNKNOWN") == 0)
    valid = 0;

#ifdef _DEBUG
  multilog(pwcm->log, LOG_INFO, "Checking if header is valid: %d\n", valid);
#endif

  return valid;
}

/*
 * Error function
 */
int udpdb_error_function (dada_pwc_main_t* pwcm) {

  udpdb_t *ctx = (udpdb_t*)pwcm->context;

  /* If UTC_START has been received, the buffer function should be returning 
     data */
  if (udpdb_header_valid_function (pwcm) == 1) {

    ctx->error_seconds--;

    multilog (pwcm->log, LOG_WARNING, "Error Function has %"PRIu64" error "
              "seconds remaining\n",ctx->error_seconds);

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
#ifdef _DEBUG 
  fprintf(stderr, "udpdb_start_function()\n");
#endif

  udpdb_t *ctx = (udpdb_t*)pwcm->context;
  multilog_t* log = pwcm->log;

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
  char myhostname[HOST_NAME_MAX] = "unknown";
  gethostname(myhostname,HOST_NAME_MAX); 
  ascii_header_set (pwcm->header, "RECV_HOST", "%s", myhostname);

  /* setup the next_seq to the initial value */
  ctx->next_seq = 0;
  ctx->error_seconds = 10;

  return 0;
}


/*
 * Read data from the UDP socket and return a pointer to the data read
 * and set the size to the data read
 */
void* udpdb_buffer_function (dada_pwc_main_t* pwcm, uint64_t* size)
{

#ifdef _DEBUG 
  fprintf(stderr, "udpdb_buffer_function()\n");
#endif

  udpdb_t *ctx = (udpdb_t*)pwcm->context;
  multilog_t* log = pwcm->log;

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
  int64_t max_ignore = *size;
  unsigned ignore_packet = 0;

  /* Continue to receive packets */
  while (!ctx->got_enough) {

    /* If there is a packet in the buffer from the previous call */
    if (ctx->sock->have_packet) {

      ctx->sock->have_packet = 0;
      ignore_packet = 0;

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
          ignore_packet = 0;

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

            multilog (log, LOG_ERR, "recvfrom failed: %s\n", strerror(errsv));
            ctx->got_enough = 1;

          }

        } else {

          multilog (log, LOG_ERR, "Received %d bytes, expected %d\n", 
                                  ctx->sock->got, CASPSR_UDP_PAYLOAD);

        }

        if (ctx->timer == 0) {
          multilog (log, LOG_ERR, "Timeout on receiving a packet\n");
          ctx->sock->have_packet = 0;
          ctx->got_enough = 1;
        }
      }
    }

    /* check that the packet is of the correct size */
    if (ctx->sock->have_packet && ctx->sock->got != CASPSR_UDP_PAYLOAD) {
      multilog (log, LOG_WARNING, "Packet size mismatch. received=%d, "
                "expected=\n", ctx->sock->got, CASPSR_UDP_PAYLOAD);
      ctx->got_enough = 1;
    }

    /* If we did get a packet within the timeout */
    if (!ctx->got_enough && ctx->sock->have_packet) {

      /* Decode the packets apsr specific header */
      caspsr_decode_header(ctx->header, ctx->sock->buffer);

      ctx->header->seq_no /= 2048;

      /* If we are waiting for the first packet */
      if ( (ctx->next_seq == 0) && (ctx->curr->count == 0) ) {

        if (ctx->header->seq_no < 10000) {
          fprintf(stderr,"START : received packet %"PRIu64"\n", ctx->header->seq_no);
          ctx->next_seq = ctx->header->seq_no;

          /* update the min/max sequence numbers for the receiving buffers */
          ctx->curr->min = ctx->header->seq_no;
          ctx->curr->max = ctx->curr->min + CASPSR_UDP_NPACKS;
          ctx->next->min = ctx->curr->max + 1;
          ctx->next->max = ctx->next->min + CASPSR_UDP_NPACKS;

        } else {
          ignore_packet = 1;
        }
      }

      //fprintf(stderr, "seq_no = %"PRIu64", ignore=%d\n", ctx->header->seq_no, ignore_packet);

      /* If we are still waiting for the start of data */
      if (ignore_packet) {

        max_ignore -= CASPSR_UDP_PAYLOAD;
        if (max_ignore < 0)
          ctx->got_enough = 1;

      } else {

        //fprintf(stderr, "seq_no = %"PRIu64", ignore=%d\n", ctx->header->seq_no, ignore_packet);

        /* Increment statistics */
        ctx->packets->received++;
        ctx->packets->received_per_sec++;
        ctx->bytes->received += CASPSR_UDP_DATA;
        ctx->bytes->received_per_sec += CASPSR_UDP_DATA;

        /* If we have filled the current buffer, then we can stop */
        if (ctx->curr->count >= CASPSR_UDP_NPACKS)
          ctx->got_enough = 1;

        /* we are going to process the packet we have */
        ctx->sock->have_packet = 0;
          
        if (ctx->header->seq_no < ctx->curr->min) {
          multilog (log, LOG_WARNING, "Packet underflow %"PRIu64" < min (%"PRIu64
                    ")\n", ctx->header->seq_no, ctx->curr->min); 

        } else if (ctx->header->seq_no <= ctx->curr->max) {
          memcpy( ctx->curr->buffer + (ctx->header->seq_no - ctx->curr->min) 
                  * CASPSR_UDP_DATA, ctx->sock->buffer + 16, CASPSR_UDP_DATA);
          ctx->curr->count++;

        } else if (ctx->header->seq_no <= ctx->next->max) {

          memcpy( ctx->next->buffer + (ctx->header->seq_no - ctx->next->min) * 
                  CASPSR_UDP_DATA, ctx->sock->buffer + 16, CASPSR_UDP_DATA);
          ctx->next->count++;

          if (ctx->header->seq_no > ctx->next->max-(CASPSR_UDP_NPACKS/2)) {
  
            ctx->got_enough = 1;
            multilog (log, LOG_WARNING, "Not keeping up. curr=%5.2f%, "
                      "next=%5.2f%\n",
                      ((float) ctx->curr->count / (float) CASPSR_UDP_NPACKS)*100,
                      ((float) ctx->next->count / (float) CASPSR_UDP_NPACKS)*100);

          }

        } else {

          multilog (log, LOG_WARNING, "Not keeping up. curr=%5.2f%, "
                    "next=%5.2f%\n",
                    ((float) ctx->curr->count / (float) CASPSR_UDP_NPACKS)*100,
                    ((float) ctx->next->count / (float) CASPSR_UDP_NPACKS)*100);

          ctx->sock->have_packet = 1;
          ctx->got_enough = 1;
        }
      }
    }
  } 

  /* Some checks before returning */
  if (ctx->curr->count) {

    if ((ctx->curr->count < CASPSR_UDP_NPACKS) && !(ctx->timer == 0)) {
      multilog (log, LOG_WARNING, "Dropped %"PRIu64" packets\n",
               (CASPSR_UDP_NPACKS - ctx->curr->count));

      ctx->packets->dropped += (CASPSR_UDP_NPACKS - ctx->curr->count);
      ctx->packets->dropped_per_sec += (CASPSR_UDP_NPACKS - ctx->curr->count);
      ctx->bytes->dropped += (CASPSR_UDP_DATA * 
                              (CASPSR_UDP_NPACKS - ctx->curr->count));
      ctx->bytes->dropped_per_sec += (CASPSR_UDP_DATA * 
                                     (CASPSR_UDP_NPACKS - ctx->curr->count));
    
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

  /* only update the next_seq number if packets have been received */
  if (ctx->packets->received)
    ctx->next_seq += CASPSR_UDP_NPACKS;

  /* temporal statistics */
  ctx->prev_time = ctx->current_time;
  ctx->current_time = time(0);
  
  if (ctx->prev_time != ctx->current_time) {

    if (ctx->verbose > 0) {
      multilog (log, LOG_INFO, "MB/s=%f, kP/s=%f\n", 
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

int udpdb_stop_function (dada_pwc_main_t* pwcm)
{

  udpdb_t *ctx = (udpdb_t*)pwcm->context;
  multilog_t *log = pwcm->log;

  /* get our context, contains all required params */
  if (ctx->packets->dropped && ctx->next_seq>0) {
    double percent = (double) ctx->packets->dropped / (double) ctx->next_seq;

    multilog(log, LOG_INFO, "packets dropped %"PRIu64" / %"PRIu64 " = %8.6f %\n"
             , ctx->packets->dropped, ctx->next_seq);

  }

  close(ctx->udpfd);
  free_caspsr_buffer(ctx->curr);
  free_caspsr_buffer(ctx->next);
  free_socket_buffer(ctx->sock);

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

  /* port for UDP packets */
  int u_port = CASPSR_DEFAULT_UDPDB_PORT;

  /* pwcc command port */
  int c_port = CASPSR_DEFAULT_PWC_PORT;
  
  /* multilog output port */
  int l_port = CASPSR_DEFAULT_PWC_LOGPORT;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  /* actual struct with info */
  udpdb_t udpdb;

  /* custom header from a file, implies no controlling pwcc */
  char * header_file = NULL;

  /* pwcc control flag, only disabled if a custom header is supplied */
  int pwcc_control = 1;

  /* Pointer to array of "read" data */
  char *src;

  int arg = 0;

  while ((arg=getopt(argc,argv,"c:dH:k:i:l:p:vh")) != -1) {
    switch (arg) {

    case 'c':
      if (optarg) {
        c_port = atoi(optarg);
        break;
      } else {
        usage();
        return EXIT_FAILURE;
      }

    case 'd':
      daemon = 1;
      break; 
  
    case 'H':
      if (optarg) {
        header_file = optarg;
        pwcc_control = 0;
      } else {
        fprintf(stderr,"-H flag requires a file arguement\n");
        usage();
      }
      break;

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr,"caspsr_udpdb: could not parse key from %s\n",optarg);
        return -1;
      }
      break;

    case 'i':
      if (optarg)
        interface = optarg;
      break;
    
    case 'l':
      if (optarg) {
        l_port = atoi(optarg);
        break;
      } else {
        usage();
        return EXIT_FAILURE;
      }

    case 'p':
      u_port = atoi (optarg);
      break;

    case 'v':
      verbose = 1;
      break;

    case 'h':
      usage();
      return 0;
      
    default:
      usage ();
      return 0;
      
    }
  }

  log = multilog_open ("caspsr_udpdb", 0);

  if (daemon)
    be_a_daemon ();
  else
    multilog_add (log, stderr);

  multilog_serve (log, l_port);

  pwcm = dada_pwc_main_create();

  pwcm->log                   = log;
  pwcm->context               = &udpdb;
  pwcm->start_function        = udpdb_start_function;
  pwcm->buffer_function       = udpdb_buffer_function;
  pwcm->stop_function         = udpdb_stop_function;
  pwcm->error_function        = udpdb_error_function;
  pwcm->header_valid_function = udpdb_header_valid_function;
  pwcm->verbose               = verbose;

  /* Setup context information */
  udpdb.verbose = verbose;
  udpdb.port = u_port;
  udpdb.interface = strdup(interface);
  udpdb.state = NOTRECORDING;

  /* connect to the shared memory block */
  hdu = dada_hdu_create (pwcm->log);
  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_write (hdu) < 0)
    return EXIT_FAILURE;

  pwcm->data_block = hdu->data_block;
  pwcm->header_block = hdu->header_block;

  /* we are controlled by PWC control interface */
  if (pwcc_control) {

    pwcm->header = hdu->header;

    if (verbose) 
      fprintf (stdout, "caspsr_udpdb: creating dada pwc control interface\n");

    pwcm->pwc = dada_pwc_create();

    pwcm->pwc->port = c_port;

    if (verbose) 
      fprintf (stdout, "caspsr_udpdb: creating dada server\n");
    if (dada_pwc_serve (pwcm->pwc) < 0) {
      fprintf (stderr, "caspsr_udpdb: could not start server\n");
      return EXIT_FAILURE;
    }

    if (verbose) 
      fprintf (stdout, "caspsr_udpdb: entering PWC main loop\n");

    if (dada_pwc_main (pwcm) < 0) {
      fprintf (stderr, "caspsr_udpdb: error in PWC main loop\n");
      return EXIT_FAILURE;
    }

    if (dada_hdu_unlock_write (hdu) < 0)
      return EXIT_FAILURE;

    if (dada_hdu_disconnect (hdu) < 0)
      return EXIT_FAILURE;

    if (verbose) 
      fprintf (stdout, "caspsr_udpdb: destroying pwc\n");
    dada_pwc_destroy (pwcm->pwc);

    if (verbose) 
      fprintf (stdout, "caspsr_udpdb: destroying pwc main\n");
    dada_pwc_main_destroy (pwcm);


  /* If a custom header argument was supplied */
  } else {

    fprintf(stdout,"caspsr_udpdb: Manual Mode\n"); 

    char *   header_buf = 0;
    uint64_t header_size = 0;
    char *   buffer = 0;
    unsigned buffer_size = 64;

    header_size = ipcbuf_get_bufsz (hdu->header_block);
    multilog (pwcm->log, LOG_INFO, "header block size = %llu\n", header_size);
    header_buf = ipcbuf_get_next_write (hdu->header_block);
    pwcm->header = header_buf;

    if (!header_buf)  {
      multilog (pwcm->log, LOG_ERR, "could not get next header block\n");
      return EXIT_FAILURE;
    }

    fprintf(stdout,"caspsr_udpdb: reading custom header %s\n", header_file);
    if (fileread (header_file, header_buf, header_size) < 0)  {
      multilog (pwcm->log, LOG_ERR, "could not read header from %s\n",
                header_file);
      return EXIT_FAILURE;
    }

    /* create a buffer for reading the UTC_START */
    if (!buffer)
      buffer = malloc (buffer_size);
    assert (buffer != 0);

    if (verbose)
      fprintf(stdout, "caspsr_udpdb: running start function\n");

    time_t utc = udpdb_start_function(pwcm,0);
    if (utc == -1 ) {
      multilog(pwcm->log, LOG_ERR, "Could not run start function\n");
      return EXIT_FAILURE;
    } 

    strftime (buffer, buffer_size, DADA_TIMESTR, gmtime(&utc));
    fprintf(stdout, "caspsr_udpdb: UTC_START = %s\n",buffer);

    /* write UTC_START to the header */
    if (ascii_header_set (header_buf, "UTC_START", "%s", buffer) < 0) {
      multilog (pwcm->log, LOG_ERR, "failed ascii_header_set UTC_START\n");
      return -1;
    }

    multilog (pwcm->log, LOG_INFO, "UTC_START %s written to header\n", buffer);

    /* do not set header parameters anymore - acqn. doesn't start */
    if (ipcbuf_mark_filled (hdu->header_block, header_size) < 0)  {
      multilog (pwcm->log, LOG_ERR, "Could not mark filled header block\n");
      return EXIT_FAILURE;
    }

    multilog (pwcm->log, LOG_INFO, "header marked filled\n");

    uint64_t bsize = 0;
    while (1) {

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

  return EXIT_SUCCESS;

}


