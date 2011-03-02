/*
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <inttypes.h>
#include <assert.h>
#include <errno.h>

#include <infiniband/arch.h>
#include <rdma/rdma_cma.h>

#include "caspsr_rdma.h"
#include "dada_ib.h"

void usage()
{
  fprintf (stdout,
    "caspsr_rdma_client3 [options]\n"
    " -h             print help text\n"
    " -b bufsz       size of each shared memory buffer, must be divisible by 8k [default %d]\n"
    " -n bufs        number of shared memory buffers to simulate [default %d]\n"
    " -p port        connect to server on port [default %d]\n"
    " -x nxfers      number of 'xfers' to send [default %d]\n"
    " -v             verbose output\n",
    CASPSR_RDMA_BUFSZ, CASPSR_RDMA_NBUFS, CASPSR_RDMA_CM_PORT, CASPSR_RDMA_NXFERS);
}


int main(int argc, char *argv[])
{

  struct timeval                  start, end;

  unsigned                        i = 0;
  unsigned                        j = 0;

  void **                         buffers;
  void *                          buffer;

  /* server memory keys passed between server and client */
  dada_ib_shm_block_t * shm_blocks;

  /* size of each shared memory buffer, must be modulo 8k */
  uint64_t buf_size = CASPSR_RDMA_BUFSZ;

  /* number of shared memory buffers */
  int n_bufs = CASPSR_RDMA_NBUFS;

  /* port to listen for rdma_cm connections on */
  int port = CASPSR_RDMA_CM_PORT;

  /* number of WR's per xfer */ 
  uint64_t pkts_per_xfer;

  uint64_t total_xfers_to_send = CASPSR_RDMA_NXFERS;

  uint64_t xfer = 0;

  unsigned verbose = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"b:hn:p:x:v")) != -1) {
    switch (arg) {

    case 'b':
      if (optarg) {
        if (sscanf(optarg, "%"PRIu64, &buf_size) != 1)
        {
          fprintf(stderr, "failed to parse buf size\n");
          usage();
          return EXIT_FAILURE;
        }
        if (buf_size % 8192 != 0) {
          usage();
          return EXIT_FAILURE;
        }
        break;
      } else {
        usage();
        return EXIT_FAILURE;
      }

    case 'h':
      usage();
      return 0;

    case 'n':
     if (optarg) {
        n_bufs = atoi(optarg);
        break;
      } else {
        usage();
        return EXIT_FAILURE;
      }

   case 'p':
     if (optarg) {
       port = atoi(optarg);
       break;
     } else {
       usage();
       return EXIT_FAILURE;
     }

   case 'x':
     if (optarg) {
       total_xfers_to_send = atoi(optarg);
       break;
     } else {
       usage();
       return EXIT_FAILURE;
     }
  
    case 'v':
      verbose++;
      break;

    default:
      usage ();
      return 0;

    }
  }

  multilog_t* log = multilog_open ("caspsr_rdma_client3", 0);
  multilog_add (log, stderr);

  // allocate memory for sending data
  if (verbose)
    multilog(log, LOG_INFO, "allocating memory array %d bytes\n", sizeof(char) * buf_size); 

  // only 1 sending buffer
  buffers = (void **) malloc(sizeof(void *) * 1);
  assert (buffers);

  buffers[0] = (void *) malloc(sizeof(char) * buf_size);
  assert (buffers[0]);

  buffer = buffers[0];

  if (mlock((void *) buffer, (size_t) buf_size) < 0)
    fprintf(stderr, "failed to lock buffer memory: %s\n", strerror(errno));

  // create the CM and CM channel
  dada_ib_cm_t * ib_cm = dada_ib_create_cm(1, log);
  if (!ib_cm)
  {
    multilog(log, LOG_ERR, "dada_ib_create_cm failed\n");
    return EXIT_FAILURE;
  }

  ib_cm->verbose = verbose;

  pkts_per_xfer = buf_size / CASPSR_RDMA_PKTSZ;

  // resolve the route to the server
  if (dada_ib_connect_cm(ib_cm, argv[optind], port, log) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_connect_cm failed\n");
    return EXIT_FAILURE;
  }

  // create the IB verb structures necessary
  if (dada_ib_create_verbs(ib_cm, pkts_per_xfer + 1, log) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_create_verbs failed\n");
    return EXIT_FAILURE;
  }

  /* intialize char array */
  char * tmp_ptr = buffer;
  for (i=0; i<buf_size; i++)
    tmp_ptr[i] = 0;

  // register local memory buffer
  int flags = IBV_ACCESS_LOCAL_WRITE;
  if (dada_ib_reg_buffers(ib_cm, buffers, buf_size, flags, log) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_reg_buffers failed\n");
    return EXIT_FAILURE;
  } 

  if (dada_ib_create_qp(ib_cm, pkts_per_xfer + 1, 1, log) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_create_qp failed\n");
    return EXIT_FAILURE;
  }

  shm_blocks = dada_ib_connect(ib_cm, log);
  if (!shm_blocks)
  {
    multilog(log, LOG_ERR, "dada_ib_create_qp failed\n");
    return EXIT_FAILURE;
  }

  unsigned w_buf = 0;

  if (gettimeofday(&start, NULL)) {
    perror("gettimeofday");
    return 1;
  }

  uint64_t bytes_per_xfer = pkts_per_xfer * CASPSR_RDMA_PKTSZ;

  uint64_t remote_block_id = 0;
  uint64_t bytes_written = 0;

  /* first send data via RDMA */
  while (xfer < total_xfers_to_send)
  {

    ib_cm->sync_from_val = 0;

    // wait for the server to send us a message indicate the byte it is expecting
    if (dada_ib_post_recv (ib_cm, ib_cm->sync_from, log) < 0)
    {
      multilog(log, LOG_ERR, "dada_ib_rdma_post_recv_completion failed\n");
      return EXIT_FAILURE;
    }
    
    if (dada_ib_wait_recv (ib_cm, ib_cm->sync_from, log) < 0)
    {
      multilog(log, LOG_ERR, "dada_ib_rdma_wait_recv failed\n");
      return EXIT_FAILURE;
    }

    remote_block_id = ib_cm->sync_from_val;

    // send this data block buffer
    if (dada_ib_post_sends(ib_cm, buffer, bytes_per_xfer, CASPSR_RDMA_PKTSZ,
                           ib_cm->bufs[0]->mr->lkey, 
                           shm_blocks[remote_block_id].buf_rkey,
                           shm_blocks[remote_block_id].buf_va, log) < 0) 
    {
      multilog(log, LOG_ERR, "dada_ib_post_sends failed\n");
      return EXIT_FAILURE;
    }

    // send the confirmation byte
    ib_cm->sync_to_val = bytes_per_xfer;

    if (dada_ib_post_send (ib_cm, ib_cm->sync_to, log) < 0)
    {
      multilog(log, LOG_ERR, "dada_ib_post_send failed\n");
      return EXIT_FAILURE;
    }

    if (verbose > 1)
      multilog(log, LOG_INFO, "waiting for xfer %"PRIu64" to be completed\n", xfer);

    if (dada_ib_wait_recv (ib_cm, ib_cm->sync_to, log) < 0)
    {
      multilog(log, LOG_ERR, "dada_ib_wait_recv failed\n");
      return EXIT_FAILURE;
    }

    xfer++;

  }

  if (gettimeofday(&end, NULL)) {
    perror("gettimeofday");
    return 1;
  }


  {
    uint64_t elapsed_sec = (end.tv_sec - start.tv_sec);
    uint64_t elapsed_usec = (end.tv_usec - start.tv_usec);

    uint64_t total_usec = (elapsed_sec * 1000000) + elapsed_usec;
    double usec = (double) total_usec;
    double sec = usec / 1000000;

    uint64_t bytes = total_xfers_to_send * buf_size;

    float mbytes = (float) bytes / (1024*1024);
    float mbitps = bytes * 8. / usec;
    float mbytesps = bytes / usec;

    fprintf(stderr, "================ DATA RATE STATISTICS =======================\n");
    fprintf(stderr, "%5.2f MB in %5.2f seconds. Rate %5.2f Mb/s %5.2f MB/s\n",
            mbytes, sec, mbitps, mbytesps);
    fprintf(stderr, "=============================================================\n");
  }

  if (munlock((void *) buffer, (size_t) buf_size) < 0)
    fprintf(stderr, "failed to unlock buffer: %s\n", strerror(errno));

  free(buffer);
  free(shm_blocks);

  if (dada_ib_client_destroy(ib_cm, log) < 0) 
  {
    multilog(log, LOG_ERR, "dada_ib_client_destory failed\n");
  }

  return 0;
}

