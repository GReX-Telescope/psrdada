/*
 *
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <inttypes.h>

#include "caspsr_rdma.h"
#include "dada_ib.h"

#define DADA_IB_RECV_WRID 3

void usage()
{
  fprintf (stdout,
    "caspsr_rdma_server3 [options]\n"
    " -h             print help text\n"
    " -b bufsz       size of each shared memory buffer, must be divisible by 8k [default %d]\n"
    " -n bufs        number of shared memory buffers to simulate [default %d]\n"
    " -p port        accept client connections on this port [default %d]\n"
    " -x xfers       number of 'xfers' to process [default %d]\n"
    " -v             verbose output\n",
    CASPSR_RDMA_BUFSZ, CASPSR_RDMA_NBUFS, CASPSR_RDMA_CM_PORT, CASPSR_RDMA_NXFERS);
}

int main(int argc, char *argv[])
{

  struct timeval                  start, end;

  int                             err;
  unsigned                        i = 0;
  unsigned                        j = 0;

  /* server memory keys passed between server and client */
  dada_ib_shm_block_t * shm_blocks;

  /* "shared memory" buffers */
  void  ** buffers;

  /* size of each shared memory buffer, must be modulo 8k */
  uint64_t buf_size = CASPSR_RDMA_BUFSZ;

  /* number of shared memory buffers */
  uint32_t n_bufs = CASPSR_RDMA_NBUFS;

  /* port to listen for rdma_cm connections on */
  int port = CASPSR_RDMA_CM_PORT;

  uint64_t wrs_per_xfer;

  /* current "shared memory" block to write to */
  uint64_t w_buf;

  uint64_t xfer = 0;

  uint64_t total_xfers_to_receive = CASPSR_RDMA_NXFERS;

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
        if (sscanf(optarg, "%"PRIu32, &n_bufs) != 1)
        {
          fprintf(stderr, "failed to parse n_bufs\n");
          usage();
          return EXIT_FAILURE;
        }
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
        total_xfers_to_receive = atoi(optarg);
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

  multilog_t* log = multilog_open ("caspsr_rdma_server3", 0);
  multilog_add (log, stderr);

  wrs_per_xfer = buf_size / CASPSR_RDMA_PKTSZ;

  /* allocate the shared memory buffers */
  buffers = (void **) malloc(sizeof(void *) * n_bufs);
  assert (buffers);

  shm_blocks = dada_ib_create_shm_blocks(n_bufs, log);
  assert (shm_blocks);

  for (i=0; i<n_bufs; i++) {

    buffers[i] = (void *) malloc(sizeof(char) * buf_size);
    assert (buffers[i]);

    if (mlock((void *) buffers[i], (size_t) buf_size) < 0)
      fprintf(stderr, "failed to lock buffers[%d] memory: %s\n", i, strerror(errno));

    memset(buffers[i], 0, buf_size);

  }

  dada_ib_cm_t * ib_cm = dada_ib_create_cm(n_bufs, log);
  if (!ib_cm) 
  {
    multilog(log, LOG_ERR, "dada_ib_create_cm failed\n");
    return EXIT_FAILURE;
  }

  ib_cm->verbose = verbose;

  // listen for a connection request on the specified port
  if (dada_ib_listen_cm(ib_cm, port, 1, log) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_listen_cm failed\n");
    return EXIT_FAILURE;
  }

  // create the IB verb structures necessary
  if (dada_ib_create_verbs(ib_cm, wrs_per_xfer + 1, log) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_create_verbs failed\n");
    return EXIT_FAILURE;
  }

  int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

  if (dada_ib_reg_buffers(ib_cm, buffers, buf_size, flags, log) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_register_memory_buffers failed\n");
    return EXIT_FAILURE;
  }

  if (dada_ib_register_shm_blocks(ib_cm, shm_blocks, log) < 0) 
  {
    multilog(log, LOG_ERR, "dada_ib_register_shm_blocks failed\n");
    return EXIT_FAILURE;
  }

  if (dada_ib_create_qp (ib_cm, 1, 1, log) < 0) 
  {
    multilog(log, LOG_ERR, "dada_ib_create_qp failed\n");
    return EXIT_FAILURE;
  }

  w_buf = 0;

  // pre-post receive for the first sync message
  if (dada_ib_post_recv (ib_cm, ib_cm->sync_from, log) < 0) 
  {
    multilog(log, LOG_ERR, "dada_ib_post_recv failed\n");
    return EXIT_FAILURE;
  }

  // accept the proper IB connection
  if (dada_ib_accept (ib_cm, shm_blocks, log) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_accept failed\n");
    return EXIT_FAILURE;
  }

  if (gettimeofday(&start, NULL)) {
    perror("gettimeofday");
    return 1;
  }

  uint64_t bytes_per_xfer = wrs_per_xfer * CASPSR_RDMA_PKTSZ;
  uint64_t wbuf = 0;

  while (xfer < total_xfers_to_receive) 
  {

    wbuf = xfer % n_bufs;

    // send the confirmation start byte
    multilog(log, LOG_INFO, "server instructs receiving buffer=%"PRIu64"\n", wbuf);

    ib_cm->sync_to_val = wbuf;
    if (dada_ib_post_send (ib_cm, ib_cm->sync_to, log) < 0)
    {
      multilog(log, LOG_ERR, "dada_ib_post_send failed\n");
      return EXIT_FAILURE;
    }

    // confirm that the byte has been sent
    if (dada_ib_wait_recv(ib_cm, ib_cm->sync_to, log) < 0)
    {
      multilog(log, LOG_ERR, "dada_ib_wait_recv failed\n");
      return EXIT_FAILURE;
    }
    
    if (verbose > 1)
      multilog(log, LOG_INFO, "xfer %"PRIu64" now running...\n", xfer);

    /* only post receive if xfer > 0 */
    if (xfer)
    {
      // pre-post recv for the "next" transfer
      if (dada_ib_post_recv (ib_cm, ib_cm->sync_from, log) < 0)
      {
        multilog(log, LOG_ERR, "dada_ib_post_recv failed\n");
        return EXIT_FAILURE;
      }
    }

    if (verbose > 1)
      fprintf(stderr, "waiting for xfer %"PRIu64" to be completed\n", xfer);

    // accept the completion message for the "current" transfer
    if (dada_ib_wait_recv(ib_cm, ib_cm->sync_from, log) < 0)
    {
      multilog(log, LOG_ERR, "dada_ib_wait_recv failed\n");
      return EXIT_FAILURE;
    }
    multilog(log, LOG_INFO, "client reports bytes sent=%"PRIu64"\n", ib_cm->sync_from_val);

    // the RDMA transport should have finished, check the data
    if (verbose > 1)
    {
      fprintf(stderr, "bufs[%"PRIu64"]=", w_buf);
      char * tmp_ptr = (char *) buffers[w_buf];
      for (j=0; j<3; j++) {
        fprintf(stderr, "%d ", tmp_ptr[j]);
      }
      fprintf(stderr,"...");
      for (j=buf_size-3; j<buf_size; j++)
        fprintf(stderr, " %d", tmp_ptr[j]);
      fprintf(stderr, "\n");
    }

    // increment counts
    xfer++;

  }

  if (gettimeofday(&end, NULL)) {
    perror("gettimeofday");
    return 1;
  }

  {
    float usec = (end.tv_sec - start.tv_sec) * 1000000 +
      (end.tv_usec - start.tv_usec);
    uint64_t bytes = total_xfers_to_receive * buf_size;

    float mbytes = bytes / (1024*1024);
    float secs = usec / 1000000;
    float mbitps = bytes * 8. / usec;
    float mbytesps = bytes / usec;

    fprintf(stderr, "================ DATA RATE STATISTICS =======================\n");
    fprintf(stderr, "%5.2f MB in %5.2f seconds. Rate %5.2f Mb/s %5.2f MB/s\n",
            mbytes, secs, mbitps, mbytesps);
    fprintf(stderr, "=============================================================\n");
  }

  for (i=0; i<n_bufs; i++)
  {
    if (munlock((void *) buffers[i], (size_t) buf_size) < 0)
      fprintf(stderr, "failed to unlock buffers[%d]: %s\n", i, strerror(errno));
    free(buffers[i]);
  }
  free(buffers);
  free(shm_blocks);

  dada_ib_destroy(ib_cm, log);

  return 0;
}

