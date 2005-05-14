#include "dada_hdu.h"
#include "dada_def.h"

#include <stdlib.h>
#include <assert.h>

/*! Create a new DADA Header plus Data Unit */
dada_hdu_t* dada_hdu_create (multilog_t* log)
{
  dada_hdu_t* hdu = malloc (sizeof(dada_hdu_t));
  assert (hdu != 0);

  hdu -> log = log;
  hdu -> data_block = 0;
  hdu -> header_block = 0;

  return hdu;
}

/*! Destroy a DADA primary write client main loop */
void dada_hdu_destroy (dada_hdu_t* hdu)
{
  assert (hdu != 0);

  if (hdu->data_block)
    dada_hdu_disconnect (hdu);

  free (hdu);
}

/*! Connect the DADA Header plus Data Unit */
int dada_hdu_connect (dada_hdu_t* hdu)
{
  ipcbuf_t ipcbuf_init = IPCBUF_INIT;
  ipcio_t ipcio_init = IPCIO_INIT;

  assert (hdu != 0);
  assert (hdu->log != 0);

  if (hdu->data_block) {
    fprintf (stderr, "dada_hdu_connect: already connected\n");
    return -1;
  }

  hdu->header_block = malloc (sizeof(ipcbuf_t));
  assert (hdu->header_block != 0);
  *(hdu->header_block) = ipcbuf_init;

  hdu->data_block = malloc (sizeof(ipcio_t));
  assert (hdu->data_block != 0);
  *(hdu->data_block) = ipcio_init;

  /* connect to the shared memory */
  if (ipcbuf_connect (hdu->header_block, DADA_HEADER_BLOCK_KEY) < 0) {
    multilog (hdu->log, LOG_ERR, "Failed to connect to Header Block\n");
    free (hdu->header_block);
    hdu->header_block = 0;
    free (hdu->data_block);
    hdu->data_block = 0;
    return -1;
  }

  if (ipcio_connect (hdu->data_block, DADA_DATA_BLOCK_KEY) < 0) {
    multilog (hdu->log, LOG_ERR, "Failed to connect to Data Block\n");
    free (hdu->header_block);
    hdu->header_block = 0;
    free (hdu->data_block);
    hdu->data_block = 0;
    return -1;
  }

  return 0;
}


/*! Disconnect the DADA Header plus Data Unit */
int dada_hdu_disconnect (dada_hdu_t* hdu)
{
  int status = 0;

  assert (hdu != 0);
  assert (hdu->log != 0);

  if (!hdu->data_block) {
    fprintf (stderr, "dada_hdu_disconnect: not connected\n");
    return -1;
  }

  if (ipcio_disconnect (hdu->data_block) < 0) {
    multilog (hdu->log, LOG_ERR, "Failed to disconnect from Data Block\n");
    status = -1;
  }

  if (ipcbuf_disconnect (hdu->header_block) < 0) {
    multilog (hdu->log, LOG_ERR, "Failed to disconnect from Header Block\n");
    status = -1;
  }

  free (hdu->header_block);
  hdu->header_block = 0;
  free (hdu->data_block);
  hdu->data_block = 0;
  
  return status;
}

/*! Lock DADA Header plus Data Unit designated reader */
int dada_hdu_lock_read (dada_hdu_t* hdu)
{
  assert (hdu != 0);
  assert (hdu->log != 0);

  if (!hdu->data_block) {
    fprintf (stderr, "dada_hdu_disconnect: not connected\n");
    return -1;
  }

  if (ipcbuf_lock_read (hdu->header_block) < 0) {
    multilog (hdu->log, LOG_ERR, "Could not lock Header Block for reading\n");
    return -1;
  }

  if (ipcio_open (hdu->data_block, 'R') < 0) {
    multilog (hdu->log, LOG_ERR, "Could not lock Data Block for reading\n");
    return -1;
  }

  return 0;
}

/*! Unlock DADA Header plus Data Unit designated reader */
int dada_hdu_unlock_read (dada_hdu_t* hdu)
{
  assert (hdu != 0);
  assert (hdu->log != 0);

  if (!hdu->data_block) {
    fprintf (stderr, "dada_hdu_disconnect: not connected\n");
    return -1;
  }

  if (ipcio_close (hdu->data_block) < 0) {
    multilog (hdu->log, LOG_ERR, "Could not unlock Header Block read\n");
    return -1;
  }

  if (ipcbuf_unlock_read (hdu->header_block) < 0) {
    multilog (hdu->log, LOG_ERR,"Could not unlock Data Block read\n");
    return -1;
  }

  return 0;
}

/*! Lock DADA Header plus Data Unit designated writer */
int dada_hdu_lock_write (dada_hdu_t* hdu)
{
  assert (hdu != 0);
  assert (hdu->log != 0);

  if (!hdu->data_block) {
    fprintf (stderr, "dada_hdu_disconnect: not connected\n");
    return -1;
  }

  if (ipcbuf_lock_write (hdu->header_block) < 0) {
    multilog (hdu->log, LOG_ERR, "Could not lock Header Block for writing\n");
    return -1;
  }

  if (ipcio_open (hdu->data_block, 'W') < 0) {
    multilog (hdu->log, LOG_ERR, "Could not lock Data Block for writing\n");
    return -1;
  }

  return 0;
}

/*! Unlock DADA Header plus Data Unit designated writer */
int dada_hdu_unlock_write (dada_hdu_t* hdu)
{
  assert (hdu != 0);
  assert (hdu->log != 0);

  if (!hdu->data_block) {
    fprintf (stderr, "dada_hdu_disconnect: not connected\n");
    return -1;
  }

  if (ipcio_close (hdu->data_block) < 0) {
    multilog (hdu->log, LOG_ERR, "Could not unlock Header Block write\n");
    return -1;
  }

  if (ipcbuf_unlock_write (hdu->header_block) < 0) {
    multilog (hdu->log, LOG_ERR,"Could not unlock Data Block write\n");
    return -1;
  }

  return 0;
}








