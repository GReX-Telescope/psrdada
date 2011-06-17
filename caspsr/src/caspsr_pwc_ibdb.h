/***************************************************************************
 *  
 *    Copyright (C) 2010 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>
#include <assert.h>
#include <netinet/in.h>
#include <signal.h>

#include "futils.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "dada_pwc_main.h"
#include "dada_ib.h"
#include "multilog.h"
#include "ipcio.h"
#include "ascii_header.h"

//#include "caspsr_def.h"


typedef struct caspsr_pwc_ibdb {

  // port to listen on for connections 
  unsigned port;

  // chunk size for IB transport
  unsigned chunk_size;

  // number of chunks in a data block buffer
  unsigned chunks_per_block;

  // verbose messages
  char verbose;

  // flag for active RDMA connection
  // unsigned * connected;

  // number of distributors
  unsigned n_distrib;

  // header memory buffers
  char ** headers;

  /* current observation id, as defined by OBS_ID attribute */
  char obs_id [DADA_OBS_ID_MAXLEN];

  /* xfer counter */
  int64_t xfer_count;

  /* flag to indicate that the xfer is near ending */
  unsigned xfer_ending;

  /* flag to indicate end of obs and data in buffer */
  unsigned sod;

  // total bytes transferred in the observation
  uint64_t observation_bytes;

  // total bytes transferred in the XFER
  uint64_t xfer_bytes;

  // Infiniband Connection Manager
  dada_ib_cm_t ** ib_cms;

  // multilog interface
  multilog_t * log;

} caspsr_pwc_ibdb_t;

#define CASPSR_PWC_IBDB_INIT { 0, 0, 0, 0, 0, 0, "", 0, 0, 0, 0, 0}


// initialize Infiniband resources
int      caspsr_pwc_ibdb_ib_init (caspsr_pwc_ibdb_t * ctx, dada_hdu_t * hdu, multilog_t * log);

// thread for accepting a single IB connection
void *   caspsr_pwc_ibdb_ib_init_thread (void * arg);

// PWCM start function
time_t   caspsr_pwc_ibdb_start (dada_pwc_main_t* pwcm, time_t start_utc);

// PWCM buffer function
void *   caspsr_pwc_ibdb_recv (dada_pwc_main_t* pwcm, int64_t* size);

// PWCM buffer_block function
int64_t  caspsr_pwc_ibdb_recv_block (dada_pwc_main_t* pwcm, void* data,
                                     uint64_t data_size, uint64_t block_id);

// PWCM stop function
int      caspsr_pwc_ibdb_stop (dada_pwc_main_t* pwcm);

// PWCM xfer_pending function
int      caspsr_pwc_ibdb_xfer_pending (dada_pwc_main_t* pwcm);

// PWCM new_xfer function
uint64_t caspsr_pwc_ibdb_new_xfer (dada_pwc_main_t *pwcm);

// PWCM header_valid function
int      caspsr_pwc_ibdb_header_valid (dada_pwc_main_t* pwcm);

// PWCM error function
int      caspsr_pwc_ibdb_error (dada_pwc_main_t* pwcm);

// Utility functions
void     quit (caspsr_pwc_ibdb_t * ctx);
void     signal_handler (int signalValue); 
void     stats_thread(void * arg);
