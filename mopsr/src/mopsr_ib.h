/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#ifndef __MOPSR_IB_H
#define __MOPSR_IB_H

#include "dada_ib.h"
#include "mopsr_def.h"

/*
typedef struct {

  unsigned int host_id;

  unsigned int chan_start;

  unsigned int chan_end;

} mopsr_ibdst_t;

typedef struct {

  unsigned int host_id;

  unsigned int ant_start;

  unsigned int ant_end;

} mopsr_ibsrc_t;
*/
typedef struct {

  dada_ib_cm_t * ib_cm;

  char host[64]; 

  int port;

  int chan;

  int ant_first;

  int ant_last;

} mopsr_conn_t;

typedef struct {

  // multilog inteface
  multilog_t * log;

  // cornerturn information
  mopsr_conn_t * conn_info;

  // number of IB receivers
  unsigned int n_dst;

  // total number of antenna [e.g. 352]
  unsigned int nant;

  // total number of input channels [e.g. 39]
  unsigned int nchan;

  // first antenna in input
  unsigned int ant_first;

  // last antenna in input
  unsigned int ant_last;

  unsigned verbose;

  char * header;

  // quit signal for exiting after 1 observation
  int quit;

  // IB Connection Managers
  dada_ib_cm_t ** ib_cms;

} mopsr_dbib_t;

#define MOPSR_DBIB_INIT { 0, 0, 0, 0, 0, 0, 0, 0, "", 0, 0 }

typedef struct {

  // multilog inteface
  multilog_t * log;

  // cornerturn information
  mopsr_conn_t * conn_info;

  // number of IB receivers
  unsigned int n_src;

  // total number of antenna [e.g. 352]
  unsigned int nant;

  // total number of input channels [e.g. 39]
  unsigned int nchan;

  // channel for this instance 
  unsigned int chan;

  unsigned verbose;

  char * header;

  // quit signal for exiting after 1 observation
  int quit;

  // IB Connection Managers
  dada_ib_cm_t ** ib_cms;

} mopsr_ibdb_t;

#define MOPSR_IBDB_INIT {  0, 0, 0, 0, 0, 0, 0, "", 0, 0 }

#endif //  MOPSR_IB_H
