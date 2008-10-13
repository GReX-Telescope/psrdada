/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "multibob.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

void ibob_thread_init (ibob_thread_t* ibob)
{
  ibob->ibob = ibob_construct ();

  pthread_mutex_init(&(ibob->mutex), NULL);
  pthread_cond_init (&(ibob->cond), NULL);

  ibob->bramdump = 0;
  ibob->quit = 0;
}

/*! allocate and initialize a new ibob_t struct */
multibob_t* multibob_construct (unsigned nibob)
{
  multibob_t* multibob = malloc (sizeof(multibob_t));

  multibob->threads = malloc (sizeof(ibob_thread_t) * nibob);
  multibob->nthread = nibob;

  unsigned ibob = 0;

  for (ibob = 0; ibob < nibob; ibob++)
  {
    ibob_thread_init (multibob->threads + ibob);
    ibob_set_number (multibob->threads->ibob, ibob);
  }

  multibob->parser = command_parse_create ();

  command_parse_add (multibob->parser, multibob_cmd_state, multibob,
                     "state", "get the current state", NULL);

  command_parse_add (multibob->parser, multibob_cmd_hostport, multibob,
                     "hostport", "set the hostname and port number", NULL);

  command_parse_add (multibob->parser, multibob_cmd_mac, multibob,
                     "mac", "set the target MAC address", NULL);

  command_parse_add (multibob->parser, multibob_cmd_open, multibob,
                     "open", "open command interface connections", NULL);

  command_parse_add (multibob->parser, multibob_cmd_close, multibob,
                     "close", "close command interface connections", NULL);

  command_parse_add (multibob->parser, multibob_cmd_arm, multibob,
                     "arm", "reset packet count", NULL);

  command_parse_add (multibob->parser, multibob_cmd_quit, multibob,
                     "quit", "quit", NULL);

  return multibob;
}

/*! free all resources reserved for ibob communications */
int multibob_destroy (multibob_t* bob);

/*! set the host and port number of the specified ibob */
int multibob_cmd_hostport (void* context, FILE* fptr, char* args);

/*! set the target MAC address of the specified ibob */
int multibob_cmd_mac (void* context, FILE* fptr, char* args);

/*! open the command connections to all of the ibobs */
int multibob_cmd_open (void* context, FILE* fptr, char* args);

/*! close the command connections to all of the ibobs */
int multibob_cmd_close (void* context, FILE* fptr, char* args);

/*! reset packet counter on next UTC second, returned */
int multibob_cmd_arm (void* context, FILE* fptr, char* args);

/*! mutex lock all of the ibob interfaces */
void multibob_lock (multibob_t* bob);

/*! mutex unlock all of the ibob interfaces */
void multibob_lock (multibob_t* bob);
