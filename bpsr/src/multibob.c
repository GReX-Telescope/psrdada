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
  ibob->id = 0;
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
    ibob_set_number (multibob->threads->ibob, ibob + 1);
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

/*!
  The monitor thread simply sits in a loop, opening the connection
  as necessary, and polling the connection every second.  Polling
  can be either a simple ping or a bramdump.  On failure, the
  connection is closed and re-opened ad infinitum every five seconds.
*/

void* multibob_monitor (void* context)
{
  if (!context)
    return 0;

  ibob_thread_t* thread = context;
  ibob_t* ibob = thread->ibob;

  while (!thread->quit)
  {
    pthread_mutex_lock (&(thread->mutex));

    if ( ibob_open (ibob) < 0 )
    {
      fprintf (stderr, "multibob_monitor: could not open %s %d: %s\n",
	       ibob->host, ibob->port, strerror(errno));

      pthread_mutex_unlock (&(thread->mutex));

      sleep (5);
      continue;
    }

    while (!thread->quit)
    {
      int retval = 0;

      pthread_mutex_lock (&(thread->mutex));

      if (thread->bramdump)
	retval = bramdump (ibob);
      else
	retval = ibob_ping (ibob);

      pthread_mutex_unlock (&(thread->mutex));

      if (retval < 0 || thread->quit)
	break;

      sleep (1);
    }
  }

  return 0;
}

/*! open the command connections to all of the ibobs */
int multibob_cmd_open (void* context, FILE* fptr, char* args)
{
  if (!context)
    return -1;

  multibob_t* multibob = context;

  unsigned ibob = 0;
  for (ibob = 0; ibob < multibob->nthread; ibob++)
  {
    ibob_thread_t* thread = multibob->threads + ibob;

    if (thread->id == 0)
    {
      errno = pthread_create (&(thread->id), 0, multibob_monitor, thread);
      if (errno)
	fprintf (stderr, "multibob_cmd_open: error starting thread %d - %s\n",
		 ibob, strerror (errno));
    }
  }
}

/*! close the command connections to all of the ibobs */
int multibob_cmd_close (void* context, FILE* fptr, char* args)
{
  if (!context)
    return -1;

  multibob_t* multibob = context;

  unsigned ibob = 0;
  for (ibob = 0; ibob < multibob->nthread; ibob++)
  {
    ibob_thread_t* thread = multibob->threads + ibob;
    thread -> quit = 1;
  }

  for (ibob = 0; ibob < multibob->nthread; ibob++)
  {
    ibob_thread_t* thread = multibob->threads + ibob;
    if (thread->id)
    {
      void* result = 0;
      pthread_join (thread->id, &result);
      thread->id = 0;
    }
  }
}

/*! reset packet counter on next UTC second, returned */
int multibob_cmd_arm (void* context, FILE* fptr, char* args);

/*! mutex lock all of the ibob interfaces */
void multibob_lock (multibob_t* bob);

/*! mutex unlock all of the ibob interfaces */
void multibob_lock (multibob_t* bob);
