/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "multibob.h"
#include "dada_def.h"

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <time.h>

void ibob_thread_init (ibob_thread_t* ibob, int number)
{
  ibob->ibob = ibob_construct ();
  ibob_set_number (ibob->ibob, number);

  pthread_mutex_init(&(ibob->mutex), NULL);
  pthread_cond_init (&(ibob->cond), NULL);

  ibob->id = 0;
  ibob->bramdump = 0;
  ibob->alive = 0;

  ibob->quit = 1;
}

/*! allocate and initialize a new ibob_t struct */
multibob_t* multibob_construct (unsigned nibob)
{
  multibob_t* multibob = malloc (sizeof(multibob_t));

  multibob->threads = malloc (sizeof(ibob_thread_t) * nibob);
  multibob->nthread = nibob;

  unsigned ibob = 0;

  for (ibob = 0; ibob < nibob; ibob++)
    ibob_thread_init (multibob->threads + ibob, ibob + 1);

  multibob->parser = command_parse_create ();
  multibob->server = 0;
  multibob->port = 0;

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
int multibob_destroy (multibob_t* multibob)
{
}

/*!
  The monitor thread simply sits in a loop, opening the connection
  as necessary, and polling the connection every second.  Polling
  can be either a simple ping or a bramdump.  On failure, the
  connection is closed and re-opened ad infinitum every five seconds.
*/

int bramdump (ibob_t* ibob)
{
  fprintf (stderr, "bramdump not implemented\n");
  return 0;
}

void* multibob_monitor (void* context)
{
  if (!context)
    return 0;

  ibob_thread_t* thread = context;
  ibob_t* ibob = thread->ibob;

  while (!thread->quit)
  {
    pthread_mutex_lock (&(thread->mutex));

    fprintf (stderr, "multibob_monitor: opening %s:%d\n",
             ibob->host, ibob->port);

    if ( !ibob_is_open(ibob) && ibob_open (ibob) < 0 )
    {
      fprintf (stderr, "multibob_monitor: could not open %s:%d - %s\n",
	       ibob->host, ibob->port, strerror(errno));

      pthread_mutex_unlock (&(thread->mutex));

      if (thread->quit)
        break;

      sleep (5);
      continue;
    }

    pthread_mutex_unlock (&(thread->mutex));

    while (!thread->quit)
    {
      int retval = 0;

      pthread_mutex_lock (&(thread->mutex));

      if (thread->bramdump)
	retval = bramdump (ibob);
      else
	retval = ibob_ping (ibob);

      pthread_mutex_unlock (&(thread->mutex));

      if (retval < 0)
      {
        fprintf (stderr, "multibob_monitor: communicaton failure on %s:%d\n",
                 ibob->host, ibob->port);
	break;
      }

      thread->alive = time(0);

      if (thread->quit)
        break;

      sleep (1);
    }

    fprintf (stderr, "multibob_monitor: closing connection with %s:%d\n",
             ibob->host, ibob->port);

    ibob_close (ibob);
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

    // if thread is currently running, then connection is open
    if (!thread->quit)
      continue;

    // if thread has been spawned but is not running, clean up its resources
    if (thread->id)
    {
      void* result = 0;
      pthread_join (thread->id, &result);
    }

    thread->id = 0;
    thread->quit = 0;

    int err = pthread_create (&(thread->id), 0, multibob_monitor, thread);
    if (err)
      fprintf (stderr, "multibob_cmd_open: error starting thread %d - %s\n",
	       ibob, strerror (err));
  }
}

/*! close the command connections to all of the ibobs */
int multibob_cmd_close (void* context, FILE* fptr, char* args)
{
  if (!context)
    return -1;

  unsigned ibob = 0;
  multibob_t* multibob = context;

  for (ibob = 0; ibob < multibob->nthread; ibob++)
    multibob->threads[ibob].quit = 1;
}

/*! reset packet counter on next UTC second, returned */
int multibob_cmd_state (void* context, FILE* fptr, char* args)
{
  unsigned ibob = 0;
  multibob_t* multibob = context;

  time_t current = time(0);

  for (ibob = 0; ibob < multibob->nthread; ibob++)
  {
    ibob_thread_t* thread = multibob->threads + ibob;
    fprintf (fptr, "IBOB%02d %s:%d ", ibob+1, 
	     thread->ibob->host, thread->ibob->port);

    if (thread->quit)
      fprintf (fptr, "closed");

    else if (current - thread->alive < 5)
      fprintf (fptr, "alive");

    else
      fprintf (fptr, "dead");

    fprintf (fptr, "\n");
  }
}

/*! set the host and port number of the specified ibob */
int multibob_cmd_hostport (void* context, FILE* fptr, char* args)
{
  unsigned ibob = 0;
  multibob_t* multibob = context;

}

/*! set the target MAC address of the specified ibob */
int multibob_cmd_mac (void* context, FILE* fptr, char* args)
{
}

int multibob_send (multibob_t* multibob, const char* message)
{
  ssize_t length = 0;

#ifdef _DEBUG
  fprintf (stderr, "async send\n");
#endif

  unsigned ibob = 0;
  for (ibob = 0; ibob < multibob->nthread; ibob++)
  {
    ibob_thread_t* thread = multibob->threads + ibob;
    if (ibob_is_open (thread->ibob))
      length = ibob_send_async (thread->ibob, message);

#ifdef _DEBUG
    fprintf (stderr, "sent %d len=%u\n", ibob, length);
#endif
  }

#ifdef _DEBUG
  fprintf (stderr, "receive echo length %u\n", length);
#endif

  for (ibob = 0; ibob < multibob->nthread; ibob++)
  {
#ifdef _DEBUG
    fprintf (stderr, "recv %d: %u\n", ibob, length);
#endif

    ibob_thread_t* thread = multibob->threads + ibob;
    if (ibob_is_open (thread->ibob))
      ibob_recv_echo (thread->ibob, length);
  }
}

int multibob_arm (void* context)
{
#ifdef _DEBUG
  fprintf (stderr, "regwrite reg_arm 0\n");
#endif

  multibob_t* multibob = context;
  multibob_send (multibob, "regwrite reg_arm 0");

#ifdef _DEBUG
  fprintf (stderr, "pause\n");
#endif

  // pause for 10 milliseconds
  struct timeval pause;
  pause.tv_sec=0;
  pause.tv_usec=10000;
  select(0,NULL,NULL,NULL,&pause);

#ifdef _DEBUG
  fprintf (stderr, "regwrite reg_arm 1\n");
#endif
  multibob_send (multibob, "regwrite reg_arm 1");
}

/* defined in start_observation.c */
time_t start_observation( int(*start_routine)(void*), void* arg );

/*! reset packet counter on next UTC second, returned */
int multibob_cmd_arm (void* context, FILE* fptr, char* args)
{
  unsigned ibob = 0;
  multibob_t* multibob = context;

#ifdef _DEBUG
fprintf (stderr, "locking\n");
#endif

  multibob_lock (multibob);

#ifdef _DEBUG
fprintf (stderr, "locked\n");
#endif

  time_t utc = start_observation (multibob_arm, context);

#ifdef _DEBUG
fprintf (stderr, "started\n");
#endif
  multibob_unlock (multibob);

#ifdef _DEBUG
fprintf (stderr, "unlocked\n");
#endif

  char date[64];
  strftime (date, 64, DADA_TIMESTR, gmtime(&utc));

  fprintf (fptr, "%s\n", date);
}

/*! reset packet counter on next UTC second, returned */
int multibob_cmd_quit (void* context, FILE* fptr, char* args)
{
}

void* lock_thread (void* context)
{
  if (!context)
    return 0;

  ibob_thread_t* thread = context;
  if (pthread_mutex_lock (&(thread->mutex)) != 0)
    return 0;

  return context;
}

/*! mutex lock all of the ibob interfaces */
void multibob_lock (multibob_t* multibob)
{
  unsigned ibob = 0;

  /* quickly grab all locks at once */
  for (ibob = 0; ibob < multibob->nthread; ibob++)
  {
#ifdef _DEBUG
    fprintf (stderr, "multibob_lock: launch lock_thread %u\n", ibob);
#endif

    ibob_thread_t* thread = multibob->threads + ibob;
    int err = pthread_create (&(thread->lock), 0, lock_thread, thread);
    if (err)
      fprintf (stderr, "multibob_lock: error starting lock_thread %d - %s\n",
	       ibob, strerror (err));
  }

  /* wait for the locks to be made */
  for (ibob = 0; ibob < multibob->nthread; ibob++)
  {
    ibob_thread_t* thread = multibob->threads + ibob;

#ifdef _DEBUG
    fprintf (stderr, "multibob_lock: join lock_thread %u\n", ibob);
#endif

    void* result = 0;
    pthread_join (thread->lock, &result);
    if (result != thread)
      fprintf (stderr, "multibob_lock: error in lock_thread\n");
  }
}

/*! mutex unlock all of the ibob interfaces */
void multibob_unlock (multibob_t* multibob)
{
  unsigned ibob = 0;

  /* quickly grab all locks at once */
  for (ibob = 0; ibob < multibob->nthread; ibob++)
  {
    ibob_thread_t* thread = multibob->threads + ibob;
    pthread_mutex_unlock (&(thread->mutex));
  }
}

/*! */
int multibob_serve (multibob_t* multibob)
{
  if (!multibob)
    return -1;

  if (multibob->port)
  {
    if (multibob->server)
    {
      fprintf (stderr, "multibob_serve: server already launched \n");
      return -1;
    }

    multibob -> server = command_parse_server_create (multibob -> parser);

    command_parse_server_set_welcome (multibob -> server,
				      "multibob command");

    /* open the command/control port */
    command_parse_serve (multibob->server, multibob->port);

    void* result = 0;
    pthread_join (multibob->server->thread, &result);
  }
  else
  {
    fprintf (stderr, "multibob_serve: stdin/out interface not implemented \n");
    return -1;
  }
}
