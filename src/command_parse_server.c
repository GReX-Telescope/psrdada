#include "command_parse_server.h"
#include "sock.h"

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// #define _DEBUG 1

/* create a new command parse server */
command_parse_server_t* 
command_parse_server_create (command_parse_t* parser)
{
  command_parse_server_t* server = 0;

  if (!parser)
    return 0;

  server = (command_parse_server_t*) malloc (sizeof(command_parse_server_t));

  server -> parser = parser;
  server -> welcome = 0;
  server -> prompt = strdup ("> ");
  server -> ok = strdup ("ok\n");
  server -> fail = strdup ("fail\n");
  server -> port = 0;

  pthread_mutex_init(&(server->mutex), NULL);
  
  return server;
}

/* destroy a command parse server */
int command_parse_server_destroy (command_parse_server_t* server)
{
  if (!server)
    return -1;

  if (server->welcome)
    free (server->welcome);
  if (server->prompt)
    free (server->prompt);
  if (server->ok)
    free (server->ok);
  if (server->fail)
    free (server->fail);

  free (server);
  return 0;
}

typedef struct {

  /* command parse server */
  command_parse_server_t* server;

  /* I/O stream from which to read commands */
  FILE* input;

  /* I/O stream to which output is written */
  FILE* output;

} command_parse_thread_t;

/*! this thread does the actual command I/O */
static void* command_parser (void * arg)
{
  command_parse_thread_t* parser = (command_parse_thread_t*) arg;
  command_parse_server_t* server = parser->server;

  char buffer [1024];
  int ret;

#ifdef _DEBUG
  fprintf (stderr, "command_parser: server=%p\n", server);
  fprintf (stderr, "command_parser: parser=%p\n", server->parser);
  fprintf (stderr, "command_parser: input=%p\n",  parser->input);
  fprintf (stderr, "command_parser: output=%p\n", parser->output);
#endif

  assert (server != 0);
  assert (server->parser != 0);
  assert (parser->input  != 0);
  assert (parser->output != 0);

  if (server->welcome) {
#ifdef _DEBUG
    fprintf (stderr, "command_parser: print welcome %s\n", server->welcome);
#endif
    fprintf (parser->output, "%s\n", server->welcome);
  }

  while (!feof (parser->output)) {

    if (server->prompt)
      fprintf (parser->output, server->prompt);

    if (!fgets (buffer, 1024, parser->input) || feof (parser->input))
      break;

    ret = command_parse_output (server->parser, buffer, parser->output);

    if (ret == COMMAND_PARSE_EXIT)
      break;

    if (ret < 0 && server->fail)
      fprintf (parser->output, server->fail);

    if (ret >= 0 && server->ok)
      fprintf (parser->output, server->ok);

  }

#ifdef _DEBUG
  fprintf (stderr, "command_parser: closing\n");
#endif

  fclose (parser->input);
  fclose (parser->output);

  free (parser);

  return 0;
}

static void* command_parse_server (void * arg)
{
  command_parse_server_t* server = (command_parse_server_t*) arg;
  command_parse_thread_t* parser;
 
  int listen_fd = 0;
  int comm_fd = 0;
  FILE* fptr = 0;

  pthread_t tmp_thread;

  listen_fd = sock_create (&(server->port));
  if (listen_fd < 0)  {
    perror ("command_parse_server: Error creating socket");
    return 0;
  }

  while (server->port) {

    comm_fd = sock_accept (listen_fd);

    if (comm_fd < 0)  {
      perror ("command_parse_server: Error accepting connection");
      return 0;
    }

#ifdef _DEBUG
    fprintf (stderr, "command_parse_server: connection. server=%p\n", server);
#endif

    parser = (command_parse_thread_t*) malloc (sizeof(command_parse_thread_t));
    parser->server = server;

    fptr = fdopen (comm_fd, "r");
    if (!fptr)  {
      perror ("command_parse_server: Error creating I/O stream");
      return 0;
    }

    parser->input = fptr;
#ifdef _DEBUG
    fprintf (stderr, "command_parse_server: input=%p\n", parser->input);
#endif

    fptr = fdopen (comm_fd, "w");
    if (!fptr)  {
      perror ("command_parse_server: Error creating I/O stream");
      return 0;
    }

    /* do not buffer the output */
    setbuf (fptr, 0);

    parser->output = fptr;

#ifdef _DEBUG
    fprintf (stderr, "command_parse_server: output=%p\n", parser->output);
#endif

    if (pthread_create (&tmp_thread, 0, command_parser, parser) < 0) {
      perror ("command_parse_serve: Error creating new thread");
      return 0;
    }

    /* thread cannot be joined; resources will be deallocated on exit */
    pthread_detach (tmp_thread);

  }

  return 0;
}
/* set the welcome message */
int command_parse_server_set_welcome (command_parse_server_t* s, const char* w)
{
  if (!s)
    return -1;
  if (s->welcome)
    free (s->welcome);
  if (w)
    s->welcome = strdup (w);

  return 0;
}

int command_parse_serve (command_parse_server_t* server, int port)
{
  sighandler_t handler = signal (SIGPIPE, SIG_IGN);
  if (handler != SIG_DFL)
    signal (SIGPIPE, handler);

#ifdef _DEBUG
  fprintf (stderr, "command_parse_serve: server=%p\n", server);
#endif

  server->port = port;

  if (pthread_create (&(server->thread), 0, command_parse_server, server) < 0)
    {
      perror ("command_parse_serve: Error creating new thread");
      return -1;
    }

  return 0;
}

