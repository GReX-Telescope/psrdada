#include "command_parse.h"
#include "sock.h"

#include <signal.h>
#include <stdlib.h>

typedef struct {

  /* command parser */
  command_parse_t parse;

  /* I/O stream from which to read commands */
  FILE* input;

  /* thread id */
  pthread_t thread;

  /* exit flag */
  char exit;

} command_parser_t;


static void* command_parser (void * arg)
{
  command_parser_t* parser = (command_parser_t*) arg;
  char buffer [1024];

  while (!parser->exit && !feof(parser->input)) {

    if (fgets (buffer, 1024, parser->input) == 0)
      break;

    command_parse (&(parser->parse), buffer);

  }

  fclose (parser->input);
  free (parser);

  return 0;
}



static void* command_parse_server (void * arg)
{
  command_parse_t* parse = (command_parse_t*) arg;

  int listen_fd = 0;
  int comm_fd = 0;
  FILE* fptr = 0;

  listen_fd = sock_create (&(parse->port));
  if (listen_fd < 0)  {
    perror ("command_parse_server: Error creating socket");
    return 0;
  }

  while (parse->port) {

    comm_fd = sock_accept (listen_fd);

    if (comm_fd < 0)  {
      perror ("command_parse_server: Error accepting connection");
      return 0;
    }

    fptr = fdopen (comm_fd, "w");
    if (!fptr)  {
      perror ("command_parse_server: Error creating I/O stream");
      return 0;
    }

    /* line buffer the output */
    setvbuf (fptr, 0, _IOLBF, 0);

    command_parser_t* parser =
      (command_parser_t*) malloc (sizeof(command_parser_t));

    parser->parse = *parse;
    parser->parse.reply = fptr;

    fptr = fdopen (comm_fd, "r");
    if (!fptr)  {
      perror ("command_parse_server: Error creating I/O stream");
      return 0;
    }

    parser->input = fptr;

    if (pthread_create (&(parser->thread), 0, command_parser, parser) < 0) {
      perror ("command_parse_serve: Error creating new thread");
      return -1;
    }

  }

  return 0;
}

int command_parse_serve (command_parse_t* log, int port)
{
  sighandler_t handler = signal (SIGPIPE, SIG_IGN);
  if (handler != SIG_DFL)
    signal (SIGPIPE, handler);

  log->port = port;
  if (pthread_create (&(log->thread), 0, command_parse_server, log) < 0) {
    perror ("command_parse_serve: Error creating new thread");
    return -1;
  }
  return 0;
}

