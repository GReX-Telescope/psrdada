#include "command_parse.h"

#include <string.h>
#include <stdlib.h>

// typedef int (*command) (void* context, FILE* reply, char* arguments);

int command_parse_help (void* context, FILE* fptr, char* arg)
{
  command_parse_t* parser = (command_parse_t*) context;
  unsigned icmd;
  
  fprintf (fptr, "Available commands:\n\n");

  for (icmd=0; icmd < parser->ncmd; icmd++)
    fprintf (fptr, "%s\t%s\n",
	     parser->cmds[icmd].name,
	     parser->cmds[icmd].help);

  return 0;
}


/* create a new command parser */
command_parse_t* command_parse_create ()
{
  command_parse_t* c = malloc (sizeof(command_parse_t));
  c -> cmds = 0;
  c -> ncmd = 0;
  c -> reply = stdout;

  command_parse_add (c, command_parse_help, c, "help", "print this list", 0);
  return c;
}

/* destroy a command parser */
int command_parse_destroy (command_parse_t* parser)
{
  unsigned icmd = 0;
  for (icmd=0; icmd<parser->ncmd; icmd++) {
    free (parser->cmds[icmd].name);
    free (parser->cmds[icmd].help);
    free (parser->cmds[icmd].more);
  }

  free (parser->cmds);
  free (parser);

  return 0;
}

/* set the stream to be used when replying */
int command_parse_reply (command_parse_t* parser, FILE* fptr)
{
  parser -> reply = fptr;
  return 0;
}

/* add a command to the list of available commands */
int command_parse_add (command_parse_t* parser, 
		       command cmd, void* context,
		       const char* command_name,
		       const char* short_help,
		       const char* long_help)
{
  if (!command_name) {
    fprintf (stderr, "command_parse_add: command name not provided\n");
    return -1;
  }

  parser->cmds = (command_t*) realloc (parser->cmds,
				       (parser->ncmd+1)*sizeof(command_t));

  parser->cmds[parser->ncmd].cmd = cmd;
  parser->cmds[parser->ncmd].context = context;

  parser->cmds[parser->ncmd].name = strdup(command_name);

  if (short_help)
    parser->cmds[parser->ncmd].help = strdup(short_help);
  else
    parser->cmds[parser->ncmd].help = 0;

  if (long_help)
    parser->cmds[parser->ncmd].more = strdup(long_help);
  else
    parser->cmds[parser->ncmd].more = 0;

  parser->ncmd ++;

  return 0;
}

/* parse a command */
int command_parse (command_parse_t* parser, const char* command)
{
  char* dup = strdup (command);
  char* key = strsep (&dup, " \t\n");
  char* arg = dup;
  unsigned icmd;

  for (icmd=0; icmd < parser->ncmd; icmd++)
    if (strcmp (key, parser->cmds[icmd].name) == 0)
      return parser->cmds[icmd].cmd (parser->cmds[icmd].context,
				     parser->reply, arg);

  return -1;
}

