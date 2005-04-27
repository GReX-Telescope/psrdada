#include "dada_pwc.h"

#include <stdlib.h>

/*! Create a new DADA primary write client connection */
dada_pwc_t* dada_pwc_create ()
{
  dada_pwc_t* pwc = (dada_pwc_t*) malloc (sizeof(dada_pwc_t));

  /* default header size */
  pwc -> header_size = 4096;

  /* default command port */
  pwc -> port = 0xdada;

  pthread_mutex_init(&(pwc->mutex), NULL);

  return pwc;
}

/*! Destroy a DADA primary write client connection */
int dada_pwc_destroy (dada_pwc_t* pwc)
{
  return 0;
}

/*! Check to see if a command has arrived */
int dada_pwc_command_check (dada_pwc_t* pwc)
{
  return 0;
}

/*! Get the next command from the connection; wait until command received */
int dada_pwc_command_get (dada_pwc_t* pwc)
{
  return 0;
}

/*! Reply to the last command received */
int dada_pwc_command_reply (dada_pwc_t* pwc)
{
  return 0;
}

