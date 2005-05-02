#include "dada_pwc.h"
#include "multilog.h"

int main ()
{
  dada_primary_t* pwc = 0;
  int command = 0;
  time_t utc = 0;

  fprintf (stderr, "Creating dada_primary\n");
  pwc = dada_primary_create ();

  while (!dada_primary_quit (pwc)) {

    command = dada_primary_command_get (pwc);

    fprintf (stderr, "command = %d\n", command);

    dada_primary_command_ack (pwc, 1);

  }

  fprintf (stderr, "Destroying pwc\n");
  dada_primary_destroy (pwc);

  return 0;
}

