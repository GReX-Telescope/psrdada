#include "dada_pwc.h"
#include "multilog.h"

int main ()
{
  dada_pwc_t* pwc = 0;
  int command = 0;
  time_t utc = 0;

  fprintf (stderr, "Creating dada_pwc\n");
  pwc = dada_pwc_create ();

  if (dada_pwc_serve (pwc) < 0) {
    fprintf (stderr, "test_dada_pwc: could not start server\n");
    return -1;
  }

  while (!dada_pwc_quit (pwc)) {

    command = dada_pwc_command_get (pwc);

    fprintf (stderr, "command = %d\n", command);

    switch (command) {

    case dada_pwc_header:
      fprintf (stderr, "HEADER=%s", pwc->header);
      dada_pwc_command_ack (pwc, dada_pwc_prepared);
      break;

    case dada_pwc_clock:
      fprintf (stderr, "start clocking\n");
      dada_pwc_command_ack (pwc, dada_pwc_clocking);
      break;

    case dada_pwc_record_start:
      fprintf (stderr, "clocking->recording\n");
      dada_pwc_command_ack (pwc, dada_pwc_recording);
      break;
      
    case dada_pwc_record_stop:
      fprintf (stderr, "recording->clocking\n");
      dada_pwc_command_ack (pwc, dada_pwc_clocking);
      break;

    case dada_pwc_start:
      fprintf (stderr, "start recording\n");
      dada_pwc_command_ack (pwc, dada_pwc_recording);
      break;

    case dada_pwc_stop:
      fprintf (stderr, "stopping\n");
      dada_pwc_command_ack (pwc, dada_pwc_idle);
      break;

    }

  }

  fprintf (stderr, "Destroying pwc\n");
  dada_pwc_destroy (pwc);

  return 0;
}

