#include "multilog.h"

#include <unistd.h>
#include <time.h>

int main ()
{
  multilog_t* log = 0;
  time_t seconds;

  fprintf (stderr, "Opening multilog\n");
  log = multilog_open (0);

  // fprintf (stderr, "Adding stderr to multilog\n");
  // multilog_add (log, stderr);

  multilog_serve (log, 2020);

  while (1) {
    time(&seconds);
    multilog (log, LOG_INFO, ctime(&seconds));
    sleep (1);
  }

  fprintf (stderr, "Closing log\n");
  multilog_close (log);

  return 0;
}

