#include "daemon.h"

#include <stdlib.h>
#include <unistd.h>

void be_a_daemon ()
{
  pid_t pid = fork();

  if (pid < 0)
    exit(EXIT_FAILURE);
  
  if (pid > 0)
    exit(EXIT_SUCCESS);
  
  /* Create a new SID for the child process */
  if (setsid() < 0)
    exit (EXIT_FAILURE);
  
  /* Change the current working directory */
  if (chdir("/") < 0)
    exit (EXIT_FAILURE);
  
  /* Close out the standard file descriptors */
  close(STDIN_FILENO);
  close(STDOUT_FILENO);
  close(STDERR_FILENO);
}
