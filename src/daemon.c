#include "daemon.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define PS_IOERR 1

void be_a_daemon ()
{
  // fork off the parent process
  pid_t pid = fork();

  if (pid < 0)
    exit(EXIT_FAILURE);
  
  // if the pid is good, then we can exit the parent process
  if (pid > 0)
    exit(EXIT_SUCCESS);

  // change the file mode mask
  umask (0);
  
  // create a new SID for the child process
  if (setsid() < 0)
    exit (EXIT_FAILURE);
  
  // change the current working directory 
  if (chdir("/") < 0)
    exit (EXIT_FAILURE);
  
  // Close out the standard file descriptors
  close(STDIN_FILENO);
  close(STDOUT_FILENO);
  close(STDERR_FILENO);
}


int be_a_daemon_with_log(char * logfile) 
{

  // fork off the parent process
  pid_t pid = fork();

  if (pid < 0)
    exit(EXIT_FAILURE);

  // if the pid is good, then we can exit the parent process
  if (pid > 0)
    exit(EXIT_SUCCESS);

  // change the file mode mask
  umask (0);

  // create a new SID for the child process
  if (setsid() < 0)
    exit (EXIT_FAILURE);

  // change the current working directory 
  if (chdir("/") < 0)
    exit (EXIT_FAILURE);

  // close STDIN and reopen it as /dev/null
  close(STDIN_FILENO);
  if (open("/dev/null", O_RDONLY) < 0) {
    return (PS_IOERR);
  }

  // if we are redirecting the STDERR and STDOUT to a logfile
  if (logfile)
  {
    // open the logfile
    int fd = open(logfile, O_RDWR|O_CREAT|O_APPEND, S_IRUSR|S_IWUSR|S_IRGRP);
    if (fd == -1) 
      return(PS_IOERR);

    // dup2 will close STD[OUT|ERR]_FILENO if neccesary
    dup2(fd, STDOUT_FILENO);
    dup2(fd, STDERR_FILENO);

    close(fd);

  } 
  else // just close STDOUT and STDERR
  {
    close(STDOUT_FILENO);
    close(STDERR_FILENO);
  }
  return 0;
}
