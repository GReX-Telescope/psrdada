#include "multilog.h"

#include <stdlib.h>
#include <stdarg.h>

multilog_t* multilog_open (char syslog)
{
  multilog_t* m = (multilog_t*) malloc (sizeof(multilog_t));

  m->syslog = syslog;
  m->logs = 0;
  m->nlog = 0;

  pthread_mutex_init(&(m->mutex), NULL);

  return m;
}

int multilog_close (multilog_t* m)
{
  pthread_mutex_lock (&(m->mutex));

  if (m->logs)
    free (m->logs);
  m->logs = 0;

  pthread_mutex_unlock (&(m->mutex));
  pthread_mutex_destroy (&(m->mutex));

  free (m);
  return 0;
}

int multilog_add (multilog_t* m, FILE* fptr)
{
  pthread_mutex_lock (&(m->mutex));

  m->logs = realloc (m->logs, (m->nlog+1)*sizeof(multilog_t));
  m->logs[m->nlog] = fptr;
  m->nlog ++;

  pthread_mutex_unlock (&(m->mutex));

  return 0;
}

int multilog (multilog_t* m, int priority, const char* format, ...)
{
  unsigned ilog = 0;
  va_list arguments;

  va_start (arguments, format);

  if (m->syslog)
    vsyslog (priority, format, arguments);

  pthread_mutex_lock (&(m->mutex));

  for (ilog=0; ilog < m->nlog; ilog++)
    vfprintf(m->logs[ilog], format, arguments);

  pthread_mutex_unlock (&(m->mutex));

  va_end (arguments);

  return 0;
}
