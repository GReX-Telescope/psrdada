#include "multilog.h"

#include <stdlib.h>
#include <stdarg.h>

multilog_t* multilog_open (char syslog)
{
  multilog_t* m = (multilog_t*) malloc (sizeof(multilog_t));

  m->syslog = syslog;
  m->logs = 0;
  m->nlog = 0;

  return m;
}

int multilog_close (multilog_t* m)
{
  if (m->logs)
    free (m->logs);
  free (m);
  return 0;
}

int multilog_add (multilog_t* m, FILE* fptr)
{
  m->nlog ++;
  m->logs = realloc (m->logs, m->nlog);
  m->logs[m->nlog-1] = fptr;

  return 0;
}

int multilog (multilog_t* m, int priority, const char* format, ...)
{
  unsigned ilog = 0;
  va_list arguments;

  va_start (arguments, format);

  if (m->syslog)
    vsyslog (priority, format, arguments);

  for (ilog=0; ilog < m->nlog; ilog++)
    vfprintf(m->logs[ilog], format, arguments);

  va_end (arguments);

  return 0;
}
