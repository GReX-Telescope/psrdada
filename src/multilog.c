#include "multilog.h"

#include <stdlib.h>
#include <stdarg.h>

/*! Initialize parameter values */
int multilog_init (multilog_t* m)
{
  m->syslog = 0;
  m->logs = 0;
  m->nlog = 0;

  return 0;
}

/*! Add a listener to the multilog */
int multilog_add (multilog_t* m, FILE* fptr)
{
  m->nlog ++;
  m->logs = realloc (m->logs, m->nlog);
  m->logs[m->nlog-1] = fptr;

  return 0;
}

/*! Write a message to all listening streams */
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
