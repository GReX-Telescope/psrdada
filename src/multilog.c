#include "multilog.h"

#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

// #define _DEBUG 1

multilog_t* multilog_open (const char* program_name, char syslog)
{
  multilog_t* m = (multilog_t*) malloc (sizeof(multilog_t));

  if (syslog)
    openlog (program_name, LOG_CONS, LOG_USER);

  m->syslog = syslog;
  m->name = strdup(program_name);
  m->logs = 0;
  m->nlog = 0;
  m->port = 0;

  pthread_mutex_init(&(m->mutex), NULL);

  return m;
}

int multilog_close (multilog_t* m)
{
  pthread_mutex_lock (&(m->mutex));

  free (m->name);
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

#ifdef _DEBUG
  fprintf (stderr, "multilog: %d logs\n", m->nlog);
#endif

  for (ilog=0; ilog < m->nlog; ilog++)  {
    if (ferror (m->logs[ilog]))  {
#ifdef _DEBUG
      fprintf (stderr, "multilog: error on log[%d]", ilog);
#endif
      fclose (m->logs[ilog]);
      m->logs[ilog] = m->logs[m->nlog-1];
      m->nlog --;
      ilog --;
    }
    else {
      fprintf (m->logs[ilog], "%s: ", m->name);
      if (vfprintf (m->logs[ilog], format, arguments) < 0)
	perror ("multilog: error vfprintf");
    }
  }

  pthread_mutex_unlock (&(m->mutex));

  va_end (arguments);

  return 0;
}
