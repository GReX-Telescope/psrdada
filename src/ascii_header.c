#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "ascii_header.h"

#define STRLEN 128

static char* whitespace = " \t\n";

int ascii_header_set (char* header, const char* keyword,
		      const char* format, ...)
{
  va_list arguments;

  char value[STRLEN];
  char* eol = 0;
  char* dup = 0;
  int ret = 0;

  /* find the keyword (also the insertion point) */
  char* key = strstr (header, keyword);  

  if (key) {
    /* if the keyword is present, find the first '#' or '\n' to follow it */
    eol = key + strcspn (key, "#\n");
  }
  else {
    /* if the keyword is not present, append to the end, before "DATA" */
    eol = strstr (header, "DATA\n");
    if (eol)
      /* insert in front of DATA */
      key = eol;
    else
      /* insert at end of string */
      key = header + strlen (header);
  }

  va_start (arguments, format);
  ret = vsnprintf (value, STRLEN, format, arguments);
  va_end (arguments);

  if (ret < 0) {
    perror ("ascii_header_set: error snprintf\n");
    return -1;
  }

  if (eol)
    /* make a copy */
    dup = strdup (eol);

  /* %Xs dictates only a minumum string length */
  if (sprintf (key, "%-12s %-20s   ", keyword, value) < 0) {
    if (dup)
      free (dup);
    perror ("ascii_header_set: error sprintf\n");
    return -1;
  }

  if (dup) {
    strcat (key, dup);
    free (dup);
  }
  else
    strcat (key, "\n");

  return 0;
}

int ascii_header_get (const char* header, const char* keyword,
		      const char* format, ...)
{
  va_list arguments;

  char* value = 0;
  int ret = 0;

  /* find the keyword */
  char* key = strstr (header, keyword);
  if (!key)
    return -1;

  /* find the value after the keyword */
  value = key + strcspn (key, whitespace);

  /* parse the value */
  va_start (arguments, format);
  ret = vsscanf (value, format, arguments);
  va_end (arguments);

  return ret;
}
