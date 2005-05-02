/* $Source$
   $Revision$
   $Date$
   $Author$ */

#ifndef DADA_UTC_H
#define DADA_UTC_H

#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

  /*! parse a string into struct tm; return equivalent time_t */
  time_t str2tm (struct tm* time, const char* str);

  /*! parse a string and return equivalent time_t */
  time_t str2time (const char* str);

#ifdef __cplusplus
}
#endif

#endif

