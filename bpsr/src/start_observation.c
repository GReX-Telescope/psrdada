/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <sys/time.h>
#include <time.h>

/*! Starts an observation on the next UTC second, returned as time_t

    \param start_routine the function that does the starting
    \param arguments the arguments passed to start_routine
*/

time_t start_observation( int(*start_routine)(void*), void* arg )
{
  start_routine (arg);
  return time(0);
}
