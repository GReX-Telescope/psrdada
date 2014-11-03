#include "slalib.h"
#include "slamac.h"
void slaCd2tf ( int ndp, float days, char *sign, int ihmsf[4] )
/*
**  - - - - - - - - -
**   s l a C d 2 t f
**  - - - - - - - - -
**
**  Convert an interval in days into hours, minutes, seconds.
**
**  (single precision)
**
**  Given:
**     ndp       int      number of decimal places of seconds
**     days      float    interval in days
**
**  Returned:
**     sign      char*    '+' or '-'
**     ihmsf     int[4]   hours, minutes, seconds, fraction
**
**  Notes:
**
**  1  ndp less than zero is interpreted as zero.
**
**  2  The largest useful value for ndp is determined by the size of
**     days, the format of float floating-point numbers on the target
**     machine, and the risk of overflowing ihmsf[3].  On some
**     architectures, for days up to 1.0f, the available floating-
**     point precision corresponds roughly to ndp=3.  This is well below
**     the ultimate limit of ndp=9 set by the capacity of a typical
**     32-bit ihmsf[3].
**
**  3  The absolute value of days may exceed 1.0f.  In cases where it
**     does not, it is up to the caller to test for and handle the case
**     where days is very nearly 1.0f and rounds up to 24 hours, by
**     testing for ihmsf[0]=24 and setting ihmsf[0-3] to zero.
**
**  Called:  slaDd2tf
**
**  Last revision:   26 December 2004
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
/* Use double version */
   slaDd2tf ( ndp, (double) days, sign, ihmsf );
}
