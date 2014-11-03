#include "slalib.h"
#include "slamac.h"
void slaDr2tf ( int ndp, double angle, char *sign, int ihmsf[4] )
/*
**  - - - - - - - - -
**   s l a D r 2 t f
**  - - - - - - - - -
**
**  Convert an angle in radians to hours, minutes, seconds.
**
**  (double precision)
**
**  Given:
**     ndp       int          number of decimal places of seconds
**     angle     double       angle in radians
**
**  Returned:
**     sign      char*        '+' or '-'
**     ihmsf     int[4]       hours, minutes, seconds, fraction
**
**  Notes:
**
**  1  ndp less than zero is interpreted as zero.
**
**  2  The largest useful value for ndp is determined by the size of
**     angle, the format of double floating-point numbers on the target
**     machine, and the risk of overflowing ihmsf[3].  On some
**     architectures, for angle up to 2pi, the available floating-
**     point precision corresponds roughly to ndp=12.  However, the
**     practical limit is ndp=9, set by the capacity of a typical 32-bit
**     ihmsf[3].
**
**  3  The absolute value of angle may exceed 2pi.  In cases where it
**     does not, it is up to the caller to test for and handle the case
**     where angle is very nearly 2pi and rounds up to 24 hours, by
**     testing for ihmsf[0]=24 and setting ihmsf[0-3] to zero.
**
**  Called:
**     slaDd2tf
**
**  Defined in slamac.h:  D2PI
**
**  Last revision:   18 November 1993
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
/* Scale then use days to h,m,s routine */
   slaDd2tf ( ndp, angle / D2PI, sign, ihmsf );
}
