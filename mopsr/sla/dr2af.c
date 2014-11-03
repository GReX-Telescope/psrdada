#include "slalib.h"
#include "slamac.h"
void slaDr2af ( int ndp, double angle, char *sign, int idmsf[4] )
/*
**  - - - - - - - - -
**   s l a D r 2 a f
**  - - - - - - - - -
**
**  Convert an angle in radians to degrees, arcminutes, arcseconds.
**
**  (double precision)
**
**  Given:
**     ndp       int          number of decimal places of arcseconds
**     angle     double       angle in radians
**
**  Returned:
**     sign      char*        '+' or '-'
**     idmsf     int[4]       degrees, arcminutes, arcseconds, fraction
**
**  Notes:
**
**  1  ndp less than zero is interpreted as zero.
**
**  2  The largest useful value for ndp is determined by the size of
**     angle, the format of double floating-point numbers on the target
**     machine, and the risk of overflowing idmsf[3].  On some
**     architectures, for angle up to 2pi, the available floating-point
**     precision corresponds roughly to ndp=12.  However, the practical
**     limit is ndp=9, set by the capacity of a typical 32-bit idmsf[3].
**
**  3  The absolute value of angle may exceed 2pi.  In cases where it
**     does not, it is up to the caller to test for and handle the case
**     where angle is very nearly 2pi and rounds up to 360 deg, by
**     testing for idmsf[0]=360 and setting idmsf[0-3] to zero.
**
**  Called:
**     slaDd2tf
**
**  Defined in slamac.h:  D15B2P
**
**  Last revision:   26 December 2004
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
/* Scale then use days to h,m,s routine */
   slaDd2tf ( ndp, angle * D15B2P, sign, idmsf );
}
