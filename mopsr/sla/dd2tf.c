#include "slalib.h"
#include "slamac.h"
void slaDd2tf ( int ndp, double days, char *sign, int ihmsf[4] )
/*
**  - - - - - - - - -
**   s l a D d 2 t f
**  - - - - - - - - -
**
**  Convert an interval in days into hours, minutes, seconds.
**
**  (double precision)
**
**  Given:
**     ndp       int      number of decimal places of seconds
**     days      double   interval in days
**
**  Returned:
**     *sign     char     '+' or '-'
**     ihmsf     int[4]   hours, minutes, seconds, fraction
**
**  Notes:
**
**  1  ndp less than zero is interpreted as zero.
**
**  2  The largest useful value for ndp is determined by the size of
**     days, the format of double floating-point numbers on the target
**     machine, and the risk of overflowing ihmsf[3].  On some
**     architectures, for days up to 1.0, the available floating-point
**     precision corresponds roughly to ndp=12.  However, the practical
**     limit is ndp=9, set by the capacity of a typical 32-bit ihmsf[3].
**
**  3  The absolute value of days may exceed 1.0.  In cases where it
**     does not, it is up to the caller to test for and handle the case
**     where days is very nearly 1.0 and rounds up to 24 hours, by
**     testing for ihmsf[0]=24 and setting ihmsf[0-3] to zero.
**
**  Last revision:   26 December 2004
**
**  Copyright P.T.Wallace.  All rights reserved.
*/

#define D2S 86400.0    /* days to seconds */

{
   double rs, rm, rh, a, ah, am, as, af;


/* Handle sign. */
   *sign = (char) ( ( days < 0.0 ) ?  '-' : '+' );

/* Field units in terms of least significant figure. */
   rs = pow ( 10.0, (double) gmax ( ndp, 0 ) );
   rs = dint ( rs );
   rm = rs * 60.0;
   rh = rm * 60.0;

/* Round interval and express in smallest units required. */
   a = rs * D2S * fabs ( days );
   a = dnint ( a );

/* Separate into fields. */
   ah = a / rh;
   ah = dint ( ah );
   a  = a - ah * rh;
   am = a / rm;
   am = dint ( am );
   a  = a - am * rm;
   as = a / rs;
   as = dint ( as );
   af = a - as * rs;

/* Return results. */
   ihmsf[0] = (int) ah;
   ihmsf[1] = (int) am;
   ihmsf[2] = (int) as;
   ihmsf[3] = (int) af;
}
