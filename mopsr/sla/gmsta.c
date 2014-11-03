#include "slalib.h"
#include "slamac.h"
double slaGmsta ( double date, double ut )
/*
**  - - - - - - - - -
**   s l a G m s t a
**  - - - - - - - - -
**
**  Conversion from Universal Time to Greenwich mean sidereal time,
**  with rounding errors minimized.
**
**  (double precision)
**
**  Given:
**    date   double     UT1 date (MJD: integer part of JD-2400000.5))
**    ut     double     UT1 time (fraction of a day)
**
**  The result is the Greenwich Mean Sidereal Time (double precision,
**  radians, in the range 0 to 2pi).
**
**  There is no restriction on how the UT is apportioned between the
**  date and ut1 arguments.  Either of the two arguments could, for
**  example, be zero and the entire date+time supplied in the other.
**  However, the routine is designed to deliver maximum accuracy when
**  the date argument is a whole number and the ut argument lies in
**  the range 0 to 1, or vice versa.
**
**  The algorithm is based on the IAU 1982 expression (see page S15 of
**  the 1984 Astronomical Almanac).  This is always described as giving
**  the GMST at 0 hours UT1.  In fact, it gives the difference between
**  the GMST and the UT, the steady 4-minutes-per-day drawing-ahead of
**  ST with respect to UT.  When whole days are ignored, the expression
**  happens to equal the GMST at 0 hours UT1 each day.  Note that the
**  factor 1.0027379... does not appear explicitly but in the form of
**  the coefficient 8640184.812866, which is 86400x36525x0.0027379...
**
**  In this routine, the entire UT1 (the sum of the two arguments date
**  and ut) is used directly as the argument for the standard formula.
**  The UT1 is then added, but omitting whole days to conserve accuracy.
**
**  See also the routine slaGmst, which accepts the UT1 as a single
**  argument.  Compared with slaGmst, the extra numerical precision
**  delivered by the present routine is unlikely to be important in
**  an absolute sense, but may be useful when critically comparing
**  algorithms and in applications where two sidereal times close
**  together are differenced.
**
**  Called:  slaDranrm
**
**  Defined in slamac.h:  DS2R, dmod
**
**  Last revision:   13 March 2004
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
   double d1, d2, t;

/* Julian centuries since J2000. */
   if ( date < ut ) {
      d1 = date;
      d2 = ut;
   } else {
      d1 = ut;
      d2 = date;
   }
   t = ( d1 + ( d2 - 51544.5 ) ) / 36525.0;

/* GMST at this UT1. */
   return slaDranrm ( DS2R * ( 24110.54841
                           + ( 8640184.812866
                           + ( 0.093104
                             - 6.2e-6 * t ) * t ) * t
                             + 86400.0 * ( dmod ( d1, 1.0 ) +
                                           dmod ( d2, 1.0 ) ) ) );
}
