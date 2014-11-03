#include "slalib.h"
#include "slamac.h"
void slaNut ( double date, double rmatn[3][3] )
/*
**  - - - - - - -
**   s l a N u t
**  - - - - - - -
**
**  Form the matrix of nutation for a given date - Shirai & Fukushima
**  2001 theory
**
**  (double precision)
**
**  Reference:
**     Shirai, T. & Fukushima, T., Astron.J. 121, 3270-3283 (2001).
**
**  Given:
**     date   double        TDB (loosely ET) as Modified Julian Date
**                                           (=JD-2400000.5)
**
**  Returned:
**     rmatn  double[3][3]  nutation matrix
**
**  The matrix is in the sense   v(true)  =  rmatn * v(mean) .
**
**  Called:   slaNutc, slaDeuler
**
**  Last revision:   17 September 2001
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
   double dpsi, deps, eps0;

/* Nutation components and mean obliquity */
   slaNutc ( date, &dpsi, &deps, &eps0 );

/* Rotation matrix */
   slaDeuler ( "xzx", eps0, -dpsi, - ( eps0 + deps ), rmatn );
}
