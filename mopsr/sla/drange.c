#include "slalib.h"
#include "slamac.h"
double slaDrange ( double angle )
/*
**  - - - - - - - - - -
**   s l a D r a n g e
**  - - - - - - - - - -
**
**  Normalize angle into range +/- pi.
**
**  (double precision)
**
**  Given:
**     angle     double      the angle in radians
**
**  The result is angle expressed in the range +/- pi.
**
**  Defined in slamac.h:  DPI, D2PI, dmod
**
**  Last revision:   22 July 2004
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
  double w;

  w = dmod ( angle, D2PI );
  return ( fabs ( w ) < DPI ) ? w : w - dsign ( D2PI, angle );
}
