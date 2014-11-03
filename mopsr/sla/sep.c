#include "slalib.h"
#include "slamac.h"
float slaSep ( float a1, float b1, float a2, float b2 )
/*
**  - - - - - - -
**   s l a S e p
**  - - - - - - -
**
**  Angle between two points on a sphere.
**
**  (single precision)
**
**  Given:
**     a1,b1     float     spherical coordinates of one point
**     a2,b2     float     spherical coordinates of the other point
**
**  (The spherical coordinates are [RA,Dec], [Long,Lat] etc, in radians.)
**
**  The result is the angle, in radians, between the two points.  It is
**  always positive.
**
**  Called:  slaDsep
**
**  Last revision:   7 May 2000
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{

/* Use double precision version. */
   return (float) slaDsep( (double) a1, (double) b1,
                           (double) a2, (double) b2 );

}
