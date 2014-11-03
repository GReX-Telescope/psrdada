#include "slalib.h"
#include "slamac.h"
double slaDsep ( double a1, double b1, double a2, double b2 )
/*
**  - - - - - - - -
**   s l a D s e p
**  - - - - - - - -
**
**  Angle between two points on a sphere.
**
**  (double precision)
**
**  Given:
**     a1,b1    double    spherical coordinates of one point
**     a2,b2    double    spherical coordinates of the other point
**
**  (The spherical coordinates are [RA,Dec], [Long,Lat] etc, in radians.)
**
**  The result is the angle, in radians, between the two points.  It
**  is always positive.
**
**  Called:  slaDcs2c, slaDsepv
**
**  Last revision:   7 May 2000
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
   double v1[3], v2[3];


/* Convert coordinates from spherical to Cartesian. */
   slaDcs2c ( a1, b1, v1 );
   slaDcs2c ( a2, b2, v2 );

/* Angle between the vectors. */
   return slaDsepv ( v1, v2 );

}
