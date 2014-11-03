#include "slalib.h"
#include "slamac.h"
double slaDpav ( double v1 [ 3 ], double v2 [ 3 ] )
/*
**  - - - - - - - -
**   s l a D p a v
**  - - - - - - - -
**
**  Position angle of one celestial direction with respect to another.
**
**  (double precision)
**
**  Given:
**     v1    double[3]    direction cosines of one point
**     v2    double[3]    direction cosines of the other point
**
**  (The coordinate frames correspond to RA,Dec, Long,Lat etc.)
**
**  The result is the bearing (position angle), in radians, of point
**  v2 with respect to point v1.  It is in the range +/- pi.  The
**  sense is such that if v2 is a small distance east of v1, the
**  bearing is about +pi/2.  Zero is returned if the two points
**  are coincident.
**
**  The vectors v1 and v2 need not be unit vectors.
**
**  The routine slaDbear performs an equivalent function except
**  that the points are specified in the form of spherical
**  coordinates.
**
**  Last revision:   12 December 1996
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
   double x0, y0, z0, w, x1, y1, z1, s, c;


/* Unit vector to point 1. */
   x0 = v1 [ 0 ];
   y0 = v1 [ 1 ];
   z0 = v1 [ 2 ];
   w = sqrt ( x0 * x0 + y0 * y0 + z0 * z0 );
   if ( w != 0.0 ) { x0 /= w; y0 /= w; z0 /= w; }

/* Vector to point 2. */
   x1 = v2 [ 0 ];
   y1 = v2 [ 1 ];
   z1 = v2 [ 2 ];

/* Position angle. */
   s = y1 * x0 - x1 * y0;
   c = z1 * ( x0 * x0 + y0 * y0 ) - z0 * ( x1 * x0 + y1 * y0 );
   return ( s != 0.0 || c != 0.0 ) ? atan2 ( s, c ) : 0.0;
}
