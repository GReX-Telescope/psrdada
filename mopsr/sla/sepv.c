#include "slalib.h"
#include "slamac.h"
float slaSepv ( float v1[3], float v2[3] )
/*
**  - - - - - - - -
**   s l a S e p v
**  - - - - - - - -
**
**  Angle between two vectors.
**
**  (single precision)
**
**  Given:
**     v1     float[3]     first vector
**     v2     float[3]     second vector
**
**  The result is the angle, in radians, between the two vectors.  It
**  is always positive.
**
**  Notes:
**
**  1  There is no requirement for the vectors to be unit length.
**
**  2  If either vector is null, zero is returned.
**
**  3  The simplest formulation would use dot product alone.  However,
**     this would reduce the accuracy for angles near zero and pi.  The
**     algorithm uses both cross product and dot product, which maintains
**     accuracy for all sizes of angle.
**
**  Called:  slaDsepv
**
**  Last revision:   7 May 2000
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
   int i;
   double dv1[3], dv2[3];


/* Use double precision version. */
   for ( i = 0; i < 3; i++ ) {
      dv1[i] = (double) v1[i];
      dv2[i] = (double) v2[i];
   }
   return (float) slaDsepv ( dv1, dv2 );

}
