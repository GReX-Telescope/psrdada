#include "slalib.h"
#include "slamac.h"
double slaDsepv ( double v1[3], double v2[3] )
/*
**  - - - - - - - - -
**   s l a D s e p v
**  - - - - - - - - -
**
**  Angle between two vectors.
**
**  (double precision)
**
**  Given:
**     v1      double[3]    first vector
**     v2      double[3]    second vector
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
**  Called:  slaDvxv, slaDvn, slaDvdv
**
**  Last revision:   7 May 2000
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
   double v1xv2[3], wv[3], s, c;


/* Modulus of cross product = sine multiplied by the two moduli. */
   slaDvxv ( v1, v2, v1xv2 );
   slaDvn ( v1xv2, wv, &s );

/* Dot product = cosine multiplied by the two moduli. */
   c = slaDvdv ( v1, v2 );

/* Angle between the vectors. */
   return s != 0.0 ? atan2 ( s, c ) : 0.0;

}
