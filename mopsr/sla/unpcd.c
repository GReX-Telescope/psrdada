#include "slalib.h"
#include "slamac.h"
void slaUnpcd ( double disco, double *x, double *y )
/*
**  - - - - - - - - -
**   s l a U n p c d
**  - - - - - - - - -
**
**  Remove pincushion/barrel distortion from a distorted [x,y] to give
**  tangent-plane [x,y].
**
**  Given:
**     disco    double      pincushion/barrel distortion coefficient
**     x,y      double*     distorted coordinates
**
**  Returned:
**     x,y      double*     tangent-plane coordinates
**
**  Defined in slamac.h:
**     D2PI     double      2*pi
**
**  Notes:
**
**   1)  The distortion is of the form RP = R*(1 + C*R^2), where R is
**       the radial distance from the tangent point, C is the disco
**       argument, and RP is the radial distance in the presence of
**       the distortion.
**
**   2)  For pincushion distortion, C is +ve;  for barrel distortion,
**       C is -ve.
**
**   3)  For x,y in "radians" - units of one projection radius,
**       which in the case of a photograph is the focal length of
**       the camera - the following disco values apply:
**
**           geometry          disco
**
**           astrograph         0.0
**           schmidt           -0.3333
**           AAT PF doublet  +147.069
**           AAT PF triplet  +178.585
**           AAT f/8          +21.20
**           JKT f/8          +13.32
**
**   4)  The present routine is a rigorous inverse of the companion
**       routine slaPcd.  The expression for RP in Note 1 is rewritten
**       in the form x^3+a*x+b=0 and solved by standard techniques.
**
**   5)  Cases where the cubic has multiple real roots can sometimes
**       occur, corresponding to extreme instances of barrel distortion
**       where up to three different undistorted [X,Y]s all produce the
**       same distorted [X,Y].  However, only one solution is returned,
**       the one that produces the smallest change in [X,Y].
**
**  Last revision:   3 January 2003
**
**  Copyright P.T.Wallace.  All rights reserved.
*/

#define THIRD 1.0/3.0
{
   double rp, q, r, d, w, s, t, f, c, c2, t3, f1, f2, f3, w1, w2, w3;



/* Distance of the point from the origin. */
   rp  =  sqrt ( (*x)*(*x) + (*y)*(*y) );

/* If zero, or if no distortion, no action is necessary. */
   if ( rp != 0.0 && disco != 0.0 ) {

   /* Begin algebraic solution. */
      q = 1.0 / ( 3.0 * disco );
      r = rp / ( 2.0 * disco );
      w = q*q*q + r*r;

   /* Continue if one real root, or three of which only one is positive. */
      if ( w >= 0.0 ) {
         d = sqrt ( w );
         w = r + d;
         s = pow ( fabs ( w ), THIRD );
         if ( w < 0.0 ) s = -s;
         w = r - d;
         t = pow ( fabs ( w ), THIRD );
         if ( w < 0.0 ) t = -t;
         f = s + t;
      } else {

      /* Three different real roots:  use geometrical method instead. */
         w = 2.0 / sqrt ( -3.0 * disco );
         c = 4.0 * rp / ( disco * w*w*w );
         c2 = c * c;
         s = c2 < 1.0 ? sqrt ( 1.0 - c2 ) : 0.0;
         t3 = atan2 ( s, c );

      /* The three solutions. */
         f1 = w * cos( ( D2PI - t3 ) / 3.0 );
         f2 = w * cos( ( t3 ) / 3.0 );
         f3 = w * cos( ( D2PI + t3 ) / 3.0 );

      /* Pick the one that moves [x,y] least. */
         w1 = fabs ( f1 - rp );
         w2 = fabs ( f2 - rp );
         w3 = fabs ( f3 - rp );
         f = w1 < w2 ? ( w1 < w3 ? f1 : f3 ) : ( w2 < w3 ? f2 : f3 );
      }

   /* Remove the distortion. */
      f /= rp;
      *x *= f;
      *y *= f;
   }
}
