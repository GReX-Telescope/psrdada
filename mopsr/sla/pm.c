#include "slalib.h"
#include "slamac.h"
void slaPm ( double r0, double d0, double pr, double pd,
             double px, double rv, double ep0, double ep1,
             double *r1, double *d1 )
/*
**  - - - - - -
**   s l a P m
**  - - - - - -
**
**  Apply corrections for proper motion to a star RA,Dec.
**
**  (double precision)
**
**  References:
**     1984 Astronomical Almanac, pp B39-B41.
**     (also Lederle & Schwan, Astron. Astrophys. 134, 1-6, 1984)
**
**  Given:
**     r0,d0    double     RA,Dec at epoch ep0 (rad)
**     pr,pd    double     proper motions:  RA,Dec changes per year of epoch
**     px       double     parallax (arcsec)
**     rv       double     radial velocity (km/sec, +ve if receding)
**     ep0      double     start epoch in years (e.g Julian epoch)
**     ep1      double     end epoch in years (same system as ep0)
**
**  Returned:
**     *r1,*d1  double     RA,Dec at epoch ep1 (rad)
**
**  Notes:
**
**  1  The proper motions in RA are dRA/dt rather than cos(Dec)*dRA/dt,
**     and are in the same coordinate system as R0,D0.
**
**  2  If the available proper motions are pre-FK5 they will be per
**     tropical year rather than per Julian year, and so the epochs
**     must both be Besselian rather than Julian.  In such cases, a
**     scaling factor of 365.2422D0/365.25D0 should be applied to the
**     radial velocity before use.
**
**  Called:  slaDcs2c, slaDcc2s, slaDranrm
**
**  Defined in slamac.h:  DAS2R
**
**  Last revision:   3 August 2004
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
/* Km/s to AU/year multiplied by arc seconds to radians */
   static double vfr = ( 365.25 * 86400.0 / 149597870.0 ) * DAS2R;

   int i;
   double w, em[3], t, p[3];


/* Spherical to Cartesian */
   slaDcs2c ( r0, d0, p );

/* Space motion (radians per year) */
   w = vfr * rv * px;
   em[0] = - pr * p[1] - pd * cos ( r0 ) * sin ( d0 ) + w * p[0];
   em[1] =   pr * p[0] - pd * sin ( r0 ) * sin ( d0 ) + w * p[1];
   em[2] =               pd * cos ( d0 )              + w * p[2];

/* Apply the motion */
   t = ep1 - ep0;
   for ( i = 0; i < 3; i++ ) {
      p[i] += t*em[i];
   }

/* Cartesian to spherical */
   slaDcc2s ( p, r1, d1 );
   *r1 = slaDranrm ( *r1 );
}
