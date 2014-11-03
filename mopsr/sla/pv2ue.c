#include "slalib.h"
#include "slamac.h"
void slaPv2ue ( double pv[], double date, double pmass,
                double u[], int *jstat )
/*
**  - - - - - - - - -
**   s l a P v 2 u e
**  - - - - - - - - -
**
**  Construct a universal element set based on an instantaneous position
**  and velocity.
**
**  Given:
**     pv      double[6]  heliocentric x,y,z,xdot,ydot,zdot of date,
**                        (au,au/s; Note 1)
**     date    double     date (TT Modified Julian Date = JD-2400000.5)
**     pmass   double     mass of the planet (Sun=1; Note 2)
**
**  Returned:
**
**     u       double[13] universal orbital elements (Note 3)
**
**                    [0] combined mass (M+m)
**                    [1] total energy of the orbit (alpha)
**                    [2] reference (osculating) epoch (t0)
**                  [3-5] position at reference epoch (r0)
**                  [6-8] velocity at reference epoch (v0)
**                    [9] heliocentric distance at reference epoch
**                   [10] r0.v0
**                   [11] date (t)
**                   [12] universal eccentric anomaly (psi) of date
**
**     jstat   int*       status:  0 = OK
**                                -1 = illegal pmass
**                                -2 = too close to Sun
**                                -3 = too slow
**
**  Notes
**
**  1  The pv 6-vector can be with respect to any chosen inertial frame,
**     and the resulting universal-element set will be with respect to
**     the same frame.  A common choice will be mean equator and ecliptic
**     of epoch J2000.
**
**  2  The mass, pmass, is important only for the larger planets.  For
**     most purposes (e.g. asteroids) use 0.0.  Values less than zero
**     are illegal.
**
**  3  The "universal" elements are those which define the orbit for the
**     purposes of the method of universal variables (see reference).
**     They consist of the combined mass of the two bodies, an epoch,
**     and the position and velocity vectors (arbitrary reference frame)
**     at that epoch.  The parameter set used here includes also various
**     quantities that can, in fact, be derived from the other
**     information.  This approach is taken to avoiding unnecessary
**     computation and loss of accuracy.  The supplementary quantities
**     are (i) alpha, which is proportional to the total energy of the
**     orbit, (ii) the heliocentric distance at epoch, (iii) the
**     outwards component of the velocity at the given epoch, (iv) an
**     estimate of psi, the "universal eccentric anomaly" at a given
**     date and (v) that date.
**
**  Reference:  Everhart, E. & Pitkin, E.T., Am.J.Phys. 51, 712, 1983.
**
**  Last revision:   17 March 1999
**
**  Copyright P.T.Wallace.  All rights reserved.
*/

/* Gaussian gravitational constant (exact) */
#define GCON 0.01720209895

/* Canonical days to seconds */
#define CD2S ( GCON / 86400.0 );

/* Minimum allowed distance (AU) and speed (AU per canonical day) */
#define RMIN 1e-3
#define VMIN 1e-3

{
   double t0, cm, x, y, z, xd, yd, zd, r, v2, v, alpha, rdv;


/* Reference epoch. */
   t0 = date;

/* Combined mass (mu=M+m). */
   if ( pmass < 0.0 ) {
      *jstat = -1;
      return;
   }
   cm = 1.0 + pmass;

/* Unpack the state vector, expressing velocity in AU per canonical day. */
   x = pv[0];
   y = pv[1];
   z = pv[2];
   xd = pv[3] / CD2S;
   yd = pv[4] / CD2S;
   zd = pv[5] / CD2S;

/* Heliocentric distance, and speed. */
   r = sqrt ( x * x + y * y + z * z );
   v2 = xd * xd + yd * yd + zd * zd;
   v = sqrt ( v2 );

/* Reject unreasonably small values. */
   if ( r < RMIN ) {
      *jstat = -2;
      return;
   }
   if ( v < VMIN ) {
      *jstat = -3;
      return;
   }

/* Total energy of the orbit. */
   alpha = v2 - 2.0 * cm / r;

/* Outward component of velocity. */
   rdv = x * xd + y * yd + z * zd;

/* Construct the universal-element set. */
   u[0] = cm;
   u[1] = alpha;
   u[2] = t0;
   u[3] = x;
   u[4] = y;
   u[5] = z;
   u[6] = xd;
   u[7] = yd;
   u[8] = zd;
   u[9 ] = r;
   u[10] = rdv;
   u[11] = t0;
   u[12] = 0.0;

/* Exit. */
   *jstat = 0;

}
