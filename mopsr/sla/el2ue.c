#include "slalib.h"
#include "slamac.h"
void slaEl2ue ( double date, int jform, double epoch, double orbinc,
                double anode, double perih, double aorq, double e,
                double aorl, double dm, double u[], int *jstat )
/*
**  - - - - - - - - -
**   s l a E l 2 u e
**  - - - - - - - - -
**
**  Transform conventional osculating orbital elements into "universal"
**  form.
**
**  Given:
**     date    double     epoch (TT MJD) of osculation (Note 3)
**     jform   int        choice of element set (1-3, Note 6)
**     epoch   double     epoch (TT MJD) of the elements
**     orbinc  double     inclination (radians)
**     anode   double     longitude of the ascending node (radians)
**     perih   double     longitude or argument of perihelion (radians)
**     aorq    double     mean distance or perihelion distance (AU)
**     e       double     eccentricity
**     aorl    double     mean anomaly or longitude (radians, jform=1,2 only)
**     dm      double     daily motion (radians, jform=1 only)
**
**  Returned:
**     u       double[13] universal orbital elements (Note 1)
**
**                    [0] combined mass (M+m)
**                    [1] total energy of the orbit (alpha)
**                    [2] reference (osculating) epoch (t0)
**                  [3-5] position at reference epoch (r0)
**                  [6-8] velocity at reference epoch (v0)
**                    [9] heliocentric distance at reference epoch
**                   [10] r0.v0
**                   [11] date (t)
**                   [12] universal eccentric anomaly (psi) of date, approx
**
**     jstat   int*       status:  0 = OK
**                                -1 = illegal jform
**                                -2 = illegal e
**                                -3 = illegal aorq
**                                -4 = illegal dm
**                                -5 = numerical error
**
**  Called:  slaUe2pv, slaPv2ue
**
**  Notes
**
**  1  The "universal" elements are those which define the orbit for the
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
**  2  The companion routine is slaUe2pv.  This takes the set of numbers
**     that the present routine outputs and uses them to derive the
**     object's position and velocity.  A single prediction requires one
**     call to the present routine followed by one call to slaUe2pv;
**     for convenience, the two calls are packaged as the routine
**     slaPlanel.  Multiple predictions may be made by again calling the
**     present routine once, but then calling slaUe2pv multiple times,
**     which is faster than multiple calls to slaPlanel.
**
**  3  date is the epoch of osculation.  It is in the TT timescale
**     (formerly Ephemeris Time, ET) and is a Modified Julian Date
**     (JD-2400000.5).
**
**  4  The supplied orbital elements are with respect to the J2000
**     ecliptic and equinox.  The position and velocity parameters
**     returned in the array u are with respect to the mean equator and
**     equinox of epoch J2000, and are for the perihelion prior to the
**     specified epoch.
**
**  5  The universal elements returned in the array u are in canonical
**     units (solar masses, AU and canonical days).
**
**  6  Three different element-format options are available:
**
**     Option jform=1, suitable for the major planets:
**
**     epoch  = epoch of elements (TT MJD)
**     orbinc = inclination i (radians)
**     anode  = longitude of the ascending node, big omega (radians)
**     perih  = longitude of perihelion, curly pi (radians)
**     aorq   = mean distance, a (AU)
**     e      = eccentricity, e (range 0 to <1)
**     aorl   = mean longitude L (radians)
**     dm     = daily motion (radians)
**
**     Option jform=2, suitable for minor planets:
**
**     epoch  = epoch of elements (TT MJD)
**     orbinc = inclination i (radians)
**     anode  = longitude of the ascending node, big omega (radians)
**     perih  = argument of perihelion, little omega (radians)
**     aorq   = mean distance, a (AU)
**     e      = eccentricity, e (range 0 to <1)
**     aorl   = mean anomaly M (radians)
**
**     Option jform=3, suitable for comets:
**
**     epoch  = epoch of perihelion (TT MJD)
**     orbinc = inclination i (radians)
**     anode  = longitude of the ascending node, big omega (radians)
**     perih  = argument of perihelion, little omega (radians)
**     aorq   = perihelion distance, q (AU)
**     e      = eccentricity, e (range 0 to 10)
**
**  7  Unused elements (dm for jform=2, aorl and dm for jform=3) are
**     not accessed.
**
**  8  The algorithm was originally adapted from the EPHSLA program of
**     D.H.P.Jones (private communication, 1996).  The method is based on
**     Stumpff's Universal Variables.
**
**  Reference:  Everhart, E. & Pitkin, E.T., Am.J.Phys. 51, 712, 1983.
**
**  Last revision:   31 December 2002
**
**  Copyright P.T.Wallace.  All rights reserved.
*/

/* Gaussian gravitational constant (exact) */
#define GCON 0.01720209895

/* Sin and cos of J2000 mean obliquity (IAU 1976) */
#define SE 0.3977771559319137
#define CE 0.9174820620691818

{
   int j;
   double pht, argph, q, w, cm, alpha, phs, sw, cw, si, ci, so, co,
          x, y, z, px, py, pz, vx, vy, vz, dt, fc, fp, psi, ul[13], pv[6];



/* Validate arguments. */
   if ( jform < 1 || jform > 3 ) {
      *jstat = -1;
      return;
   }
   if ( e < 0.0 || e > 10.0 || ( e >= 1.0 && jform != 3 ) ) {
      *jstat = -2;
      return;
   }
   if ( aorq <= 0.0 ) {
      *jstat = -3;
      return;
   }
   if ( jform == 1 && dm <= 0.0 ) {
      *jstat = -4;
      return;
   }

/*
** Transform elements into standard form:
**
** pht   = epoch of perihelion passage
** argph = argument of perihelion (little omega)
** q     = perihelion distance (q)
** cm    = combined mass, M+m (mu)
*/

   switch ( jform ) {

/* Major planet. */
   case 1:
      pht = epoch - ( aorl - perih ) / dm;
      argph = perih - anode;
      q = aorq * ( 1.0 - e );
      w = dm / GCON;
      cm =  w * w * aorq * aorq * aorq;
      break;

/* Minor planet. */
   case 2:
      pht = epoch - aorl * sqrt ( aorq * aorq * aorq ) / GCON;
      argph = perih;
      q = aorq * ( 1.0 - e );
      cm = 1.0;
      break;

/* Comet. */
   default:
      pht = epoch;
      argph = perih;
      q = aorq;
      cm = 1.0;
   }

/*
** The universal variable alpha.  This is proportional to the total
** energy of the orbit:  -ve for an ellipse, zero for a parabola,
** +ve for a hyperbola.
*/

   alpha = cm * ( e - 1.0 ) / q;

/* Speed at perihelion. */

   phs = sqrt ( alpha + 2.0 * cm / q );

/*
** In a Cartesian coordinate system which has the x-axis pointing
** to perihelion and the z-axis normal to the orbit (such that the
** object orbits counter-clockwise as seen from +ve z), the
** perihelion position and velocity vectors are:
**
**   position   [Q,0,0]
**   velocity   [0,phs,0]
**
** To express the results in J2000 equatorial coordinates we make a
** series of four rotations of the Cartesian axes:
**
**           axis      Euler angle
**
**     1      z        argument of perihelion (little omega)
**     2      x        inclination (i)
**     3      z        longitude of the ascending node (big omega)
**     4      x        J2000 obliquity (epsilon)
**
** In each case the rotation is clockwise as seen from the +ve end
** of the axis concerned.
*/

/* Functions of the Euler angles. */
   sw = sin ( argph );
   cw = cos ( argph );
   si = sin ( orbinc );
   ci = cos ( orbinc );
   so = sin ( anode );
   co = cos ( anode );

/* Position at perihelion (AU). */
   x = q * cw;
   y = q * sw;
   z = y * si;
   y = y * ci;
   px = x * co - y * so;
   y = x * so + y * co;
   py = y * CE - z * SE;
   pz = y * SE + z * CE;

/* Velocity at perihelion (AU per canonical day). */
   x = - phs * sw;
   y = phs * cw;
   z = y * si;
   y = y * ci;
   vx = x * co - y * so;
   y = x * so + y * co;
   vy = y * CE - z * SE;
   vz = y * SE + z * CE;

/* Time from perihelion to date (in Canonical Days: a canonical */
/* day is 58.1324409... days, defined as 1/GCON).               */

   dt = ( date - pht ) * GCON;

/* First Approximation to the Universal Eccentric Anomaly, psi, */
/* based on the circle (fc) and parabola (fp) values.           */
   fc = dt / q;
   w = pow ( 3.0 * dt + sqrt ( 9.0 * dt * dt + 8.0 * q * q * q ),
             1.0 / 3.0 );
   fp = w - 2.0 * q / w;
   psi = ( 1.0 - e ) * fc + e * fp;

/* Assemble local copy of element set. */
   ul[0] = cm;
   ul[1] = alpha;
   ul[2] = pht;
   ul[3] = px;
   ul[4] = py;
   ul[5] = pz;
   ul[6] = vx;
   ul[7] = vy;
   ul[8] = vz;
   ul[9] = q;
   ul[10] = 0.0;
   ul[11] = date;
   ul[12] = psi;

/* Predict position+velocity at epoch of osculation. */
   slaUe2pv ( date, ul, pv, &j );
   if ( j ) {
      *jstat = -5;
      return;
   }

/* Convert back to universal elements. */
   slaPv2ue ( pv, date, cm - 1.0, u, &j );
   if ( j ) {
      *jstat = -5;
      return;
   }

/* OK exit. */
   *jstat = 0;

}
