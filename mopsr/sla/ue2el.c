#include "slalib.h"
#include "slamac.h"
void slaUe2el ( double u[], int jformr,
                int *jform, double *epoch, double *orbinc,
                double *anode, double *perih, double *aorq, double *e,
                double *aorl, double *dm, int *jstat )
/*
**  - - - - - - - - -
**   s l a U e 2 e l
**  - - - - - - - - -
**
**  Transform universal elements into conventional heliocentric
**  osculating elements.
**
**  Given:
**
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
**     jformr  int        requested element set (1-3; Note 3)
**
**  Returned:
**     jform   double*    element set actually returned (1-3; Note 4)
**     epoch   double*    epoch of elements (TT MJD)
**     orbinc  double*    inclination (radians)
**     anode   double*    longitude of the ascending node (radians)
**     perih   double*    longitude or argument of perihelion (radians)
**     aorq    double*    mean distance or perihelion distance (AU)
**     e       double*    eccentricity
**     aorl    double*    mean anomaly or longitude (radians, jform=1,2 only)
**     dm      double*    daily motion (radians, jform=1 only)
**     jstat   int*       status:  0 = OK
**                                -1 = illegal combined mass
**                                -2 = illegal jformr
**                                -3 = position/velocity out of range
**
**  Notes
**
**  1  The "universal" elements are those which define the orbit for the
**     purposes of the method of universal variables (see reference 2).
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
**  2  The universal elements are with respect to the mean equator and
**     equinox of epoch J2000.  The orbital elements produced are with
**     respect to the J2000 ecliptic and mean equinox.
**
**  3  Three different element-format options are supported:
**
**     Option jform=1, suitable for the major planets:
**
**     epoch  = epoch of elements (TT MJD)
**     orbinc = inclination i (radians)
**     anode  = longitude of the ascending node, big omega (radians)
**     perih  = longitude of perihelion, curly pi (radians)
**     aorq   = mean distance, a (AU)
**     e      = eccentricity, e
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
**     e      = eccentricity, e
**     aorl   = mean anomaly M (radians)
**
**     Option jform=3, suitable for comets:
**
**     epoch  = epoch of perihelion (TT MJD)
**     orbinc = inclination i (radians)
**     anode  = longitude of the ascending node, big omega (radians)
**     perih  = argument of perihelion, little omega (radians)
**     aorq   = perihelion distance, q (AU)
**     e      = eccentricity, e
**
**  4  It may not be possible to generate elements in the form
**     requested through jformr.  The caller is notified of the form
**     of elements actually returned by means of the jform argument:
**
**      jformr   jform     meaning
**
**        1        1       OK - elements are in the requested format
**        1        2       never happens
**        1        3       orbit not elliptical
**
**        2        1       never happens
**        2        2       OK - elements are in the requested format
**        2        3       orbit not elliptical
**
**        3        1       never happens
**        3        2       never happens
**        3        3       OK - elements are in the requested format
**
**  5  The arguments returned for each value of jform (cf Note 6: jform
**     may not be the same as jformr) are as follows:
**
**         jform         1              2              3
**         epoch         t0             t0             T
**         orbinc        i              i              i
**         anode         Omega          Omega          Omega
**         perih         curly pi       omega          omega
**         aorq          a              a              q
**         e             e              e              e
**         aorl          L              M              -
**         dm            n              -              -
**
**     where:
**
**         t0           is the epoch of the elements (MJD, TT)
**         T              "    epoch of perihelion (MJD, TT)
**         i              "    inclination (radians)
**         Omega          "    longitude of the ascending node (radians)
**         curly pi       "    longitude of perihelion (radians)
**         omega          "    argument of perihelion (radians)
**         a              "    mean distance (AU)
**         q              "    perihelion distance (AU)
**         e              "    eccentricity
**         L              "    longitude (radians, 0-2pi)
**         M              "    mean anomaly (radians, 0-2pi)
**         n              "    daily motion (radians)
**         -             means no value is set
**
**  6  At very small inclinations, the longitude of the ascending node
**     anode becomes indeterminate and under some circumstances may be
**     set arbitrarily to zero.  Similarly, if the orbit is close to
**     circular, the true anomaly becomes indeterminate and under some
**     circumstances may be set arbitrarily to zero.  In such cases,
**     the other elements are automatically adjusted to compensate,
**     and so the elements remain a valid description of the orbit.
**
**  References:
**
**     1  Sterne, Theodore E., "An Introduction to Celestial Mechanics",
**        Interscience Publishers Inc., 1960.  Section 6.7, p199.
**
**     2  Everhart, E. & Pitkin, E.T., Am.J.Phys. 51, 712, 1983.
**
**  Called:  slaPv2el
**
**  Last revision:   18 March 1999
**
**  Copyright P.T.Wallace.  All rights reserved.
*/

/* Gaussian gravitational constant (exact) */
#define GCON 0.01720209895

/* Canonical days to seconds */
#define CD2S ( GCON / 86400.0 )

{
   int i;
   double pmass, date, pv[6];


/* Unpack the universal elements. */
   pmass = u[0] - 1.0;
   date = u[2];
   for ( i = 0; i < 3; i++ ) {
      pv[i] = u[i+3];
      pv[i+3] = u[i+6] * CD2S;
   }

/* Convert the position and velocity etc into conventional elements. */
   slaPv2el ( pv, date, pmass, jformr, jform, epoch, orbinc, anode,
              perih, aorq, e, aorl, dm, jstat );

}
