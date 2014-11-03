#include "slalib.h"
#include "slamac.h"
void slaPv2el ( double pv[], double date, double pmass, int jformr,
                int *jform, double *epoch, double *orbinc,
                double *anode, double *perih, double *aorq, double *e,
                double *aorl, double *dm, int *jstat )
/*
**  - - - - - - - - -
**   s l a P v 2 e l
**  - - - - - - - - -
**
**  Heliocentric osculating elements obtained from instantaneous position
**  and velocity.
**
**  Given:
**     pv      double[6]  heliocentric x,y,z,xdot,ydot,zdot of date,
**                         J2000 equatorial triad (AU,AU/s; Note 1)
**     date    double     date (TT Modified Julian Date = JD-2400000.5)
**     pmass   double     mass of the planet (Sun=1; Note 2)
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
**                                -1 = illegal pmass
**                                -2 = illegal jformr
**                                -3 = position/velocity out of range
**
**  Notes
**
**  1  The pv 6-vector is with respect to the mean equator and equinox of
**     epoch J2000.  The orbital elements produced are with respect to
**     the J2000 ecliptic and mean equinox.
**
**  2  The mass, pmass, is important only for the larger planets.  For
**     most purposes (e.g. asteroids) use 0.0.  Values less than zero
**     are illegal.
**
**  3  Three different element-format options are supported:
**
**     Option jformr=1, suitable for the major planets:
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
**     Option jformr=2, suitable for minor planets:
**
**     epoch  = epoch of elements (TT MJD)
**     orbinc = inclination i (radians)
**     anode  = longitude of the ascending node, big omega (radians)
**     perih  = argument of perihelion, little omega (radians)
**     aorq   = mean distance, a (AU)
**     e      = eccentricity, e
**     aorl   = mean anomaly M (radians)
**
**     Option jformr=3, suitable for comets:
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
**  5  The arguments returned for each value of jform (cf Note 5: jform
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
**  7  The osculating epoch for the returned elements is the argument
**     "date".
**
**  Reference:  Sterne, Theodore E., "An Introduction to Celestial
**              Mechanics", Interscience Publishers, 1960
**
**  Called:  slaDranrm
**
**  Last revision:   31 December 2002
**
**  Copyright P.T.Wallace.  All rights reserved.
*/

/* Seconds to days */
#define DAY 86400.0

/* Gaussian gravitational constant (exact) */
#define GCON 0.01720209895

/* Sin and cos of J2000 mean obliquity (IAU 1976) */
#define SE 0.3977771559319137
#define CE 0.9174820620691818

/* Minimum allowed distance (AU) and speed (AU/day) */
#define RMIN 1e-3
#define VMIN 1e-8

/* How close to unity the eccentricity has to be to call it a parabola */
#define PARAB 1e-8

{
   double x, y, z, xd, yd, zd, r, v2, v, rdv, gmu, hx, hy, hz,
          hx2py2, h2, h, oi, bigom, ar, e2, ecc, s, c, at, u, om,
          gar3, em1, ep1, hat, shat, chat, ae, am, dn, pl,
          el, q, tp, that, thhf, f;
   int jf;


/* Validate arguments pmass and jformr. */
   if ( pmass < 0.0 ) {
      *jstat = -1;
      return;
   }
   if ( jformr < 1 || jformr > 3 ) {
      *jstat = -2;
      return;
   }

/* Provisionally assume the elements will be in the chosen form. */
   jf = jformr;

/* Rotate the position from equatorial to ecliptic coordinates. */
   x = pv [ 0 ];
   y = pv [ 1 ] * CE + pv [ 2 ] * SE;
   z = - pv [ 1 ] * SE + pv [ 2 ] * CE;

/* Rotate the velocity similarly, scaling to AU/day. */
   xd = DAY * pv [ 3 ];
   yd = DAY * ( pv [ 4 ] * CE + pv [ 5 ] * SE );
   zd = DAY * ( - pv [ 4 ] * SE + pv [ 5 ] * CE );

/* Distance and speed. */
   r = sqrt ( x * x + y * y + z * z );
   v2 = xd * xd + yd * yd + zd * zd;
   v = sqrt ( v2 );

/* Reject unreasonably small values. */
   if ( r < RMIN || v < VMIN ) {
      *jstat = -3;
      return;
   }

/* R dot V. */
   rdv = x * xd + y * yd + z * zd;

/* Mu. */
   gmu = ( 1.0 + pmass ) * GCON * GCON;

/* Vector angular momentum per unit reduced mass. */
   hx = y * zd - z * yd;
   hy = z * xd - x * zd;
   hz = x * yd - y * xd;

/* Areal constant. */
   hx2py2 = hx * hx + hy * hy;
   h2 = hx2py2 + hz * hz;
   h = sqrt ( h2 );

/* Inclination. */
   oi = atan2 ( sqrt ( hx2py2 ), hz );

/* Longitude of ascending node. */
   bigom = ( hx != 0.0 || hy != 0.0 ) ? atan2 ( hx, -hy ) : 0.0;

/* Reciprocal of mean distance etc. */
   ar = 2.0 / r - v2 / gmu;

/* Eccentricity. */
   e2 = 1.0 - ar * h2 / gmu;
   ecc = ( e2 >= 0.0 ) ? sqrt ( e2 ) : 0.0;

/* True anomaly. */
   s = h * rdv;
   c = h2 - r * gmu;
   at = ( s != 0.0 || c != 0.0 ) ? atan2 ( s, c ) : 0.0;

/* Argument of the latitude. */
   s = sin ( bigom );
   c = cos ( bigom );
   u = atan2 ( ( - x * s + y * c ) * cos ( oi ) + z * sin ( oi ),
               x * c + y * s );

/* Argument of perihelion. */
   om = u - at;

/* Capture near-parabolic cases. */
   if ( fabs ( ecc - 1.0 ) < PARAB ) ecc = 1.0;

/* Comply with jformr = 1 or 2 only if orbit is elliptical. */
   if ( ecc >= 1.0 ) jf = 3;

/* Functions. */
   gar3 = gmu * ar * ar * ar;
   em1 = ecc - 1.0;
   ep1 = ecc + 1.0;
   hat = at / 2.0;
   shat = sin ( hat );
   chat = cos ( hat );

/* Ellipse? */
   if ( ecc < 1.0  ) {

   /* Eccentric anomaly. */
      ae = 2.0 * atan2 ( sqrt ( -em1 ) * shat, sqrt ( ep1 ) * chat );

   /* Mean anomaly. */
      am = ae - ecc * sin ( ae );

   /* Daily motion. */
      dn = sqrt ( gar3 );
   }

/* "Major planet" element set? */
   if ( jf == 1 ) {

   /* Longitude of perihelion. */
      pl = bigom + om;

   /* Longitude at epoch. */
      el = pl + am;
   }

/* "Comet" element set? */
   if ( jf == 3 ) {

   /* Perihelion distance. */
      q = h2 / ( gmu * ep1 );

   /* Ellipse, parabola, hyperbola? */
      if ( ecc < 1.0 ) {

      /* Ellipse: epoch of perihelion. */
         tp = date - am / dn;
      } else {

      /* Parabola or hyperbola: evaluate tan ( ( true anomaly ) / 2 ) */
         that = shat / chat;
         if ( ecc == 1.0 ) {

         /* Parabola: epoch of perihelion. */
            tp = date - that * ( 1.0 + that * that / 3.0 ) * h * h2 /
                               ( 2.0 * gmu * gmu );
         } else {

         /* Hyperbola: epoch of perihelion. */
            thhf = sqrt ( em1 / ep1 ) * that;
            f = log ( 1.0 + thhf ) - log ( 1.0 - thhf );
            tp = date - ( ecc * sinh ( f ) - f ) / sqrt ( - gar3 );
         }
      }
   }

/* Return the appropriate set of elements. */
   *jform = jf;
   *orbinc = oi;
   *anode = slaDranrm ( bigom );
   *e = ecc;
   if ( jf == 1 ) {
      *perih = slaDranrm ( pl );
      *aorl = slaDranrm ( el );
      *dm = dn;
   } else {
      *perih = slaDranrm ( om );
      if ( jf == 2 ) *aorl = slaDranrm ( am );
   }
   if ( jf != 3 ) {
      *epoch = date;
      *aorq = 1.0 / ar;
   } else {
      *epoch = tp;
      *aorq = q;
   }
   *jstat = 0;

}
