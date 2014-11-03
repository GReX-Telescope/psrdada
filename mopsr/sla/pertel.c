#include "slalib.h"
#include "slamac.h"
void slaPertel (int jform, double date0, double date1,
                double epoch0, double orbi0, double anode0,
                double perih0, double aorq0, double e0, double am0,
                double *epoch1, double *orbi1, double *anode1,
                double *perih1, double *aorq1, double *e1, double *am1,
                int *jstat )
/*
**  - - - - - - - - - -
**   s l a P e r t e l
**  - - - - - - - - - -
**
**  Update the osculating orbital elements of an asteroid or comet by
**  applying planetary perturbations.
**
**  Given (format and dates):
**     jform   int      choice of element set (2 or 3; Note 1)
**     date0   double   date of osculation (TT MJD) for the given elements
**     date1   double   date of osculation (TT MJD) for the updated elements
**
**  Given (the unperturbed elements):
**     epoch0  double   epoch (TT MJD) of the given element set (Note 2)
**     orbi0   double   inclination (radians)
**     anode0  double   longitude of the ascending node (radians)
**     perih0  double   argument of perihelion (radians)
**     aorq0   double   mean distance or perihelion distance (AU)
**     e0      double   eccentricity
**     am0     double   mean anomaly (radians, jform=2 only)
**
**  Returned (the updated elements):
**     epoch1  double*  epoch (TT MJD) of the updated element set (Note 2)
**     orbi1   double*  inclination (radians)
**     anode1  double*  longitude of the ascending node (radians)
**     perih1  double*  argument of perihelion (radians)
**     aorq1   double*  mean distance or perihelion distance (AU)
**     e1      double*  eccentricity
**     am1     double*  mean anomaly (radians, jform=2 only)
**
**  Returned (status flag):
**     jstat   int*     status: +102 = warning, distant epoch
**                              +101 = warning, large timespan ( > 100 years)
**                         +1 to +10 = coincident with planet (Note 6)
**                                 0 = OK
**                                -1 = illegal jform
**                                -2 = illegal e0
**                                -3 = illegal aorq0
**                                -4 = internal error
**                                -5 = numerical error
**
**  Notes:
**
**  1  Two different element-format options are available:
**
**     Option jform=2, suitable for minor planets:
**
**     epoch   = epoch of elements (TT MJD)
**     orbi    = inclination i (radians)
**     anode   = longitude of the ascending node, big omega (radians)
**     perih   = argument of perihelion, little omega (radians)
**     aorq    = mean distance, a (AU)
**     e       = eccentricity, e
**     am      = mean anomaly M (radians)
**
**     Option jform=3, suitable for comets:
**
**     epoch   = epoch of perihelion (TT MJD)
**     orbi    = inclination i (radians)
**     anode   = longitude of the ascending node, big omega (radians)
**     perih   = argument of perihelion, little omega (radians)
**     aorq    = perihelion distance, q (AU)
**     e       = eccentricity, e
**
**  2  date0, date1, epoch0 and epoch1 are all instants of time in
**     the TT timescale (formerly Ephemeris Time, ET), expressed
**     as Modified Julian Dates (JD-2400000.5).
**
**     date0 is the instant at which the given (i.e. unperturbed)
**     osculating elements are correct.
**
**     date1 is the specified instant at which the updated osculating
**     elements are correct.
**
**     epoch0 and epoch1 will be the same as date0 and date1
**     (respectively) for the jform=2 case, normally used for minor
**     planets.  For the jform=3 case, the two epochs will refer to
**     perihelion passage and so will not, in general, be the same as
**     date0 and/or date1 though they may be similar to one another.
**
**  3  The elements are with respect to the J2000 ecliptic and equinox.
**
**  4  Unused elements (am0 and am1 for jform=3) are not accessed.
**
**  5  See the slaPertue routine for details of the algorithm used.
**
**  6  This routine is not intended to be used for major planets, which
**     is why jform=1 is not available and why there is no opportunity
**     to specify either the longitude of perihelion or the daily
**     motion.  However, if jform=2 elements are somehow obtained for a
**     major planet and supplied to the routine, sensible results will,
**     in fact, be produced.  This happens because the slaPertue  routine
**     that is called to perform the calculations checks the separation
**     between the body and each of the planets and interprets a
**     suspiciously small value (1E-3 AU) as an attempt to apply it to
**     the planet concerned.  If this condition is detected, the
**     contribution from that planet is ignored, and the status is set to
**     the planet number (1-10 = Mercury, Venus, EMB, Mars, Jupiter,
**     Saturn, Uranus, Neptune, Earth, Moon) as a warning.
**
**  Reference:
**
**     Sterne, Theodore E., "An Introduction to Celestial Mechanics",
**     Interscience Publishers Inc., 1960.  Section 6.7, p199.
**
**  Called:  slaEl2ue, slaPertue, slaUe2el
**
**  Last revision:   19 June 2004
**
**  Copyright 2002 P.T.Wallace.  All rights reserved.
*/
{
   double u[13], dm;
   int j, jf;


/* Check that the elements are either minor-planet or comet format. */
   if ( jform < 2 || jform > 3 ) {
      *jstat = -1;
      return;
   } else {

   /* Provisionally set the status to OK. */
      *jstat = 0;
   }

/* Transform the elements from conventional to universal form. */
   slaEl2ue ( date0, jform, epoch0, orbi0, anode0, perih0,
              aorq0, e0, am0, 0.0, u, &j );
   if ( j ) {
      *jstat = j;
      return;
   }

/* Update the universal elements. */
   slaPertue ( date1, u, &j );
   if ( j > 0 ) {
      *jstat = j;
   } else if ( j < 0 ) {
      *jstat = -5;
      return;
   }

/* Transform from universal to conventional elements. */
   slaUe2el ( u, jform,
              &jf, epoch1, orbi1, anode1, perih1, aorq1, e1, am1, &dm, &j );
   if ( jf != jform || j ) *jstat = -5;

}
