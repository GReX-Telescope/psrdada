#include "slalib.h"
#include "slamac.h"
void slaUe2pv ( double date, double u[], double pv[], int *jstat )
/*
**  - - - - - - - - -
**   s l a U e 2 p v
**  - - - - - - - - -
**
**  Heliocentric position and velocity of a planet, asteroid or comet,
**  starting from orbital elements in the "universal variables" form.
**
**  Given:
**     date    double     date, Modified Julian Date (JD-2400000.5)
**
**  Given and returned:
**     u       double[13] universal orbital elements (updated; Note 1)
**
**         given      [0] combined mass (M+m)
**           "        [1] total energy of the orbit (alpha)
**           "        [2] reference (osculating) epoch (t0)
**           "      [3-5] position at reference epoch (r0)
**           "      [6-8] velocity at reference epoch (v0)
**           "        [9] heliocentric distance at reference epoch
**           "       [10] r0.v0
**       returned    [11] date (t)
**           "       [12] universal eccentric anomaly (psi) of date, approx
**
**  Returned:
**     pv      double[6]  position (AU) and velocity (AU/s)
**     jstat   int*       status:  0 = OK
**                                -1 = radius vector zero
**                                -2 = failed to converge
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
**  2  The companion routine is slaEl2ue.  This takes the conventional
**     orbital elements and transforms them into the set of numbers
**     needed by the present routine.  A single prediction requires one
**     one call to slaEl2ue followed by one call to the present routine;
**     for convenience, the two calls are packaged as the routine
**     slaPlanel.   Multiple predictions may be made by again calling
**     slaEl2ue once, but then calling the present routine multiple times,
**     which is faster than multiple calls to slaPlanel.
**
**     It is not obligatory to use slaEl2ue to obtain the parameters.
**     However, it should be noted that because slaEl2ue performs its
**     own validation, no checks on the contents of the array U are made
**     by the present routine.
**
**  3  date is the instant for which the prediction is required.  It is
**     in the TT timescale (formerly Ephemeris Time, ET) and is a
**     Modified Julian Date (JD-2400000.5).
**
**  4  The universal elements supplied in the array u are in canonical
**     units (solar masses, AU and canonical days).  The position and
**     velocity are not sensitive to the choice of reference frame.  The
**     slaEl2ue routine in fact produces coordinates with respect to the
**     J2000 equator and equinox.
**
**  5  The algorithm was originally adapted from the EPHSLA program of
**     D.H.P.Jones (private communication, 1996).  The method is based
**     on Stumpff's Universal Variables.
**
**  Reference:  Everhart, E. & Pitkin, E.T., Am.J.Phys. 51, 712, 1983.
**
**  Last revision:   19 March 1999
**
**  Copyright P.T.Wallace.  All rights reserved.
*/

/* Gaussian gravitational constant (exact) */
#define GCON 0.01720209895

/* Canonical days to seconds */
#define CD2S ( GCON / 86400.0 )

/* Test value for solution and maximum number of iterations */
#define TEST 1e-13
#define NITMAX 20

{
   int i, nit, n;
   double cm, alpha, t0, p0[3], v0[3], r0, sigma0, t, psi, dt, w,
          tol, psj, psj2, beta, s0, s1, s2, s3, ff, r, f, g, fd, gd;



/* Unpack the parameters. */
   cm = u[0];
   alpha = u[1];
   t0 = u[2];
   for ( i = 0; i < 3; i++ ) {
      p0[i] = u[i+3];
      v0[i] = u[i+6];
   }
   r0 = u[9];
   sigma0 = u[10];
   t = u[11];
   psi = u[12];

/* Approximately update the universal eccentric anomaly. */
   psi = psi + ( date - t ) * GCON / r0;

/* Time from reference epoch to date (in Canonical Days: a canonical */
/* day is 58.1324409... days, defined as 1/GCON).                    */
   dt = ( date - t0 ) * GCON;

/* Refine the universal eccentric anomaly. */
   nit = 1;
   w = 1.0;
   tol = 0.0;
   while ( fabs ( w ) >= tol ) {

   /* Form half angles until BETA small enough. */
      n = 0;
      psj = psi;
      psj2 = psj * psj;
      beta = alpha * psj2;
      while ( fabs ( beta ) > 0.7 ) {
         n++;
         beta /= 4.0;
         psj /= 2.0;
         psj2 /= 4.0;
      }

   /* Calculate Universal Variables S0,S1,S2,S3 by nested series. */
      s3 = psj * psj2 * ( ( ( ( ( ( beta / 210.0 + 1.0 )
                                  * beta / 156.0 + 1.0 )
                                  * beta / 110.0 + 1.0 )
                                  * beta / 72.0 + 1.0 )
                                  * beta / 42.0 + 1.0 )
                                  * beta / 20.0 + 1.0 ) / 6.0;
      s2 = psj2 * ( ( ( ( ( ( beta / 182.0 + 1.0 )
                            * beta / 132.0 + 1.0 )
                            * beta / 90.0 + 1.0 )
                            * beta / 56.0 + 1.0 )
                            * beta / 30.0 + 1.0 )
                            * beta / 12.0 + 1.0 ) / 2.0;
      s1 = psj + alpha * s3;
      s0 = 1.0 + alpha * s2;

   /* Undo the angle-halving. */
      tol = TEST;
      while ( n > 0 ) {
         s3 = 2.0 * ( s0 * s3 + psj * s2 );
         s2 = 2.0 * s1 * s1;
         s1 = 2.0 * s0 * s1;
         s0 = 2.0 * s0 * s0 - 1.0;
         psj += psj;
         tol += tol;
         n--;
      }

   /* Improve the approximation to psi. */
      ff = r0 * s1 + sigma0 * s2 + cm * s3 - dt;
      r = r0 * s0 + sigma0 * s1 + cm * s2;
      if ( r == 0.0 ) {
         *jstat = -1;
         return;
      }
      w = ff / r;
      psi -= w;

   /* Next iteration, unless too many already. */
      if ( nit >= NITMAX ) {
         *jstat = -2;
         return;
      }
      nit++;
   }

/* Project the position and velocity vectors (scaling velocity to AU/s). */
   w = cm * s2;
   f = 1.0 - w / r0;
   g = dt - cm * s3;
   fd = - cm * s1 / ( r0 * r );
   gd = 1.0 - w / r;
   for ( i = 0; i < 3; i++ ) {
      pv[i] = p0[i] * f + v0[i] * g;
      pv[i+3] = CD2S * ( p0[i] * fd + v0[i] * gd );
   }

/* Update the parameters to allow speedy prediction of psi next time. */
   u[11] = date;
   u[12] = psi;

/* OK exit. */
   *jstat = 0;

}
