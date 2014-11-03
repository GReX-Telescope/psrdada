#include "slalib.h"
#include "slamac.h"
void slaPertue ( double date, double u[], int *jstat )
/*
**  - - - - - - - - - -
**   s l a P e r t u e
**  - - - - - - - - - -
**
**  Update the universal elements of an asteroid or comet by applying
**  planetary perturbations.
**
**  Given:
**     date    double     final epoch (TT MJD) for the updated elements
**
**  Given and returned:
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
**  Returned:
**     jstat   int*       status:
**                          +102 = warning, distant epoch
**                          +101 = warning, large timespan ( > 100 years)
**                     +1 to +10 = coincident with major planet (Note 5)
**                             0 = OK
**                            -1 = numerical error
**
**  Called:  slaEpj, slaPlanet, slaPv2ue, slaUe2pv, slaEvp, slaPrec,
**           slaDmoon, slaDmxv
**
**  Notes:
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
**  2  The universal elements are with respect to the J2000 equator and
**     equinox.
**
**  3  The epochs date, u[2] and u[11] are all Modified Julian Dates
**     (JD-2400000.5).
**
**  4  The algorithm is a simplified form of Encke's method.  It takes as
**     a basis the unperturbed motion of the body, and numerically
**     integrates the perturbing accelerations from the major planets.
**     The expression used is essentially Sterne's 6.7-2 (reference 1).
**     Everhart and Pitkin (reference 2) suggest rectifying the orbit at
**     each integration step by propagating the new perturbed position
**     and velocity as the new universal variables.  In the present
**     routine the orbit is rectified less frequently than this, in order
**     to gain a slight speed advantage.  However, the rectification is
**     done directly in terms of position and velocity, as suggested by
**     Everhart and Pitkin, bypassing the use of conventional orbital
**     elements.
**
**     The f(q) part of the full Encke method is not used.  The purpose
**     of this part is to avoid subtracting two nearly equal quantities
**     when calculating the "indirect member", which takes account of the
**     small change in the Sun's attraction due to the slightly displaced
**     position of the perturbed body.  A simpler, direct calculation in
**     double precision proves to be faster and not significantly less
**     accurate.
**
**     Apart from employing a variable timestep, and occasionally
**     "rectifying the orbit" to keep the indirect member small, the
**     integration is done in a fairly straightforward way.  The
**     acceleration estimated for the middle of the timestep is assumed
**     to apply throughout that timestep;  it is also used in the
**     extrapolation of the perturbations to the middle of the next
**     timestep, to predict the new disturbed position.  There is no
**     iteration within a timestep.
**
**     Measures are taken to reach a compromise between execution time
**     and accuracy.  The starting-point is the goal of achieving
**     arcsecond accuracy for ordinary minor planets over a ten-year
**     timespan.  This goal dictates how large the timesteps can be,
**     which in turn dictates how frequently the unperturbed motion has
**     to be recalculated from the osculating elements.
**
**     Within predetermined limits, the timestep for the numerical
**     integration is varied in length in inverse proportion to the
**     magnitude of the net acceleration on the body from the major
**     planets.
**
**     The numerical integration requires estimates of the major-planet
**     motions.  Approximate positions for the major planets (Pluto
**     alone is omitted) are obtained from the routine slaPlanet.  Two
**     levels of interpolation are used, to enhance speed without
**     significantly degrading accuracy.  At a low frequency, the routine
**     slaPlanet is called to generate updated position+velocity "state
**     vectors".  The only task remaining to be carried out at the full
**     frequency (i.e. at each integration step) is to use the state
**     vectors to extrapolate the planetary positions.  In place of a
**     strictly linear extrapolation, some allowance is made for the
**     curvature of the orbit by scaling back the radius vector as the
**     linear extrapolation goes off at a tangent.
**
**     Various other approximations are made.  For example, perturbations
**     by Pluto and the minor planets are neglected and relativistic
**     effects are not taken into account.
**
**     In the interests of simplicity, the background calculations for
**     the major planets are carried out en masse.  The mean elements and
**     state vectors for all the planets are refreshed at the same time,
**     without regard for orbit curvature, mass or proximity.
**
**     The Earth-Moon system is treated as a single body when the body is
**     distant but as separate bodies when closer to the EMB than the
**     parameter RNE, which incurs a time penalty but improves accuracy
**     for near-Earth objects.
**
**  5  This routine is not intended to be used for major planets.
**     However, if major-planet elements are supplied, sensible results
**     will, in fact, be produced.  This happens because the routine
**     checks the separation between the body and each of the planets and
**     interprets a suspiciously small value (0.001 AU) as an attempt to
**     apply the routine to the planet concerned.  If this condition is
**     detected, the contribution from that planet is ignored, and the
**     status is set to the planet number (1-10 = Mercury, Venus, EMB,
**     Mars, Jupiter, Saturn, Uranus, Neptune, Earth, Moon) as a warning.
**
**  References:
**
**     1  Sterne, Theodore E., "An Introduction to Celestial Mechanics",
**        Interscience Publishers Inc., 1960.  Section 6.7, p199.
**
**     2  Everhart, E. & Pitkin, E.T., Am.J.Phys. 51, 712, 1983.
**
**  Last revision:   27 December 2004
**
**  Copyright P.T.Wallace.  All rights reserved.
*/

/* Distance from EMB at which Earth and Moon are treated separately */
#define RNE 1.0

/* Coincidence with major planet distance */
#define COINC 0.0001

/* Coefficient relating timestep to perturbing force */
#define TSC 1e-4

/* Minimum and maximum timestep (days) */
#define TSMIN 0.01
#define TSMAX 10.0

/* Age limit for major-planet state vector (days) */
#define AGEPMO 5.0

/* Age limit for major-planet mean elements (days) */
#define AGEPEL 50.0

/* Margin for error when deciding whether to renew the planetary data */
#define TINY 1e-6

/* Age limit for the body's osculating elements (before rectification) */
#define AGEBEL 100.0

/* Gaussian gravitational constant (exact) and square */
#define GCON 0.01720209895
#define GCON2 (GCON*GCON)

{

/* The final epoch */
   double tfinal;

/* The body's current universal elements */
   double ul[13];

/* Current reference epoch */
   double t0;

/* Timespan from latest orbit rectification to final epoch (days) */
   double tspan;

/* Time left to go before integration is complete */
   double tleft;

/* Time direction flag: +1=forwards, -1=backwards */
   double fb;

/* First-time flag */
   int first;

/* The current perturbations */
   double rtn,      /* Epoch (days relative to current reference epoch) */
          perp[3],  /* Position (AU) */
          perv[3],  /* Velocity (AU/d) */
          pera[3];  /* Acceleration (AU/d/d) */

/* Length of current timestep (days), and half that */
   double ts, hts;

/* Epoch of middle of timestep */
   double t;

/* Epoch of planetary mean elements */
   double tpel;

/* Planet number (1=Mercury, 2=Venus, 3=EMB...8=Neptune) */
   int np;

/* Planetary universal orbital elements */
   double up[8][13];

/* Epoch of planetary state vectors */
   double tpmo;

/* State vectors for the major planets (AU,AU/s) */
   double pvin[8][6];

/* Earth velocity and position vectors (AU,AU/s) */
   double vb[3], pb[3], vh[3], pe[3];

/* Moon geocentric state vector (AU,AU/s) and position part */
   double pvm[6], pm[3];

/* Date to J2000 de-precession matrix */
   double pmat[3][3];

/* Correction terms for extrapolated major planet vectors */
   double r2x3[8], /* Sun-to-planet distances squared multiplied by 3 */
          gc[8],   /* Sunward acceleration terms, G/2R^3 */
          fc,      /* Tangential-to-circular correction factor */
          fg;      /* Radial correction factor due to Sunwards acceleration */

/* The body's unperturbed and perturbed state vectors (AU,AU/s) */
   double pv0[6], pv[6];

/* The body's perturbed and unperturbed heliocentric distances (AU) cubed */
   double r03, r3;

/* The perturbating accelerations, indirect and direct */
   double fi[3], fd[3];

/* Sun-to-planet vector, and distance cubed */
   double rho[3], rho3;

/* Body-to-planet vector, and distance cubed */
   double delta[3], delta3;

/* Miscellaneous */
   int i, j, npm1, ne;
   double r2, w, dt, dt2, r, ft;

/* Planetary inverse masses, Mercury thru Neptune then Earth & Moon */
   static double amas[] = {
      6023600.0,
       408523.5,
       328900.5,
      3098710.0,
         1047.355,
         3498.5,
        22869.0,
        19314.0,
       332946.038,
     27068709.0
   };



/* Preset the status to OK. */
   *jstat = 0;

/* Copy the final epoch. */
   tfinal = date;

/* Copy the elements (which will be periodically updated). */
   for ( i = 0; i < 13; i++ ) {
      ul[i] = u[i];
   }

/* Initialize the working reference epoch. */
   t0 = ul[2];

/* Total timespan (days) and hence time left. */
   tspan = tfinal - t0;
   tleft = tspan;

/* Warn if excessive. */
   if ( fabs ( tspan ) > 36525.0 ) *jstat = 101;

/* Time direction: +1 for forwards, -1 for backwards. */
   fb = dsign ( 1.0, tspan );

/* Initialize relative epoch for start of current timestep. */
   rtn = 0.0;

/* Reset the perturbations (position, velocity, acceleration). */
   for ( i = 0; i < 3; i++ ) {
      perp[i] = 0.0;
      perv[i] = 0.0;
      pera[i] = 0.0;
   }

/* Set "first iteration" flag. */
   first = TRUE;

/* Step through the time left. */
   while ( fb * tleft > 0.0 ) {

   /* Magnitude of current acceleration due to planetary attractions. */
      if ( first ) {
         ts = TSMIN;
      } else {
         r2 = 0.0;
         for ( i = 0; i < 3; i++ ) {
            w = fd[i];
            r2 += w * w;
         }
         w = sqrt ( r2 );

      /* Use the acceleration to decide how big a timestep can be tolerated. */
         if ( w != 0.0 ) {
            ts = TSC / w;
            if ( ts > TSMAX ) {
               ts = TSMAX;
            } else if ( ts < TSMIN ) {
               ts = TSMIN;
            }
         } else {
            ts = TSMAX;
         }
      }
      ts *= fb;

   /* Override if final epoch is imminent. */
      tleft = tspan - rtn;
      if ( fabs ( ts ) > fabs ( tleft ) ) ts = tleft;

   /* Epoch of middle of timestep. */
      hts = ts / 2.0;
      t = t0 + rtn + hts;

   /* Is it time to recompute the major-planet elements? */
      if ( first || ( fabs ( t - tpel ) - AGEPEL ) >= TINY ) {

      /* Yes: go forward in time by just under the maximum allowed. */
         tpel = t + fb * AGEPEL;

      /* Compute the state vector for the new epoch. */
         for ( np = 1; np <= 8; np++ ) {
            npm1 = np - 1;

            slaPlanet ( tpel, np, pv, &j );

         /* Warning if remote epoch, abort if error. */
            if ( j == 1 ) {
               *jstat = 102;
            } else if ( j ) {
               *jstat = -1;
               return;
            }

         /* Transform the vector into universal elements. */
            slaPv2ue ( pv, tpel, 0.0, up[npm1], &j );
            if ( j ) {
               *jstat = -1;
               return;
            }
         }
      }

   /* Is it time to recompute the major-planet motions? */
      if ( first || ( fabs ( t - tpmo ) - AGEPMO ) >= TINY ) {

      /* Yes: look ahead. */
         tpmo = t + fb * AGEPMO;

      /* Compute the motions of each planet (AU,AU/d). */
         for ( np = 1; np <= 8; np++ ) {
            npm1 = np - 1;

         /* The planet's position and velocity (AU,AU/s). */
            slaUe2pv ( tpmo, up[npm1], pvin[npm1], &j );
            if ( j ) {
               *jstat = -1;
               return;
            }

         /* Scale velocity to AU/d. */
            for ( j = 3; j < 6; j++ ) {
               pvin[npm1][j] *= 86400.0;
            }

         /* Precompute also the extrapolation correction terms. */
            r2 = 0.0;
            for ( i = 0; i < 3; i++ ) {
               w = pvin[npm1][i];
               r2 += w * w;
            }
            r2x3[npm1] = r2 * 3.0;
            gc[npm1] = GCON2 / ( 2.0 * r2 * sqrt ( r2 ) );
         }
      }

   /* Reset the first-time flag. */
      first = FALSE;

   /* Unperturbed motion of the body at middle of timestep (AU,AU/s). */
      slaUe2pv ( t, ul, pv0, &j );
      if ( j ) {
         *jstat = -1;
         return;
      }

   /* Perturbed position of the body (AU) and heliocentric distance cubed. */
      r2 = 0.0;
      for ( i = 0; i < 3; i++ ) {
         w = pv0[i] + perp[i] + ( perv[i] + pera[i] * hts / 2.0 ) * hts;
         pv[i] = w;
         r2 += w * w;
      }
      r3 = r2 * sqrt ( r2 );

   /* The body's unperturbed heliocentric distance cubed. */
      r2 = 0.0;
      for ( i = 0; i < 3; i++ ) {
         w = pv0[i];
         r2 += w * w;
      }
      r03 = r2 * sqrt ( r2 );

   /* Compute indirect and initialize direct parts of the perturbation. */
      for ( i = 0; i < 3; i++ ) {
         fi[i] = pv0[i] / r03 - pv[i] / r3;
         fd[i] = 0.0;
      }

   /* Ready to compute the direct planetary effects. */

   /* Reset the "near-Earth" flag. */
      ne = FALSE;

   /* Interval from state-vector epoch to middle of current timestep. */
      dt = t - tpmo;
      dt2 = dt * dt;

   /* Planet by planet. */
      for ( np = 1; np <= 10; np++ ) {
         npm1 = np - 1;

      /* Which perturbing body? */
         if ( np <= 8 ) {

         /* Planet: compute the extrapolation in longitude (squared). */
            r2 = 0.0;
            for ( j = 3; j < 6; j++ ) {
               w = pvin[npm1][j] * dt;
               r2 += w * w;
            }

         /* Hence the tangential-to-circular correction factor. */
            fc = 1.0 + r2 / r2x3[npm1];

         /* The radial correction factor due to the inwards acceleration. */
            fg = 1.0 - gc[npm1] * dt2;

         /* Planet's position. */
            for ( i = 0; i < 3; i++ ) {
               rho[i] = fg * ( pvin[npm1][i] + fc * pvin[npm1][i+3] * dt );
            }

         } else if ( ne ) {

         /* Near-Earth and either Earth or Moon. */

            if ( np == 9 ) {

            /* Earth: position. */
               slaEvp ( t, 2000.0, vb, pb, vh, pe );
               for ( i = 0; i < 3; i++ ) {
                  rho[i] = pe[i];
               }

            } else {

            /* Moon: position. */
               slaPrec ( slaEpj ( t ), 2000.0, pmat );
               slaDmoon ( t, pvm );
               slaDmxv ( pmat, pvm, pm );
               for ( i = 0; i < 3; i++ ) {
                  rho[i] = pm[i] + pe[i];
               }
            }
         }

      /* Proceed unless Earth or Moon and not the near-Earth case. */
         if ( np <= 8 || ne ) {

         /* Heliocentric distance cubed. */
            r2 = 0.0;
            for ( i = 0; i < 3; i++ ) {
               w = rho[i];
               r2 += w * w;
            }
            r = sqrt ( r2 );

            rho3 = r2 * r;

         /* Body-to-planet vector, and distance cubed. */
            r2 = 0.0;
            for ( i = 0; i < 3; i++ ) {
               w = rho[i] - pv[i];
               delta[i] = w;
               r2 += w * w;
            }
            r = sqrt ( r2 );

         /* If this is the EMB, set the near-Earth flag appropriately. */
            if ( np == 3 && r < RNE ) ne = TRUE;

         /* Proceed unless EMB and this is the near-Earth case. */
            if ( ! ( ne && ( np == 3 ) ) ) {

            /* If too close, ignore this planet and set a warning. */
               if ( r < COINC ) {
                  *jstat = np;

               } else {

               /* Accumulate "direct" part of perturbation acceleration. */
                  delta3 = r2 * r;
                  w = amas[npm1];
                  for ( i = 0; i < 3; i++ ) {
                     fd[i] += ( delta[i] / delta3 - rho[i] / rho3 ) / w;
                  }
               }
            }
         }
      }

   /* Update the perturbations to the end of the timestep. */
      rtn = rtn + ts;
      for ( i = 0; i < 3; i++ ) {
         w = ( fi[i] + fd[i] ) * GCON2;
         ft = w * ts;
         perp[i] += ( perv[i] + ft / 2.0 ) * ts;
         perv[i] += ft;
         pera[i] = w;
      }

   /* Time still to go. */
      tleft = tspan - rtn;

   /* Is it either time to rectify the orbit or the last time through? */
      if ( fabs ( rtn ) >= AGEBEL || ( fb * tleft ) <= 0.0 ) {

      /* Yes: update to the end of the current timestep. */
         t0 += rtn;
         rtn = 0.0;

      /* The body's unperturbed motion (AU,AU/s). */
         slaUe2pv ( t0, ul, pv0, &j );
         if ( j ) {
            *jstat = -1;
            return;
         }

      /* Add and re-initialize the perturbations. */
         for ( i = 0; i < 3; i++ ) {
            j = i + 3;
            pv[i] = pv0[i] + perp[i];
            pv[j] = pv0[j] + perv[i] / 86400.0;
            perp[i] = 0.0;
            perv[i] = 0.0;
            pera[i] = fd[i] * GCON2;
         }

      /* Use the position and velocity to set up new universal elements. */
         slaPv2ue ( pv, t0, 0.0, ul, &j );
         if ( j ) {
            *jstat = -1;
            return;
         }

      /* Adjust the timespan and time left. */
         tspan = tfinal - t0;
         tleft = tspan;
      }

   /* Next timestep. */
   }

/* Return the updated universal-element set. */
   for ( i = 0; i < 13; i++ ) {
      u[i] = ul[i];
   }

}
