#include "slalib.h"
#include "slamac.h"
void slaCombn  ( int nsel, int ncand, int list[], int* j )
/*
**  - - - - - - - - -
**   s l a C o m b n
**  - - - - - - - - -
**
**  Generate the next combination, a subset of a specified size chosen
**  from a specified number of items.
**
**  Given:
**     nsel    int        number of items (subset size)
**     ncand   int        number of candidates (set size)
**
**  Given and Returned:
**     list    int[nsel]  latest combination, list[0]=0 to initialize
**
**  Returned:
**     *j      int        status: -1 = illegal nsel or ncand
**                                 0 = OK
**                                +1 = no more combinations available
**
**  Notes:
**
**  1) nsel and ncand must both be at least 1, and nsel must be less
**     than or equal to ncand.
**
**  2) This routine returns, in the list array, a subset of nsel integers
**     chosen from the range 1 to ncand inclusive, in ascending order.
**     Before calling the routine for the first time, the caller must set
**     the first element of the list array to zero (any value less than 1
**     will do) to cause initialization.
**
**  2) The first combination to be generated is:
**
**        list[0]=1, list[1]=2, ..., list[nsel-1]=nsel
**
**     This is also the combination returned for the "finished" (j=1)
**     case.
**
**     The final permutation to be generated is:
**
**        list[0]=ncand, list[1]=ncand-1, ..., list[nsel-1]=ncand-nsel+1
**
**  3) If the "finished" (j=1) status is ignored, the routine
**     continues to deliver combinations, the pattern repeating
**     every ncand!/(nsel!*(ncand-nsel)!) calls.
**
**  4) The algorithm is by R.F.Warren-Smith (private communication).
**
**  Last revision:   25 August 1999
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
   int i, more, nmax, m;


/* Validate, and set status. */
   if ( nsel < 1 || ncand < 1 || nsel > ncand ) {
      *j = -1;
      return;
   } else {
      *j = 0;
   }

/* Just starting? */
   if ( list[0] < 1 ) {

   /* Yes: return 1,2,3... */
      for ( i = 0; i < nsel; i++ ) {
         list[i] = i+1;
      }

   } else {

   /* No: find the first selection that we can increment. */

   /* Start with the first list item. */
      i = 0;

   /* Loop. */
      more = 1;
      while ( more ) {

      /* Is this the final list item? */
         if ( i == nsel-1 ) {

         /* Yes:  comparison value is number of candidates plus one. */
            nmax = ncand+1;

         } else {

         /* No:  comparison value is next list item. */
            nmax = list[i+1];
         }

      /* Can the current item be incremented? */
         if ( nmax - list[i] > 1 ) {

         /* Yes:  increment it. */
            list[i]++;

         /* Reinitialize the preceding items. */
            for ( m = 0; m < i; m++ ) {
               list[m] = m+1;
            }

         /* Quit the loop. */
            more = 0;

         } else {

         /* Can't increment the current item:  is it the final one? */
            if ( i == nsel-1 ) {

            /* Yes:  set the status. */
               *j = 1;

            /* Restart the sequence. */
               for ( i = 0; i < nsel; i++ ) {
                  list[i] = i+1;
               }

            /* Quit the loop. */
               more = 0;

            } else {

            /* No:  next list item. */
               i++;
            }
         }
      }
   }
}
