#include "slalib.h"
#include "slamac.h"
void slaPermut ( int n, int istate[], int iorder[], int* j )
/*
**  - - - - - - - - - -
**   s l a  P e r m u t
**  - - - - - - - - - -
**
**  Generate the next permutation of a specified number of items.
**
**  Given:
**     n        int      number of items:  there will be n! permutations
**
**  Given and Returned:
**     istate   int[n]   state, istate[0]=-1 to initialize
**
**  Returned:
**     istate   int[n]   state, updated ready for next time
**     iorder   int[n)   next permutation of numbers 1,2,...,n
**     *j       int      status: -1 = illegal n (zero or less is illegal)
**                                0 = OK
**                               +1 = no more permutations available
**
**  Notes:
**
**  1) This routine returns, in the iorder array, the integers 1 to n
**     inclusive, in an order that depends on the current contents of
**     the istate array.  Before calling the routine for the first
**     time, the caller must set the first element of the istate array
**     to -1 (any negative number will do) to cause the istate array
**     to be fully initialized.
**
**  2) The first permutation to be generated is:
**
**          iorder[0]=n, iorder[1]=n-1, ..., iorder[n-1]=1
**
**     This is also the permutation returned for the "finished"
**     (j=1) case.
**
**     The final permutation to be generated is:
**
**          iorder[0]=1, iorder[1]=2, ..., iorder[n-1]=n
**
**  3) If the "finished" (j=1) status is ignored, the routine continues
**     to deliver permutations, the pattern repeating every n! calls.
**
**  Last revision:   14 July 1999
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
   int i, ip1, islot, iskip;


/* ------------- */
/* Preliminaries */
/* ------------- */

/* Validate, and set status. */
   if ( n < 1 ) {
      *j = -1;
      return;
   } else {
      *j = 0;
   }

/* If just starting, initialize state array */
   if ( istate[0] < 0 ) {
      istate[0] = -1;
      for ( i = 1; i < n; i++ ) {
         istate[i] = 0;
      }
   }

/* -------------------------- */
/* Increment the state number */
/* -------------------------- */

/* The state number, maintained in the istate array, is a mixed-radix   */
/* number with n! states.  The least significant digit, with a radix of */
/* 1, is in istate[0].  The next digit, in istate[1], has a radix of 2, */
/* and so on.                                                           */

/* Increment the least-significant digit of the state number. */
   istate[0]++;

/* Digit by digit starting with the least significant. */
   for ( i = 0; i < n; i++ ) {
      ip1 = i + 1;

   /* Carry? */
      if ( istate[i] >= ip1 ) {

      /* Yes:  reset the current digit. */
         istate[i] = 0;

      /* Overflow? */
         if ( ip1 >= n ) {

         /* Yes:  there are no more permutations. */
            *j = 1;

         } else {

         /* No:  carry. */
            istate[ip1]++;
         }
      }
   }

/* ------------------------------------------------------------------- */
/* Translate the state number into the corresponding permutation order */
/* ------------------------------------------------------------------- */

/* Initialize the order array.  All but one element will be overwritten. */
   for ( i = 0; i < n; i++ ) {
      iorder[i] = 1;
   }

/* Look at each state number digit, starting with the most significant. */
   for ( i = n-1; i > 0; i-- ) {

   /* Initialize the position where the new number will go. */
      islot = -1;

   /* The state number digit says which unfilled slot is to be used. */
      for ( iskip = 0; iskip <= istate[i]; iskip++ ) {

      /* Increment the slot number until an unused slot is found. */
         islot++;
         while ( iorder[islot] > 1 ) {
            islot++;
         }
      }

   /* Store the number in the permutation order array. */
      iorder[islot] = i + 1;
   }
}
