#include "slalib.h"
#include "slamac.h"
void slaDimxv ( double dm[3][3], double va[3], double vb[3] )
/*
**  - - - - - - - - -
**   s l a D i m x v
**  - - - - - - - - -
**
**  Performs the 3-d backward unitary transformation:
**
**     vector vb = (inverse of matrix dm) * vector va
**
**  (double precision)
**
**  (n.b.  The matrix must be unitary, as this routine assumes that
**   the inverse and transpose are identical)
**
**
**  Given:
**     dm       double[3][3]   matrix
**     va       double[3]      vector
**
**  Returned:
**     vb       double[3]      result vector
**
**  Note:  va and vb may be the same array.
**
**  Last revision:   6 November 1999
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
  int i, j;
  double w, vw[3];

/* Inverse of matrix dm * vector va -> vector vw */
   for ( j = 0; j < 3; j++ ) {
      w = 0.0;
      for ( i = 0; i < 3; i++ ) {
         w += dm[i][j] * va[i];
      }
      vw[j] = w;
   }

/* Vector vw -> vector vb */
   for ( j = 0; j < 3; j++ ) {
     vb[j] = vw[j];
   }
}
