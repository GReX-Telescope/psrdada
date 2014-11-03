#include "slalib.h"
#include "slamac.h"
double rms ( double a, double b );
void slaSvd ( int m, int n, int mp, int np, double *a, double *w,
              double *v, double *work, int *jstat )
/*
**  - - - - - - -
**   s l a S v d
**  - - - - - - -
**
**  Singular value decomposition.
**
**  (double precision)
**
**  This routine expresses a given matrix a as the product of
**  three matrices u, w, v:
**
**     a = u x w x vt
**
**  where:
**
**     a   is any m (rows) x n (columns) matrix, where m >= n
**     u   is an m x n column-orthogonal matrix
**     w   is an n x n diagonal matrix with w(i,i) >= 0
**     vt  is the transpose of an n x n orthogonal matrix
**
**     Note that m and n, above, are the logical dimensions of the
**     matrices and vectors concerned, which can be located in
**     arrays of larger physical dimensions, given by mp and np.
**
**  Given:
**     m,n    int            numbers of rows and columns in matrix a
**     mp,np  int            physical dimensions of the array containing a
**     a      double[mp,np]  array containing m x n matrix a
**
**  Returned:
**     *a     double[mp,np]  array containing m x n column-orthogonal matrix u
**     *w     double[n]      n x n diagonal matrix w (diagonal elements only)
**     *v     double[np,np]  array containing n x n orthogonal matrix v
**     *work  double[n]      workspace
**     *jstat int            0 = OK
**                          -1 = the a array is the wrong shape
**                          >0 = 1 + index of w for which convergence failed.
**
**     (n.b. v contains matrix v, not the transpose of matrix v)
**
**  References:
**     The algorithm is an adaptation of the routine SVD in the EISPACK
**     library (Garbow et al 1977, Eispack guide extension, Springer
**     Verlag), which is a Fortran 66 implementation of the Algol
**     routine SVD of Wilkinson & Reinsch 1971 (Handbook for Automatic
**     Computation, vol 2, Ed Bauer et al, Springer Verlag).  For the
**     non-specialist, probably the clearest general account of the use
**     of SVD in least squares problems is given in Numerical Recipes
**     (Press et al 1986, Cambridge University Press).
**
**  From slamac.h:  TRUE, FALSE
**
**  Example call (note handling of "adjustable dimension" 2D arrays):
**
**    double a[MP][NP], w[NP], v[NP][NP], work[NP];
**    int m, n, j;
**     :
**    slaSvd ( m, n, MP, NP, (double *) a, w, (double *) v, work, &j );
**
**  Last revision:   19 August 2001
**
**  Copyright P.T.Wallace.  All rights reserved.
*/

/* Maximum number of iterations in QR phase */
#define ITMAX 30
{

   int i, k, l, j, k1, its, l1, i1, cancel;
   double g, scale, an, s, x, f, h, cn, c, y, z;

/* Check that the matrix is the right size and shape. */
   if ( m < n || m > mp || n > np ) {
      *jstat = -1;
   } else {
      *jstat = 0;

   /* Householder reduction to bidiagonal form. */
      g = 0.0;
      scale = 0.0;
      an = 0.0;
      for ( i = 0; i < n; i++ ) {
         l = i + 1;
         work[i] = scale * g;
         g = 0.0;
         s = 0.0;
         scale = 0.0;
         if ( i < m ) {
            for ( k = i; k < m; k++ ) {
               scale += fabs ( a[k*np+i] );
            }
            if ( scale != 0.0 ) {
               for ( k = i; k < m; k++ ) {
                  x = a[k*np+i] / scale;
                  a[k*np+i] = x;
                  s += x * x;
               }
               f = a[i*np+i];
               g = - dsign ( sqrt ( s ), f );
               h = f * g - s;
               a[i*np+i] = f - g;
               if ( i != n-1 ) {
                  for ( j = l; j < n; j++ ) {
                     s = 0.0;
                     for ( k = i; k < m; k++ ) {
                        s += a[k*np+i] * a[k*np+j];
                     }
                     f = s / h;
                     for ( k = i; k < m; k++ ) {
                        a[k*np+j] += f * a[k*np+i];
                     }
                  }
               }
               for ( k = i; k < m; k++ ) {
                  a[k*np+i] *= scale;
               }
            }
         }
         w[i] = scale * g;
         g = 0.0;
         s = 0.0;
         scale = 0.0;
         if ( i < m && i != n-1 ) {
            for ( k = l;  k < n;  k++ ) {
               scale += fabs ( a[i*np+k] );
            }
            if ( scale != 0.0 ) {
               for ( k = l; k < n; k++ ) {
                  x = a[i*np+k] / scale;
                  a[i*np+k] = x;
                  s += x * x;
               }
               f = a[i*np+l];
               g = - dsign ( sqrt ( s ), f );
               h = f * g - s;
               a[i*np+l] = f - g;
               for ( k = l; k < n; k++ ) {
                  work[k] = a[i*np+k] / h;
               }
               if ( i != m-1 ) {
                  for ( j = l; j < m; j++ ) {
                     s = 0.0;
                     for ( k = l; k < n; k++ ) {
                        s += a[j*np+k] * a[i*np+k];
                     }
                     for ( k = l; k < n; k++ ) {
                        a[j*np+k] += s * work[k];
                     }
                  }
               }
               for ( k = l; k < n; k++ ) {
                  a[i*np+k] *= scale;
               }
            }
         }

      /* Overestimate of largest column norm for convergence test. */
         cn = fabs ( w[i] ) + fabs ( work[i] );
         an = gmax ( an, cn );
      }

   /* Accumulation of right-hand transformations. */
      for ( i = n-1; i >= 0; i-- ) {
         if ( i != n-1 ) {
            if ( g != 0.0 ) {
               for ( j = l; j < n; j++ ) {
                  v[j*np+i] = ( a[i*np+j] / a[i*np+l] ) / g;
               }
               for ( j = l; j < n; j++ ) {
                  s = 0.0;
                  for ( k = l; k < n; k++ ) {
                     s += a[i*np+k] * v[k*np+j];
                  }
                  for ( k = l; k < n; k++ ) {
                     v[k*np+j] += s * v[k*np+i];
                  }
               }
            }
            for ( j = l; j < n; j++ ) {
               v[i*np+j] = 0.0;
               v[j*np+i] = 0.0;
            }
         }
         v[i*np+i] = 1.0;
         g = work[i];
         l = i;
      }

   /* Accumulation of left-hand transformations. */
      for ( i = n-1; i >= 0; i-- ) {
         l = i + 1;
         g = w[i];
         if ( i != n-1 ) {
            for ( j = l; j < n; j++ ) {
               a[i*np+j] = 0.0;
            }
         }
         if ( g != 0.0 ) {
            if ( i != n-1 ) {
               for ( j = l; j < n; j++ ) {
                  s = 0.0;
                  for ( k = l; k < m; k++ ) {
                     s += a[k*np+i] * a[k*np+j];
                  }
                  f = ( s / a[i*np+i] ) / g;
                  for ( k = i; k < m; k++ ) {
                     a[k*np+j] += f * a[k*np+i];
                  }
               }
            }
            for ( j = i; j < m; j++ ) {
               a[j*np+i] /= g;
            }
         } else {
            for ( j = i; j < m; j++ ) {
               a[j*np+i] = 0.0;
            }
         }
         a[i*np+i] += 1.0;
      }

   /* Diagonalization of the bidiagonal form. */
      for ( k = n-1; k >= 0; k-- ) {
         k1 = k - 1;

      /* Iterate until converged. */
         for ( its = 1; its <= ITMAX; its++ ) {

         /* Test for splitting into submatrices. */
            cancel = TRUE;
            for ( l = k; l >= 0; l-- ) {
               l1 = l - 1;
               if ( an + fabs ( work[l] ) == an ) {
                  cancel = FALSE;
                  break;
               }
            /* (Following never attempted for l=0 because work[0] is zero.) */
               if ( an + fabs ( w[l1] ) == an ) break;
            }

         /* Cancellation of work[l] if l>0. */
            if ( cancel ) {
               s = 1.0;
               for ( i = l; i <= k; i++ ) {
                  f = s * work[i];
                  if ( an + fabs ( f ) == an ) break;
                  g = w[i];
                  h = rms ( f, g );
                  w[i] = h;
                  c = g / h;
                  s = - f / h;
                  for ( j = 0; j < m; j++ ) {
                     y = a[j*np+l1];
                     z = a[j*np+i];
                     a[j*np+l1] = y * c + z * s;
                     a[j*np+i] = - y * s + z * c;
                  }
               }
            }

         /* Converged? */
            z = w[k];
            if ( l == k ) {

            /* Yes: ensure singular values non-negative. */
               if ( z < 0.0 ) {
                  w[k] = -z;
                  for ( j = 0; j < n; j++ ) {
                     v[j*np+k] *= -1.0;
                  }
               }

            /* Stop iterating. */
               break;

            } else {

            /* Not converged yet: set status if iteration limit reached. */
               if ( its >= ITMAX ) {
                  *jstat = k + 1;
               }

            /* Shift from bottom 2 x 2 minor. */
               x = w[l];
               y = w[k1];
               g = work[k1];
               h = work[k];
               f = ( ( y - z ) * ( y + z )
                   + ( g - h ) * ( g + h ) ) / ( 2.0 * h * y );
               g = ( fabs ( f ) <= 1e15 ) ? rms ( f, 1.0 ) : fabs ( f );
               f = ( ( x - z ) * ( x + z )
                       + h * ( y / ( f + dsign ( g, f ) ) - h ) ) / x;

            /* Next QR transformation. */
               c = 1.0;
               s = 1.0;
               for ( i1 = l; i1 <= k1; i1++ ) {
                  i = i1 + 1;
                  g = work[i];
                  y = w[i];
                  h = s * g;
                  g = c * g;
                  z = rms ( f, h );
                  work[i1] = z;
                  if ( z != 0.0 ) {
                     c = f / z;
                     s = h / z;
                  } else {
                     c = 1.0;
                     s = 0.0;
                  }
                  f = x * c + g * s;
                  g = - x * s + g * c;
                  h = y * s;
                  y = y * c;
                  for ( j = 0; j < n; j++ ) {
                     x = v[j*np+i1];
                     z = v[j*np+i];
                     v[j*np+i1] = x * c + z * s;
                     v[j*np+i]  = - x * s + z * c;
                  }
                  z = rms ( f, h );
                  w[i1] = z;
                  if ( z != 0.0 ) {
                     c = f / z;
                     s = h / z;
                  }
                  f = c * g + s * y;
                  x = - s * g + c * y;
                  for ( j = 0; j < m; j++ ) {
                     y = a[j*np+i1];
                     z = a[j*np+i];
                     a[j*np+i1] = y * c + z * s;
                     a[j*np+i] = - y * s + z * c;
                  }
               }
               work[l] = 0.0;
               work[k] = f;
               w[k] = x;
            }
         }
      }
   }
}

double rms ( double a, double b )

/* sqrt(a*a+b*b) with protection against under/overflow. */

{
   double wa, wb, w;

   wa = fabs ( a );
   wb = fabs ( b );

   if ( wa > wb ) {
      w = wa;
      wa = wb;
      wb = w;
   }

   if ( wb == 0.0 ) {
      return 0.0;
   } else {
      w = wa / wb;
      return ( wb * sqrt ( 1.0 + w * w ) );
   }
}
