#include "slalib.h"
#include "slamac.h"
#include <limits.h>

void slaInt2in ( char *string, int *nstrt, int *ireslt, int *jflag )
/*
**  - - - - - - - - - -
**   s l a I n t 2 i n
**  - - - - - - - - - -
**
**  Convert free-format input into an integer.
**
**  Given:
**     string    char*    string containing number to be decoded
**     nstrt     int*     where to start decode (1st = 1)
**     ireslt    long*    current value of result
**
**  Returned:
**     nstrt     int*     advanced to next number
**     ireslt    int*     result
**     jflag     int*     status: -1 = -OK, 0 = +OK, 1 = null, 2 = error
**
**  Notes:
**
**     1     The reason slaInt2in has separate OK status values for +
**           and - is to enable minus zero to be detected.   This is
**           of crucial importance when decoding mixed-radix numbers.
**           For example, an angle expressed as deg, arcmin, arcsec
**           may have a leading minus sign but a zero degrees field.
**
**     2     A TAB is interpreted as a space.
**
**     3     The basic format is the sequence of fields #^, where
**           # is a sign character + or -, and ^ means a string of
**           decimal digits.
**
**     4     Spaces:
**
**             .  Leading spaces are ignored.
**
**             .  Spaces between the sign and the number are allowed.
**
**             .  Trailing spaces are ignored;  the first signifies
**                end of decoding and subsequent ones are skipped.
**
**     5     Delimiters:
**
**             .  Any character other than +,-,0-9 or space may be
**                used to signal the end of the number and terminate
**                decoding.
**
**             .  Comma is recognized by slaInt2in as a special case;
**                it is skipped, leaving the pointer on the next
**                character.  See 9, below.
**
**     6     The sign is optional.  The default is +.
**
**     7     A "null result" occurs when the string of characters being
**           decoded does not begin with +,- or 0-9, or consists
**           entirely of spaces.  When this condition is detected, jflag
**           is set to 1 and ireslt is left untouched.
**
**     8     nstrt = 1 for the first character in the string.
**
**     9     On return from slaInt2in, nstrt is set ready for the next
**           decode - following trailing blanks and any comma.  If a
**           delimiter other than comma is being used, nstrt must be
**           incremented before the next call to slaInt2in, otherwise
**           all subsequent calls will return a null result.
**
**     10    Errors (jflag=2) occur when:
**
**             .  there is a + or - but no number;  or
**
**             .  the number is larger than INT_MAX.
**
**     11    When an error has been detected, nstrt is left
**           pointing to the character following the last
**           one used before the error came to light.
**
**     12    See also slaIntin, slaFlotin and slaDfltin.
**
**  Last revision:   10 December 2002
**
**  Copyright P.T.Wallace.  All rights reserved.
*/

{
   long lreslt;


/* Decode a long integer. */
   lreslt = (long) *ireslt;
   slaIntin ( string, nstrt, &lreslt, jflag );

/* If OK, validate length. */
   if ( *jflag < 2 ) {
      if ( lreslt >= (long) INT_MIN &&
           lreslt <= (long) INT_MAX ) {

      /* OK: cast the result to int. */
         *ireslt = (int) lreslt;

      } else {

      /* Number outside int range: return error status. */
         *jflag = 2;
      }
   }
}
