
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

/*! function julday() returns a high resolution MJD */

/* Hans van Someren Greve's Fortran Routines - converted to C*/
/* Fortran codes, converted to C, using f2c utility, and code is cleaned */
/* JULDA -- Calculate Julian day moment from gregorian time and vice versa */
/*   Calculate the Julian day moment from the civil year YEAR, */
/*   the day of the year DAY, and the universal time UT in */
/*   fraction of the day. */
/*   Calculate the civil year YEAR, the day of the year DAY, */
/*   and the universal time UT in fraction of the day from the */
/*   Julian day moment. */
/*   The routine is correct after 1 januari 1901 (Because the year 1900 */
/*   is not a leap year), and will be correct until the year 2400. */
/*   Parameters are ; */
/*       D     - Direction in I2. */
/*               Positive : YEAR,DAY,UT to JD. */
/*               Negative : JD to YEAR,DAY,UT. */
/*       YEAR  - Civil year in I2. I.e. 1989. */
/*       DAY   - UT day of the year in I2. 1, 2, ....,365, OR 366. */
/*       UT    - UT time in day fraction in R8. 0 <= UT < 1. */
/*       JD    - Julian day moment in R8. */

#define _DEBUG 1

int julda(int d, int *year, int *day, double *ut, double *jd)
{
/* Julian Day on 1900 0 Jan 12.00 UT */
  int jd1900 = 2415020;
  double j;
  int ld;
  int nd;
  int yy;

  if (d >= 0) {
    yy = *year - 1900; /* Nr of years since 1900 */
    nd = yy * 365; /* Nr of days since 1900 */
    nd += (yy - 1) / 4; /* Nr of leap days to add */
    nd += *day; /* Add day of the year */
    nd += jd1900; /* Make integer Julian Day */
    *jd = (double) nd + *ut - 0.5; /* Add time minus 0.5 */
  } else {
    j = *jd + .5; /* Add correction */
    nd = (int) j; /* Make integer days */
    *ut = j - (double) nd; /* Make UT time in fraction */
    nd -= jd1900; /* Nr of days since 1900 */
    ld = (int) (nd / 1461); /* Nr of leap days */
    yy = (int) ((ld << 2) + 1900); /* year to start leap ye */
    *day = (int) (nd - ld * 1461 + 1); /* days since start leap */
    *year = yy;
    if (*day == 366) return 0;

    if (*day > 366) { /* Last day of the leap */
      yy = (int) (yy + 1);/* Not a leap year, increment year */
      *day = (int) (*day - 366);/* Subtract days in leap */

    }
    *year = (int) (yy + (*day - 1) / 365);/* Find correct year */
    *day = (int) ((*day - 1) % 365 + 1);/* Find day */
  }

  return 0;

} /* julda */


/* CIVIT -- Conversion between time in dayfraction and time in HH,MM,SS. */
/*   Transform time T in day fraction into HH-MM-SS.SSS. */
/*   Transform time in HH-MM-SS.SSS into time T in day fraction. */
/*   Parameters are ; */
/*       D   - Direction in I2. */
/*             Positive : T to HH-MM-SS.SSS. */
/*             Negative : HH-MM-SS.SSS to T. */


double civit(int d, double *t, int *hh, int *mm, double *ss)
{
  double x;

  if (d >= 0) {
    x = *t * 24.0;
    *hh = (int) x;
    x = (x - (double) (*hh)) * 60.0;
    *mm = (int) x;
    *ss = ((x - (double) (*mm)) * 60.0);
    if (*ss > 59.999) {
      *ss = 0.0;
      *mm = *mm + 1;
      if (*mm == 60) {
        *mm = 0;
        *hh = *hh + 1;
        if (*hh == 24) {/* NOT ALLOWED */
          *hh = 23; /* MAKE 23H 59M 59.999S */
          *mm = 59;
          *ss = (double)59.999;
        }
      }
    }
  } else {
    *t = ((*ss/60.0 + (double)(*mm))/60.0 + (double)(*hh)) / 24.0;
  }

  return 0;

} /* civit */

long double julday(time_t seconds,long *intmjd, long double *fracmjd)
{
   long double dayfrac, jd, sec;
   int year, yday;
   int hour, min;
   struct tm *ptr;
  
   unsigned int nd;

   ptr = gmtime(&seconds);

   hour = ptr->tm_hour;
   min = ptr->tm_min;
   sec = (long double)ptr->tm_sec;

   year = ptr->tm_year;
   yday = ptr->tm_yday + 1;

   dayfrac = ( (sec/60.0L + (long double) min)/60.0L + \
	  (long double)hour)/24.0L; 
   nd = year * 365;
   nd += (year - 1)/4;
   nd += yday + 2415020;
   
   *intmjd = nd - 2400001;
   *fracmjd = dayfrac;

   jd = (long double)nd + dayfrac - 0.5L;

   return jd;
}

