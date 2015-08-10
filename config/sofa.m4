dnl @synopsis SWIN_LIB_SOFA
dnl 
AC_DEFUN([SWIN_LIB_SOFA],
[
  AC_PROVIDE([SWIN_LIB_SOFA])

  AC_REQUIRE([SWIN_PACKAGE_OPTIONS])
  SWIN_PACKAGE_OPTIONS([sofa])

  AC_MSG_CHECKING([for SOFA Library installation])

  if test x"$SOFA" == x; then
    SOFA=sofa
  fi

  if test "$have_sofa" != "user disabled"; then

    SWIN_PACKAGE_FIND([sofa],[sofa.h])
    SWIN_PACKAGE_TRY_COMPILE([sofa],[#include <sofa.h>;#include <math.h>],
                             [iauAnp(0.0);])

    SWIN_PACKAGE_FIND([sofa],[lib$SOFA.*])
    SWIN_PACKAGE_TRY_LINK([sofa],[#include <sofa.h>],
                          [ iauAnp(0.0);],
                          [-lsofa_c -lm])

  fi

  AC_MSG_RESULT([$have_sofa])

  if test x"$have_sofa" = xyes; then

    AC_DEFINE([HAVE_SOFA],[1],
              [Define if the SOFA Library is present])
    [$1]

  else
    :
    [$2]
  fi

  SOFA_LIBS="$sofa_LIBS"
  SOFA_CFLAGS="$sofa_CFLAGS"

  AC_SUBST(SOFA_LIBS)
  AC_SUBST(SOFA_CFLAGS)
  AM_CONDITIONAL(HAVE_SOFA,[test "$have_sofa" = yes])

])

