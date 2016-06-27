dnl @synopsis SWIN_LIB_XGPU
dnl 
AC_DEFUN([SWIN_LIB_XGPU],
[
  AC_PROVIDE([SWIN_LIB_XGPU])

  AC_REQUIRE([SWIN_PACKAGE_OPTIONS])
  SWIN_PACKAGE_OPTIONS([xgpu])

  AC_MSG_CHECKING([for XGPU Library])

  if test x"$XGPU" == x; then
    XGPU=xgpu
  fi

  if test "$have_xgpu" != "user disabled"; then

    SWIN_PACKAGE_FIND([xgpu],[xgpu.h])
    SWIN_PACKAGE_TRY_COMPILE([xgpu],[#include <xgpu.h>])

    SWIN_PACKAGE_FIND([xgpu],[lib$XGPU.*])
    SWIN_PACKAGE_TRY_LINK([xgpu],[#include <xgpu.h>],
                          [XGPUInfo xgpu_info;],
                          [-l$XGPU])

  fi

  AC_MSG_RESULT([$have_xgpu])

  if test x"$have_xgpu" = xyes; then

    AC_DEFINE([HAVE_XGPU],[1],
              [Define if the XGPU Library is present])
    [$1]

  else
    :
    [$2]
  fi

  XGPU_LIBS="$xgpu_LIBS"
  XGPU_CFLAGS="$xgpu_CFLAGS"

  AC_SUBST(XGPU_LIBS)
  AC_SUBST(XGPU_CFLAGS)
  AM_CONDITIONAL(HAVE_XGPU,[test "$have_xgpu" = yes])

])

