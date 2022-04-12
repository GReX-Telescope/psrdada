#
# LIB_RDMACM([ACTION-IF-FOUND [,ACTION-IF-NOT-FOUND]])
#
# This m4 macro checks availablity of the RDMA CM library
#
# RDMACM_CFLAGS - autoconfig variable with flags required for compiling
# RDMACM_LIBS   - autoconfig variable with flags required for linking
# HAVE_RDMADM   - automake conditional
# HAVE_RDMADM   - pre-processor macro in config.h
#
# This macro tries to get RDMACM CFLAGS and LIBS using the
# pkg-config program.  If that is not available, it
# will try to link using:
#
# ----------------------------------------------------------

AC_DEFUN([LIB_RDMACM],
[
  AC_PROVIDE([LIB_RDMACM])

  AC_ARG_WITH([rdmacm-dir],
              AC_HELP_STRING([--with-rdmacm-dir=DIR],
                             [RDMACM is installed in DIR]))

  AC_MSG_CHECKING([for RDMACM Library (Infiniband) installation])

  RDMACM_CFLAGS=""
  RDMACM_LIBS="-lrdmacm -libverbs"

  ac_save_CFLAGS="$CFLAGS"
  ac_save_LIBS="$LIBS"

  LIBS="$ac_save_LIBS $RDMACM_LIBS"
  CFLAGS="$ac_save_CFLAGS $RDMACM_CFLAGS"

  AC_TRY_LINK([#include <rdma/rdma_cma.h>],[rdma_create_event_channel();],
              have_rdmacm=yes, have_rdmacm=no)

  AC_MSG_RESULT($have_rdmacm)

  LIBS="$ac_save_LIBS"
  CFLAGS="$ac_save_CFLAGS"

  if test x"$have_rdmacm" = xyes; then
    AC_DEFINE([HAVE_RDMACM], [1], [Define to 1 if you have the RDMACM library])
    [$1]
  else
    AC_MSG_WARN([RDMACM dependent code will not be compiled.])
    RDMACM_CFLAGS=""
    RDMACM_LIBS=""
    [$2]
  fi

  AC_SUBST(RDMACM_CFLAGS)
  AC_SUBST(RDMACM_LIBS)
  AM_CONDITIONAL(HAVE_RDMACM, [test x"$have_rdmacm" = xyes])

])
