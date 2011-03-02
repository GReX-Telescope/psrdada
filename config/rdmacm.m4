dnl @synopsis SWIN_LIB_RDMACM
dnl 
AC_DEFUN([SWIN_LIB_RDMACM],
[
  AC_PROVIDE([SWIN_LIB_RDMACM])

  AC_REQUIRE([SWIN_PACKAGE_OPTIONS])
  SWIN_PACKAGE_OPTIONS([rdmacm])

  AC_MSG_CHECKING([for RDMACM Library (Infiniband) installation])

  if test x"$RDMACM" == x; then
    RDMACM=rdmacm
  fi

  if test "$have_rdmacm" != "user disabled"; then

    SWIN_PACKAGE_FIND([rdmacm],[rdma/rdma_cma.h])
    SWIN_PACKAGE_TRY_COMPILE([rdmacm],[#include <rdma/rdma_cma.h>])

    SWIN_PACKAGE_FIND([rdmacm],[lib$RDMACM.*])
    SWIN_PACKAGE_TRY_LINK([rdmacm],[#include <rdma/rdma_cma.h>],
                          [ rdma_create_event_channel();],
                          [-l$RDMACM])

  fi

  AC_MSG_RESULT([$have_rdmacm])

  if test x"$have_rdmacm" = xyes; then

    AC_DEFINE([HAVE_RDMACM],[1],
              [Define if the RDMACM Library is present])
    [$1]

  else
    :
    [$2]
  fi

  RDMACM_LIBS="$rdmacm_LIBS"
  RDMACM_CFLAGS="$rdmacm_CFLAGS"

  AC_SUBST(RDMACM_LIBS)
  AC_SUBST(RDMACM_CFLAGS)
  AM_CONDITIONAL(HAVE_RDMACM,[test "$have_rdmacm" = yes])

])

