AC_DEFUN([FUNC_AFFINITY],
[
  AC_PROVIDE([FUNC_AFFINITY])

  AC_CHECK_HEADERS([sched.h])
  AC_CHECK_FUNCS([sched_setaffinity],have_affinity=yes,have_affinity=no)
  AM_CONDITIONAL(HAVE_AFFINITY,[test x"$have_affinity" = xyes])

])

