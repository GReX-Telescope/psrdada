#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
# If you change this file, you usually should run
# autoreconf -fvi
AC_PREREQ(2.61)

AC_INIT([DADA],[1.0],[straten@astron.nl])

# Check the README.md for more information on versioning the library
# values are current:revision:age
AC_SUBST([PSRDADA_SO_VERSION], [1:0:0])

AC_CONFIG_MACRO_DIR([config])
AC_CONFIG_SRCDIR([src/dada_def.h])
AC_PROG_LIBTOOL
AC_DISABLE_SHARED

# Automake initialization
AM_INIT_AUTOMAKE
AM_CONFIG_HEADER([config.h])

# Checks for programs.
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LIBTOOL

ACX_PTHREAD
LIBS="$PTHREAD_LIBS $LIBS"
CFLAGS="$CFLAGS $PTHREAD_CFLAGS"
CXXFLAGS="$CXXFLAGS $PTHREAD_CFLAGS"

LIB_CUDA
LIB_RDMACM
LIB_HWLOC

AC_CHECK_LIB([m], [sin])

#
# Checks for header files.
#
AC_HEADER_STDC
AC_CHECK_HEADERS([arpa/inet.h fcntl.h inttypes.h netdb.h netinet/in.h stdlib.h string.h sys/ioctl.h sys/socket.h sys/time.h sys/statvfs.h sys/vfs.h sys/mount.h syslog.h unistd.h])

#
# Checks for typedefs, structures, and compiler characteristics.
#
AC_C_CONST
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_HEADER_TIME
AC_STRUCT_TM

#
# Checks for library functions.
#
AC_FUNC_FORK
AC_FUNC_MALLOC
AC_FUNC_MEMCMP
AC_FUNC_MKTIME
AC_FUNC_REALLOC
AC_FUNC_SELECT_ARGTYPES
AC_FUNC_SETVBUF_REVERSED
AC_TYPE_SIGNAL
AC_FUNC_STAT
AC_FUNC_STRFTIME
AC_FUNC_VPRINTF
AC_CHECK_FUNCS([gethostbyname gethostname gettimeofday inet_ntoa memset select socket strchr strcspn strdup strerror strstr])

#if test "x$prefix" != xNONE; then
#  PSRDADA_INSTALL=$prefix
#else
#  PSRDADA_INSTALL=$ac_default_prefix
#fi
#
#AC_SUBST(PSRDADA_INSTALL)

AC_CONFIG_FILES([Makefile
  Management/Makefile
  Management/psrdada_ldflags
  Management/psrdada_cflags
  Management/psrdada.pc
  apps/Makefile
  examples/Makefile
  examples/src/Makefile
  src/Makefile
  src/tests/Makefile])

AC_OUTPUT

