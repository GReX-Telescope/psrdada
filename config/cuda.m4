#
# LIB_CUDA([ACTION-IF-FOUND [,ACTION-IF-NOT-FOUND]])
#
# cuda_nvcc   - autoconfig variable with path to nvcc compier
# CUDA_CFLAGS - autoconfig varibale with flags for compiling
# CUDA_LIBS   - autoconfig variable with flags for linking
# HAVE_CUDA   - automake conditional for prescense of CUDA
# ----------------------------------------------------------
#
AC_DEFUN([LIB_CUDA],
[
  AC_PROVIDE([LIB_CUDA])

  AC_ARG_WITH([cuda-dir],
          AC_HELP_STRING([--with-cuda-dir=DIR],
                  [Specify dir where the CUDA library is installed]))

  if test x"$with_cuda_dir" == x; then
    AC_PATH_PROG(cuda_nvcc, nvcc, no)
    cuda_bin_dir=`dirname $cuda_nvcc`
    with_cuda_dir=`dirname $cuda_bin_dir`
  fi

  ac_CUDA_NVCC="$with_cuda_dir/bin/nvcc \$(CUDA_NVCC_FLAGS) -Xcompiler \"\$(DEFAULT_INCLUDES) \$(INCLUDES) \$(AM_CPPFLAGS) \$(CPPFLAGS)\""

  AC_MSG_CHECKING([for CUDA installation])

  ac_save_CFLAGS=$CFLAGS
  ac_save_CXXFLAGS=$CXXFLAGS
  ac_save_LIBS=$LIBS

  ac_CUDA_CFLAGS="-I$with_cuda_dir/include"
  ac_CUDA_LIBS="-L$with_cuda_dir/lib64 -lcudart"

  CFLAGS="$CFLAGS $ac_CUDA_CFLAGS"
  CXXFLAGS="$CXXFLAGS $ac_CUDA_CFLAGS"
  LIBS="$LIBS $ac_CUDA_LIBS"

  AC_TRY_LINK([#include <cuda_runtime.h>],[cudaMalloc (0, 0);],
              have_cuda=yes, have_cuda=no)

  AC_MSG_RESULT($have_cuda)

  if test "$have_cuda" = "yes"; then

    AC_DEFINE([HAVE_CUDA],[1],[Define if the CUDA library is present])
    AC_SUBST(CUDA_NVCC, $ac_CUDA_NVCC)
    AC_SUBST(CUDA_LIBS, $ac_CUDA_LIBS)
    AC_SUBST(CUDA_CFLAGS, $ac_CUDA_CFLAGS)

  else

    if test "$have_cuda" = "no"; then
      AC_MSG_WARN([Ensure that CUDA nvcc is in PATH, or use the --with-cuda-dir option.])
    fi
    [$2]

  fi

  LIBS="$ac_save_LIBS"
  CFLAGS="$ac_save_CFLAGS"
  CXXFLAGS="$ac_save_CXXFLAGS"

  AM_CONDITIONAL(HAVE_CUDA,[test "$have_cuda" = "yes"])
])
