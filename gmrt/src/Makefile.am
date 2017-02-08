#############################################################################
#
# GMRT udp capture and correlator 
#
#############################################################################

noinst_LTLIBRARIES = libgmrt.la 

gmrt_hdr = gmrt_udp.h gmrt_def.h gmrt_udpdb.h gmrt_udpheader.h gmrt_delay_fns.h

libgmrt_la_SOURCES = gmrt_udp.c gmrt_delay_fns.c gmrt_corrsub.c gmrt_astro_cal.c

bin_PROGRAMS = gmrt_udpdb gmrt_udpdb_multi gmrt_udpheader gmrt_udpheader_multi \
  gmrt_delay gmrt_delay_test

LDADD = libgmrt.la \
  $(top_builddir)/src/libpsrdada.la

AM_CPPFLAGS = -I$(top_srcdir)/src

if HAVE_CUDA

include $(top_srcdir)/cuda.mk

gmrt_cuda_hdr = gmrt_corr_dada.h gmrt_corr_lib.h gmrt_corr_cuda.h

libgmrt_la_SOURCES += $(gmrt_cuda_hdr) gmrt_corr_dada.c gmrt_corr_lib.c \
  gmrt_corr_cuda.cu complex_multiply_device.hpp fill_device.hpp \
  maths_functions.hpp phase_shift_device.hpp unpack_device.hpp \
  device_interface.hpp kernel_defs.hpp maths.hpp stopwatch.hpp \
  fft_device.hpp mac_device.hpp transform_device.hpp

bin_PROGRAMS += gmrt_correlator

gmrt_correlator_SOURCES = gmrt_correlator.C

LDADD += @CUDA_LIBS@ @CUFFT_LIBS@
AM_CPPFLAGS +=  @CUFFT_CFLAGS@ @CUDA_CFLAGS@
AM_CXXFLAGS =  @CUFFT_CFLAGS@ @CUDA_CFLAGS@

#gmrt_correlator_LDADD = $(LDADD) @CUDA_LIBS@ @CUFFT_LIBS@
#gmrt_correlator_CXXFLAGS = @CUFFT_CFLAGS@ @CUDA_CFLAGS@
#AM_CFLAGS = @CUFFT_CFLAGS@ @CUDA_CFLAGS@

endif

