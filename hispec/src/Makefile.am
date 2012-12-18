
lib_LTLIBRARIES = libhispec.la

hispec_hdr = hispec_def.h hispec_udp.h

libhispec_la_SOURCES = $(hispec_hdr) hispec_udp.c

bin_PROGRAMS = hispec_udpdb

LDADD = libhispec.la $(top_builddir)/src/libpsrdada.la
AM_CFLAGS = -DUSE_DADA=1

if HAVE_PGPLOT

bin_PROGRAMS += hispec_waterfall_plot
hispec_waterfall_plot_LDADD = $(LDADD) @PGPLOT_LIBS@

endif

if HAVE_FFTW3

bin_PROGRAMS += corr_poly

LDADD += $(FFTW_LIBS)
AM_CFLAGS += $(FFTW_CFLAGS)

if HAVE_CUDA

include $(top_srcdir)/cuda.mk

AM_CFLAGS += -DUSE_GPU=1 $(CUDA_CFLAGS)
CUDA_CFLAGS += -DUSE_DADA=1 -DUSE_GPU=1

libhispec_la_SOURCES += cuda_poly.cu cuda_poly.h

corr_poly_LDADD    = $(LDADD) $(CUDA_LIBS) $(CUFFT_LIBS)
corr_poly_LDFLAGS  = $(CUDA_LDFLAGS) $(CUFFT_LDFLAGS)

endif

endif




INCLUDES = -I$(top_srcdir)/src

