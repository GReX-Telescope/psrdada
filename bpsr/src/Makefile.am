
noinst_LTLIBRARIES = libbpsr.la

nobase_include_HEADERS = bpsr_def.h bpsr_udp.h ibob.h multibob.h bpsr_ct.h

libbpsr_la_SOURCES = $(nobase_include_HEADERS) ibob.c multibob.c \
                     start_observation.c bpsr_ct.c

bin_PROGRAMS = bpsr_udpdb bpsr_dbdisk bpsr_udpdisk \
	bpsr_dbmon ibob_telnet multibob_server bpsr_dbavg bpsr_udpgenerator \
  bpsr_dbscrunchdb bpsr_dbnic bpsr_nicdb

LDADD = libbpsr.la \
  $(top_builddir)/src/libpsrdada.la \
  $(top_builddir)/3rdparty/libtimers.la

AM_CPPFLAGS = -I$(top_srcdir)/src \
  -I$(top_srcdir)/3rdparty

if HAVE_CUDA

bin_PROGRAMS += bpsr_udpheader
AM_CPPFLAGS += @CUDA_CFLAGS@
LDADD += @CUDA_LIBS@

endif

#############################################################################
#
# Depends on PGPLOT
#
#############################################################################

if HAVE_PGPLOT

bin_PROGRAMS += bpsr_dbplot bpsr_diskplot bpsr_bramplot bpsr_dadadiskareaplot \
  bpsr_dadadiskplot bpsr_udpplot bpsr_fpdiskplot bpsr_bramplot_cross \
  bpsr_bramplot_hist

AM_CPPFLAGS += @PGPLOT_CFLAGS@
LDADD += @PGPLOT_LIBS@

endif


