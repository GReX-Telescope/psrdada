
lib_LTLIBRARIES = libbpsr.la

bpsr_hdr=bpsr_def.h bpsr_udp.h ibob.h multibob.h

libbpsr_la_SOURCES = $(bpsr_hdr) ibob.c multibob.c start_observation.c

bin_PROGRAMS = bpsr_udpheader bpsr_udpdb bpsr_dbdisk bpsr_udpdisk \
	bpsr_dbmon ibob_telnet multibob_server bpsr_dbavg bpsr_udpgenerator

AM_CPPFLAGS = -I$(top_srcdir)/src \
  -I$(top_srcdir)/3rdparty

LDADD = libbpsr.la  \
  $(top_builddir)/src/libpsrdada.la \
  $(top_builddir)/3rdparty/libtimers.la

#############################################################################
#
# Depends on PGPLOT
#
#############################################################################

if HAVE_PGPLOT

bin_PROGRAMS += bpsr_dbplot bpsr_diskplot bpsr_bramplot bpsr_dadadiskareaplot bpsr_dadadiskplot
AM_CPPFLAGS += @PGPLOT_CFLAGS@
LDADD += @PGPLOT_LIBS@

#bpsr_dbplot_LDADD = $(LDADD) @PGPLOT_LIBS@
#bpsr_diskplot_LDADD = $(LDADD) @PGPLOT_LIBS@
#bpsr_dadadiskplot_LDADD = $(LDADD) @PGPLOT_LIBS@
#bpsr_dadadiskareaplot_LDADD = $(LDADD) @PGPLOT_LIBS@
#bpsr_bramplot_LDADD = $(LDADD) @PGPLOT_LIBS@
endif


