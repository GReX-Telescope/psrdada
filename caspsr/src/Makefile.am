#############################################################################
#
# Depends on 3rdparty code
#
#############################################################################

noinst_LTLIBRARIES = libcaspsr.la

casper_hdr = bibob.h caspsr_def.h mulbibob.h caspsr_rdma.h 

libcaspsr_la_SOURCES = $(casper_hdr) bibob.c mulbibob.c start_observation.c \
	caspsr_udp.c caspsr_udp.h

bin_PROGRAMS = bibob_terminal caspsr_udpthreaded caspsr_udpNdb caspsr_udpgen \
	bibob_start_observation caspsr_udpNdebug caspsr_dbnum caspsr_dbnum_demux \
	caspsr_disknumcheck caspsr_udptester caspsr_dbdecidb caspsr_udpdb \
	caspsr_udpheader caspsr_udpNnic

AM_CPPFLAGS = -I$(top_srcdir)/src \
  -I$(top_srcdir)/3rdparty

LDADD = libcaspsr.la \
  $(top_builddir)/src/libpsrdada.la \
	$(top_builddir)/3rdparty/libtimers.la

#############################################################################
#
# Depends on RDMA and Infiniband
#
#############################################################################

if HAVE_RDMACM

bin_PROGRAMS += caspsr_dbib caspsr_ibdb caspsr_pwc_ibdb
LDADD += @RDMACM_LIBS@

#caspsr_dbib_LDADD = $(LDADD) $(RDMACM_LIBS)
#caspsr_ibdb_LDADD = $(LDADD) $(RDMACM_LIBS)
#caspsr_pwc_ibdb_LDADD = $(LDADD) $(RDMACM_LIBS)

endif

#############################################################################
#
# Depends on PGPLOT
#
#############################################################################

if HAVE_PGPLOT

bin_PROGRAMS += mulbibob_server caspsr_disknumplot
AM_CPPFLAGS += @PGPLOT_CFLAGS@
LDADD += @PGPLOT_LIBS@
#mulbibob_server_LDADD = $(LDADD) @PGPLOT_LIBS@
#caspsr_disknumplot_LDADD = $(LDADD) @PGPLOT_LIBS@
endif

