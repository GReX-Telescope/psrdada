#############################################################################
#
# Depends on 3rdparty code
#
#############################################################################

if COMPILE_THIRD

lib_LTLIBRARIES = libcaspsr.la

libcaspsr_la_SOURCES = bibob.h bibob.c mulbibob.h mulbibob.c caspsr_udp.h caspsr_udp.c start_observation.c

bin_PROGRAMS = caspsr_udpdb bibob_terminal caspsr_udpdb caspsr_udpthreaded caspsr_udpNdb caspsr_udpheader caspsr_udpgen

LDADD = libcaspsr.la $(top_builddir)/src/libpsrdada.la $(top_builddir)/3rdparty/libtimers.la
INCLUDES = -I$(top_srcdir)/src -I$(top_srcdir)/3rdparty
AM_CFLAGS = -D$(ARCH)

# INCLUDES += -I$(HOME)/packages/src/libpcap-0.9.8.20081128/


#############################################################################
#
# Depends on PGPLOT
#
#############################################################################

if HAVE_PGPLOT

bin_PROGRAMS += mulbibob_server caspsr_udpNnic
AM_CPPFLAGS = @PGPLOT_CFLAGS@
mulbibob_server_LDADD = $(LDADD) @PGPLOT_LIBS@
caspsr_udpNnic_LDADD = $(LDADD) @PGPLOT_LIBS@
endif

endif
