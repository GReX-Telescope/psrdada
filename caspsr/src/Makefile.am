
lib_LTLIBRARIES = libcaspsr.la

libcaspsr_la_SOURCES = bibob.h bibob.c mulbibob.h mulbibob.c caspsr_udp.h caspsr_udp.c start_observation.c

bin_PROGRAMS = caspsr_udpheader caspsr_udpdb bibob_terminal

LDADD = libcaspsr.la $(top_builddir)/src/libpsrdada.la @PGPLOT_LIBS@

INCLUDES = -I$(top_srcdir)/src

#############################################################################
#
# Depends on 3rdparty code
#
#############################################################################

if COMPILE_THIRD

bin_PROGRAMS += caspsr_udpgen

caspsr_udpgen_LDADD = $(LDADD) $(top_builddir)/3rdparty/libtimers.la

INCLUDES += -I$(top_srcdir)/3rdparty
AM_CFLAGS = -D$(ARCH)

endif

#############################################################################
#
# Depends on PGPLOT
#
#############################################################################

if HAVE_PGPLOT

bin_PROGRAMS += mulbibob_server
AM_CPPFLAGS = @PGPLOT_CFLAGS@
mulbibob_server_LDADD = $(LDADD) @PGPLOT_LIBS@
endif

