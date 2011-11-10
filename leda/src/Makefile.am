


lib_LTLIBRARIES = libleda.la

leda_hdr = leda_udp.h leda_def.h

libleda_la_SOURCES = $(leda_hdr) leda_udp.c

bin_PROGRAMS = leda_udpdb

LDADD = libleda.la $(top_builddir)/src/libpsrdada.la
INCLUDES = -I$(top_srcdir)/src


#############################################################################
#
# Depends on 3rd Party Timers
#
#############################################################################

if COMPILE_THIRD

bin_PROGRAMS += leda_udpgen

AM_CFLAGS = -D$(ARCH)
INCLUDES += -I$(top_srcdir)/3rdparty
LDADD += $(top_builddir)/3rdparty/libtimers.la

endif

