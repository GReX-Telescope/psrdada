lib_LTLIBRARIES = libleda.la

leda_hdr = leda_udp.h leda_def.h

libleda_la_SOURCES = $(leda_hdr) leda_udp.c

bin_PROGRAMS = leda_udpdb leda_udptest leda_udpdb_thread leda_dbupdb_lookup leda_udpgen

LDADD = libleda.la \
	$(top_builddir)/src/libpsrdada.la \
	$(top_builddir)/3rdparty/libtimers.la

INCLUDES = -I$(top_srcdir)/src \
	-I$(top_srcdir)/3rdparty

if HAVE_IPP

bin_PROGRAMS += leda_dbupdb_ipp
AM_CFLAGS = $(IPP_CFLAGS)
LDADD += $(IPP_LIBS)

endif
