noinst_LTLIBRARIES = libcomap.la

comap_hdr = comap_udp.h comap_def.h

libcomap_la_SOURCES = $(comap_hdr) comap_udp.c

bin_PROGRAMS = comap_udptest comap_udpdb_thread comap_udpdb_preset

LDADD = libcomap.la \
	$(top_builddir)/src/libpsrdada.la \
	$(top_builddir)/3rdparty/libtimers.la

AM_CPPFLAGS = -I$(top_srcdir)/src \
	-I$(top_srcdir)/3rdparty

