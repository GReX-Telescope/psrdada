
noinst_LTLIBRARIES = libska1.la

ska1_hdr = ska1_udp.h ska1_def.h

libska1_la_SOURCES = $(ska1_hdr) ska1_udp.c

bin_PROGRAMS = ska1_udpdb ska1_udptest ska1_udpdb_thread ska1_udptest_vma ska1_udpgen

AM_CPPFLAGS = -I$(top_srcdir)/src -I$(top_srcdir)/3rdparty
LDADD = libska1.la $(top_builddir)/src/libpsrdada.la  $(top_builddir)/3rdparty/libtimers.la

