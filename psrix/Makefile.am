
noinst_LTLIBRARIES = libputil.la

libputil_la_SOURCES = putil.c

bin_PROGRAMS = udp2db

LDADD = $(top_builddir)/src/libpsrdada.la libputil.la

AM_CPPFLAGS = -I$(top_srcdir)/src

