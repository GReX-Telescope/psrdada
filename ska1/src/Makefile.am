
lib_LTLIBRARIES = libska1.la

ska1_hdr = ska1_udp.h ska1_def.h

libska1_la_SOURCES = $(ska1_hdr) ska1_udp.c

bin_PROGRAMS = ska1_udpdb ska1_udptest ska1_udpdb_thread ska1_udptest_vma

LDADD = libska1.la $(top_builddir)/src/libpsrdada.la
INCLUDES = -I$(top_srcdir)/src

#############################################################################
#
# Depends on 3rd Party Timers
#
#############################################################################

if COMPILE_THIRD

bin_PROGRAMS += ska1_udpgen

ska1_udpgen_CFLAGS = -D$(ARCH)
INCLUDES += -I$(top_srcdir)/3rdparty
LDADD += $(top_builddir)/3rdparty/libtimers.la

endif

