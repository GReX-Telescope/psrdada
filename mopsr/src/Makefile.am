
lib_LTLIBRARIES = libmopsr.la

mopsr_hdr = mopsr_udp.h mopsr_def.h

libmopsr_la_SOURCES = $(mopsr_hdr) mopsr_udp.c

bin_PROGRAMS = mopsr_udpheader mopsr_udpdb_thread

INCLUDES = -I$(top_srcdir)/src
LDADD = libmopsr.la $(top_builddir)/src/libpsrdada.la

#############################################################################
#
# Depends on RDMA and Infiniband
#
#############################################################################

if HAVE_RDMACM

bin_PROGRAMS += mopsr_dbib mopsr_ibdb

mopsr_dbib_LDADD = $(LDADD) $(RDMACM_LIBS)
mopsr_ibdb_LDADD = $(LDADD) $(RDMACM_LIBS)

endif

#############################################################################
#
# Depends on PGPLOT
#
#############################################################################

if HAVE_PGPLOT

bin_PROGRAMS += mopsr_dbplotbp
AM_CPPFLAGS = @PGPLOT_CFLAGS@
mopsr_dbplotbp_LDADD = $(LDADD) @PGPLOT_LIBS@

endif

#############################################################################
#
# Depends on 3rd Party Timers
#
#############################################################################

if COMPILE_THIRD

bin_PROGRAMS += mopsr_udpgen mopsr_udpplot mopsr_udpfftplot

mopsr_udpgen_CFLAGS = -D$(ARCH)
mopsr_udpplot_CFLAGS = -D$(ARCH)
mopsr_udpfftplot_CFLAGS = -D$(ARCH) -I$(HOME)/linux_64/include
mopsr_udpplot_LDADD = $(LDADD) @PGPLOT_LIBS@
mopsr_udpfftplot_LDADD = $(LDADD) @PGPLOT_LIBS@ -L$(HOME)/linux_64/lib -lfftw3f

INCLUDES += -I$(top_srcdir)/3rdparty
LDADD += $(top_builddir)/3rdparty/libtimers.la

endif

