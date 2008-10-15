
lib_LTLIBRARIES = libbpsr.la

libbpsr_la_SOURCES = ibob.h ibob.c multibob.h multibob.c

bin_PROGRAMS = bpsr_udpheader bpsr_udpdb bpsr_dbdisk bpsr_udpdisk \
	bpsr_udpdb_fast bpsr_dbmon ibob_telnet multibob_server

LDADD = libbpsr.la ../../src/libpsrdada.la

INCLUDES = -I../../src

#############################################################################
#
# Depends on 3rdparty code
#
#############################################################################

if COMPILE_THIRD

bin_PROGRAMS += bpsr_dbavg bpsr_udpgenerator

bpsr_dbavg_LDADD = $(LDADD) ../../3rdparty/libtimers.la 
bpsr_udpgenerator_LDADD = $(LDADD) ../../3rdparty/libtimers.la 

INCLUDES += -I../../3rdparty
AM_CFLAGS = -D$(ARCH)

endif

#############################################################################
#
# Depends on PGPLOT
#
#############################################################################

if HAVE_PGPLOT

bin_PROGRAMS += bpsr_dbplot bpsr_diskplot
AM_CPPFLAGS = @PGPLOT_CFLAGS@
bpsr_dbplot_LDADD = $(LDADD) @PGPLOT_LIBS@
bpsr_diskplot_LDADD = $(LDADD) @PGPLOT_LIBS@

endif

