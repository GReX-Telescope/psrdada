
bin_PROGRAMS = bpsr_udpheader bpsr_udpdb bpsr_dbdisk bpsr_udpdisk bpsr_udpdb_fast bpsr_dbmon

LDADD = ../../src/libpsrdada.la

INCLUDES = -I../../src

#############################################################################
#
# Depends on 3rdparty code
#
#############################################################################

if COMPILE_THIRD

bin_PROGRAMS += bpsr_dbavg bpsr_udpgenerator

LDADD += ../../3rdparty/libtimers.la
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

endif

