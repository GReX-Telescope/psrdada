
dist_bin_SCRIPTS = apsr_rm.csh apsr_edit.csh

bin_PROGRAMS = apsr_udpdb bat_to_utc apsr_udpdisk apsr_udpheader

LDADD = $(top_builddir)/src/libpsrdada.la

INCLUDES = -I$(top_srcdir)/src

#############################################################################
#
# Depends on 3rdparty code
#
#############################################################################

if COMPILE_THIRD

bin_PROGRAMS += apsr_dbudp test_boardudp apsr_udpgenerator

LDADD += $(top_builddir)/3rdparty/libtimers.la
INCLUDES += -I$(top_srcdir)/3rdparty
AM_CFLAGS = -D$(ARCH)

endif

