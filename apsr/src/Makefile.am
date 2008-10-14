
bin_PROGRAMS = apsr_udpdb bat_to_utc apsr_udpdisk apsr_udpheader

LDADD = ../../src/libpsrdada.la

INCLUDES = -I../../src

#############################################################################
#
# Depends on 3rdparty code
#
#############################################################################

if COMPILE_THIRD

bin_PROGRAMS += apsr_dbudp test_boardudp apsr_udpgenerator

LDADD += ../../3rdparty/libtimers.la
INCLUDES += -I../../3rdparty
AM_CFLAGS = -D$(ARCH)

endif

