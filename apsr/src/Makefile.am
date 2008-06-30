
bin_PROGRAMS = apsr_udpdb apsr_dbtriwave apsr_dbudp \
	test_boardudp apsr_dbtriripple bat_to_utc apsr_udpdisk apsr_udpheader \
  apsr_udpgenerator

LDADD = ../../src/libdada.la ../../3rdparty/libtimers.la

INCLUDES = -I../../3rdparty -I../../src

ARCH = @ARCH@

AM_CFLAGS = -D$(ARCH)
