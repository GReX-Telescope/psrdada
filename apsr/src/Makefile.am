
lib_LTLIBRARIES = libapsr.la

libapsr_la_SOURCES = apsr_udp.c apsr_def.h apsr_udp.h apsr_udpdb.h apsr_udpdisk.h apsr_udpheader.h

dist_bin_SCRIPTS = apsr_rm.csh apsr_edit.csh apsr_reprocess.csh

bin_PROGRAMS = apsr_udpdb bat_to_utc apsr_udpdisk apsr_udpheader \
	apsr_dbudp test_boardudp apsr_udpgenerator

LDADD = libapsr.la \
	$(top_builddir)/src/libpsrdada.la \
	$(top_builddir)/3rdparty/libtimers.la

INCLUDES = -I$(top_srcdir)/src \
	-I$(top_srcdir)/3rdparty
