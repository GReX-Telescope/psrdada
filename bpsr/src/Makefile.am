
bin_PROGRAMS = bpsr_udpheader bpsr_udpdb bpsr_dbdisk bpsr_dbavg bpsr_udpdisk bpsr_udpdb_fast bpsr_dbplot bpsr_udpgenerator bpsr_dbmon bpsr_diskplot

LDADD = ../../src/libdada.la ../../3rdparty/libtimers.la @PGPLOT_LIBS@

INCLUDES = -I../../3rdparty -I../../src @PGPLOT_CFLAGS@

ARCH = @ARCH@

AM_CFLAGS = -D$(ARCH)

