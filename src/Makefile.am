
lib_LTLIBRARIES = libpsrdada.la

psrdada_hdr=ascii_header.h command_parse.h command_parse_server.h \
	    dada_client.h dada_def.h dada_generator.h dada_hdu.h  \
	    dada_msg.h dada_ni.h dada_pwc.h dada_pwc_main.h       \
	    dada_pwc_main_multi.h                                 \
	    dada_pwc_nexus.h dada_udp.h daemon.h diff_time.h      \
	    disk_array.h futils.h ipcbuf.h ipcio.h ipcutil.h      \
	    median_smooth_zap.h monitor.h multilog.h nexus.h      \
	    node_array.h sock.h string_array.h tmutil.h udp.h

libpsrdada_la_SOURCES = $(psrdada_hdr) ascii_header.c multilog.c sock.c monitor.c		\
	ipcutil.c ipcbuf.c ipcio.c disk_array.c node_array.c		\
	command_parse.c command_parse_server.c string_array.c		\
	diff_time.c nexus.c filesize.c fileread.c tmutil.c daemon.c	\
	dada_hdu.c dada_pwc.c dada_pwc_nexus.c dada_ni.c		\
	dada_pwc_main.c dada_pwc_main_multi.c dada_client.c					\
	dada_pwc_nexus_config.c dada_pwc_nexus_header_parse.c		\
	multilog_server.c dada_udp.c dada_generator.c dada_affinity.c

include_HEADERS = ascii_header.h multilog.h sock.h monitor.h	  \
	ipcutil.h ipcbuf.h ipcio.h disk_array.h node_array.h	  \
	command_parse.h command_parse_server.h string_array.h	  \
	diff_time.h nexus.h futils.h tmutil.h daemon.h dada_def.h \
	dada_hdu.h dada_pwc.h dada_pwc_nexus.h dada_ni.h	  \
	dada_pwc_main.h dada_pwc_main_multi.h dada_client.h udp.h \
	dada_udp.h dada_generator.h dada_affinity.h dada_msg.h

bin_PROGRAMS = dada_db dada_write_test dada_write_block_test dada_dbdisk dada_diskdb	 \
	dada_dbnic dada_nicdb dada_dbNdb			 \
	dada_install_header dada_pwc_command dada_pwc_demo	 \
	dada_dbnull dada_dbmonitor dada_dbxferinfo dada_dbmetric \
	dada_header dada_edit slow_rm dada_junkdb dada_mem_test  \
	daemon_test load_test dada_numdb dada_dbnum dada_dbmeminfo \
	dspsr_start_time dada_dboverflow dada_dbscrubber test_disk_perf \
  dada_dbcopydb dada_dbevent dada_dbrecover dada_dbmergedb dada_zerodb

bin_SCRIPTS = ipcrme loopssh

check_PROGRAMS = test_multilog test_ipcbuf loop_ipcio	\
	test_command_parser test_multilog_server telnecho		\
	test_command_parse_server test_dada_nexus			\
	dada_writer dada_reader $(TESTS)

TESTS = test_uint64_t test_disk_array test_ipcio test_ascii_header

LDADD = libpsrdada.la 
AM_CFLAGS = @OPENMP_CFLAGS@
AM_CPPFLAGS =

EXTRA_DIST=ipcrme loopssh

###############################################################################
#
# Depends on HWLOC
#
###############################################################################

if HAVE_HWLOC

dada_db_LDFLAGS = $(LDADD) @HWLOC_LIBS@

endif

libpsrdada_la_LIBADD = @CUDA_LIBS@

###############################################################################
#
# Depends on CUDA
#
###############################################################################
if HAVE_CUDA

include $(top_srcdir)/cuda.mk

libpsrdada_la_SOURCES += dada_cuda.cu dada_cuda.h ipcio_cuda.cu ipcbuf_cuda.cu \
	ipcio_cuda.h ipcbuf_cuda.h ipcutil_cuda.cu

include_HEADERS += dada_cuda.h ipcio_cuda.h ipcbuf_cuda.h ipcutil_cuda.h

bin_PROGRAMS += dada_dbgpu

AM_CPPFLAGS += @CUDA_CFLAGS@ 
LDADD += @CUDA_LIBS@

endif

###############################################################################
#
# Depends on librdmacm
#
###############################################################################

if HAVE_RDMACM

libpsrdada_la_SOURCES += dada_ib.c dada_ib_datagram.c
include_HEADERS += dada_ib.h dada_ib_datagram.h
bin_PROGRAMS += dada_ibdb dada_dbib dada_udheader dada_dbud dada_udgen

AM_CPPFLAGS += -I$(top_srcdir)/3rdparty
libpsrdada_la_LIBADD += @RDMACM_LIBS@
LDADD += @RDMACM_LIBS@  $(top_builddir)/3rdparty/libtimers.la

endif

###############################################################################
#
# Depends on FFTW and PGPLOT
#
###############################################################################

if HAVE_FFTW3

if HAVE_PGPLOT

bin_PROGRAMS += dada_corrplot

AM_CPPFLAGS += @PGPLOT_CFLAGS@ @FFTW_CFLAGS@
LDADD += @PGPLOT_LIBS@ @FFTW_LIBS@

endif

endif

