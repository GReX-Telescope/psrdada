
bin_PROGRAMS = \
  dada_db \
  dada_write_test \
  dada_write_block_test \
  dada_dbdisk dada_diskdb \
  dada_dbnic dada_nicdb \
  dada_dbNdb \
  dada_install_header \
  dada_pwc_command \
  dada_pwc_demo	 \
  dada_dbnull \
  dada_dbmonitor \
  dada_dbxferinfo \
  dada_dbmetric \
  dada_header \
  dada_edit \
  slow_rm \
  dada_junkdb \
  dada_mem_test  \
  daemon_test \
  load_test \
  dada_dbnum dada_numdb \
  dada_dbmeminfo \
  dspsr_start_time \
  dada_dboverflow \
  dada_dbscrubber \
  test_disk_perf \
  dada_dbcopydb \
  dada_dbevent \
  dada_dbrecover \
  dada_dbmergedb \
  dada_zerodb

bin_SCRIPTS = \
  ipcrme

AM_CFLAGS = \
  -I$(top_srcdir)/3rdparty \
  -I$(top_srcdir)/src

LDADD = \
  $(top_builddir)/3rdparty/libtimers.la \
  $(top_builddir)/src/libpsrdada.la 

EXTRA_DIST = ipcrme

###############################################################################
#
# Depends on HWLOC
#
###############################################################################

if HAVE_HWLOC

LDADD += @HWLOC_LIBS@ 

endif

###############################################################################
#
# Depends on CUDA
#
###############################################################################

if HAVE_CUDA

bin_PROGRAMS += dada_dbgpu

LDADD += @CUDA_LIBS@

include $(top_srcdir)/config/Makefile.cuda

endif

###############################################################################
#
# Depends on librdmacm
#
###############################################################################

if HAVE_RDMACM

bin_PROGRAMS += \
  dada_dbib dada_ibdb \
  dada_dbud \
  dada_udheader  \
  dada_udgen

LDADD += @RDMACM_LIBS@

endif


