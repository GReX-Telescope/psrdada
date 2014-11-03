
lib_LTLIBRARIES = libmopsr.la

mopsr_hdr = mopsr_udp.h mopsr_def.h mopsr_util.h mopsr_delays.h

libmopsr_la_SOURCES = $(mopsr_hdr) mopsr_udp.c mopsr_util.c mopsr_delays.c

INCLUDES = -I$(top_srcdir)/src
# add pgplot and fftw configuration 

AM_CFLAGS = @PGPLOT_CFLAGS@ @FFTW_CFLAGS@ -D$(ARCH)

LDADD = libmopsr.la $(top_builddir)/src/libpsrdada.la $(top_builddir)/mopsr/sla/libsla.la @PGPLOT_LIBS@ @FFTW_LIBS@

bin_PROGRAMS = mopsr_udpheader mopsr_udpdb_thread mopsr_dbsplitdb \
               mopsr_udpdb mopsr_udpdb_16 mopsr_udpdb_dual mopsr_dbresampdb mopsr_dbsumdb \
               mopsr_udpdump mopsr_dbjoindb mopsr_dbchandb \
							 mopsr_test_signal_paths mopsr_dbnum mopsr_dadatxt \
               mopsr_udplongprbs mopsr_dadawaterfallplot \
	    			   mopsr_fringeplot mopsr_solve_delays \
               mopsr_corr_sum mopsr_corr_fsum mopsr_plot_baseline mopsr_plot_baselines \
               mopsr_dbtransposedb mopsr_dbselectantsdb mopsr_udpdb_selants \
							 mopsr_plot_acs mopsr_udphist_16bit mopsr_phase_differencer \
               mopsr_dadapowermonitor mopsr_getlst mopsr_getmd mopsr_udphist_8bit \
               mopsr_print_drifts mopsr_dadapearsoncoeff \
							 mopsr_corr_fscr mopsr_solve_delays_spectral mopsr_plot_baselines_vivek \
							 mopsr_udpcorr mopsr_test_cpubf mopsr_dbTBdb

#############################################################################
#
# Depends on GSL
#
#############################################################################

if HAVE_GSL

bin_PROGRAMS += mopsr_solve_delays mopsr_solve_delays_spectral \
	mopsr_plot_baselines mopsr_plot_baselines_vivek mopsr_print_drifts

CFLAGS += @GSL_CFLAGS@
LDADD += @GSL_LIBS@

endif

#############################################################################
#
# Depends on RDMA and Infiniband
#
#############################################################################

if HAVE_RDMACM

libmopsr_la_SOURCES += mopsr_ib.h mopsr_ib.c

bin_PROGRAMS += mopsr_dbib_FST mopsr_ibdb_FST mopsr_dbstats

mopsr_dbib_FST_LDADD = $(LDADD) $(RDMACM_LIBS)
mopsr_ibdb_FST_LDADD = $(LDADD) $(RDMACM_LIBS)

endif

#############################################################################
#
# Depends on CUDA
#
#############################################################################

if HAVE_CUDA

include $(top_srcdir)/cuda.mk

bin_PROGRAMS += mopsr_dbdelaydb mopsr_dbrephasedb mopsr_dbcalib mopsr_test_kernel \
                mopsr_test_delays mopsr_aqdsp mopsr_test_transpose_FT_to_TF \
                mopsr_test_ant_sum mopsr_test_transpose_TS_to_ST mopsr_test_calib \
                mopsr_bfdsp mopsr_test_bfdsp

libmopsr_la_SOURCES += mopsr_cuda.cu mopsr_delays_cuda.cu mopsr_delays_cuda.h \
                       mopsr_calib_cuda.cu mopsr_calib_cuda.h

AM_CFLAGS += @CUDA_CFLAGS@ @CUFFT_CFLAGS@
LDADD += @CUDA_LIBS@ @CUFFT_LIBS@
LDFLAGS += $(CUDA_LDFLAGS)

endif

#############################################################################
#
# Depends on 3rd Party Timers
#
#############################################################################

if COMPILE_THIRD

bin_PROGRAMS += mopsr_udpgen mopsr_udpplot mopsr_udpfftplot \
	mopsr_udpfftplot_16 mopsr_udpwaterfall mopsr_udphist mopsr_udptimeseries \
	mopsr_udpundersampleplot mopsr_dadafftplot mopsr_numdb

INCLUDES += -I$(top_srcdir)/3rdparty
LDADD += $(top_builddir)/3rdparty/libtimers.la

endif

