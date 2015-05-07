
lib_LTLIBRARIES = libmopsr.la

mopsr_hdr = mopsr_udp.h mopsr_def.h mopsr_util.h

libmopsr_la_SOURCES = $(mopsr_hdr) mopsr_udp.c mopsr_util.c

INCLUDES = -I$(top_srcdir)/src -I$(HOME)/opt/cub-1.3.2

# add pgplot and fftw configuration 

AM_CFLAGS = @PGPLOT_CFLAGS@ @FFTW_CFLAGS@ -D$(ARCH)

LDADD = libmopsr.la $(top_builddir)/src/libpsrdada.la @PGPLOT_LIBS@ @FFTW_LIBS@

bin_PROGRAMS = mopsr_udpheader mopsr_udpdb_thread mopsr_dbsplitdb \
               mopsr_udpdb mopsr_udpdb_16 mopsr_udpdb_dual mopsr_dbresampdb mopsr_dbsumdb \
               mopsr_udpdump mopsr_dbjoindb mopsr_dbchandb \
							 mopsr_dbnum mopsr_dadatxt \
               mopsr_udplongprbs mopsr_dadawaterfallplot \
	    			   mopsr_fringeplot mopsr_solve_delays mopsr_dadastats \
               mopsr_corr_sum mopsr_corr_fsum mopsr_plot_baseline \
               mopsr_dbtransposedb mopsr_dbselectantsdb mopsr_udpdb_selants \
							 mopsr_udphist_16bit \
               mopsr_dadapowermonitor mopsr_udphist_8bit \
               mopsr_dadapearsoncoeff \
							 mopsr_corr_fscr mopsr_solve_delays_spectral \
							 mopsr_udpcorr mopsr_test_cpubf mopsr_dbTBdb \
							 mopsr_dbintegratedb \
							 mopsr_test_sk_kernel mopsr_dadatestct mopsr_dadasftsplit mopsr_pulsedada \
							 mopsr_dadatfssplit mopsr_dbreblockdb

#############################################################################
#
# Depends on SOFA library
#
#############################################################################

if HAVE_SOFA

mopsr_hdr += mopsr_delays.h
libmopsr_la_SOURCES += mopsr_delays.c

bin_PROGRAMS += mopsr_phase_differencer mopsr_getmd \
								mopsr_dadacorrplot_pfb mopsr_getlst mopsr_test_signal_paths \
								mopsr_test_delays_plot mopsr_phase_diff_transit \
							  mopsr_plot_acs

AM_CFLAGS += @SOFA_CFLAGS@
LDADD += @SOFA_LIBS@

endif

#############################################################################
#
# Depends on GSL
#
#############################################################################

if HAVE_GSL

bin_PROGRAMS += mopsr_solve_delays mopsr_solve_delays_spectral

CFLAGS += @GSL_CFLAGS@
LDADD += @GSL_LIBS@

if HAVE_SOFA

bin_PROGRAMS += mopsr_plot_baselines_vivek mopsr_plot_baselines mopsr_print_drifts

endif

endif

#############################################################################
#
# Depends on RDMA and Infiniband
#
#############################################################################

if HAVE_RDMACM

libmopsr_la_SOURCES += mopsr_ib.h mopsr_ib.c

bin_PROGRAMS += mopsr_dbib_FST mopsr_ibdb_FST mopsr_dbstats mopsr_dbib_SFT mopsr_ibdb_SFT

mopsr_dbib_FST_LDADD = $(LDADD) $(RDMACM_LIBS)
mopsr_ibdb_FST_LDADD = $(LDADD) $(RDMACM_LIBS)
mopsr_dbib_SFT_LDADD = $(LDADD) $(RDMACM_LIBS)
mopsr_ibdb_SFT_LDADD = $(LDADD) $(RDMACM_LIBS)

endif

#############################################################################
#
# Depends on CUDA
#
#############################################################################

if HAVE_CUDA

include $(top_srcdir)/cuda.mk

bin_PROGRAMS += mopsr_dbrephasedb mopsr_dbcalib mopsr_test_kernel \
                mopsr_test_transpose_FT_to_TF \
                mopsr_test_ant_sum mopsr_test_transpose_TS_to_ST mopsr_test_calib

libmopsr_la_SOURCES += mopsr_cuda.cu mopsr_delays_cuda.cu mopsr_delays_cuda.h \
                       mopsr_calib_cuda.cu mopsr_calib_cuda.h

AM_CFLAGS += @CUDA_CFLAGS@ @CUFFT_CFLAGS@
LDADD += @CUDA_LIBS@ @CUFFT_LIBS@
LDFLAGS += $(CUDA_LDFLAGS)

if HAVE_SOFA

bin_PROGRAMS += mopsr_aqdsp mopsr_bfdsp mopsr_test_delays mopsr_test_bfdsp \
                mopsr_dbdelaydb

endif

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

