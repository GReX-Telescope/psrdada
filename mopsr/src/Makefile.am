
SUBDIRS = sigproc

noinst_LTLIBRARIES = libmopsr.la

mopsr_hdr = mopsr_udp.h mopsr_def.h mopsr_util.h snapsr_udp.h

libmopsr_la_SOURCES = $(mopsr_hdr) mopsr_udp.c mopsr_util.c snapsr_udp.c 

include_HEADERS = mopsr_delays.h mopsr_def.h 

AM_CPPFLAGS = -I$(top_srcdir)/src \
              -I$(top_srcdir)/3rdparty

AM_CFLAGS = @PGPLOT_CFLAGS@ @FFTW_CFLAGS@

LDADD = libmopsr.la \
        $(top_builddir)/src/libpsrdada.la \
        $(top_builddir)/3rdparty/libtimers.la \
        @PGPLOT_LIBS@ @FFTW_LIBS@

bin_PROGRAMS = mopsr_udpheader mopsr_udpdb_thread mopsr_dbsplitdb \
               mopsr_udpdb mopsr_udpdb_16 mopsr_udpdb_dual mopsr_dbresampdb mopsr_dbsumdb \
               mopsr_udpdump mopsr_dbjoindb mopsr_dbchandb \
               mopsr_dbnum mopsr_dadatxt \
               mopsr_udplongprbs mopsr_dadawaterfallplot \
               mopsr_fringeplot mopsr_dadastats \
               mopsr_corr_sum mopsr_corr_fsum mopsr_plot_baseline \
               mopsr_dbtransposedb mopsr_dbselectantsdb mopsr_udpdb_selants \
               mopsr_udphist_16bit mopsr_udpvalidate \
               mopsr_dadapowermonitor mopsr_udphist_8bit \
               mopsr_dadapearsoncoeff mopsr_corr_fscr\
               mopsr_udpcorr mopsr_udpcorr_hires mopsr_test_cpubf mopsr_dbTBdb \
               mopsr_dbintegratedb mopsr_dbrescaledb \
               mopsr_test_sk_kernel mopsr_dadatestct mopsr_dadasftsplit mopsr_pulsedada \
               mopsr_dadatfssplit mopsr_dbreblockdb mopsr_dumpplot mopsr_plot_cands \
               mopsr_maskplot mopsr_dadahistplot mopsr_test_hirespacket \
               caspsr_udpcorr snapsr_udpplot snapsr_udpfftplot mopsr_test_sk_kernel_hires \
               mopsr_test_mergesort snapsr_udpdb mopsr_test_rng mopsr_genbpdada \
               mopsr_udpgen mopsr_udpplot mopsr_udpfftplot mopsr_udpfftplot_16 mopsr_udpwaterfall mopsr_udphist \
               mopsr_udptimeseries mopsr_udpundersampleplot mopsr_dadafftplot mopsr_numdb \
               mopsr_plot_antmasks mopsr_dbfurbydb

#############################################################################
#
# Depends on SOFA library
#
#############################################################################

if HAVE_SOFA

mopsr_hdr += mopsr_delays.h mopsr_delays_hires.h
libmopsr_la_SOURCES += mopsr_delays.c mopsr_delays_hires.c

bin_PROGRAMS += mopsr_phase_differencer mopsr_getmd mopsr_getns \
                mopsr_dadacorrplot_pfb mopsr_getlst mopsr_test_signal_paths \
                mopsr_test_delays_plot mopsr_phase_diff_transit \
								mopsr_test_delays_docjer mopsr_plot_acs mopsr_dbsigproc \
                snapsr_dadacorrplot_full

AM_CFLAGS += @SOFA_CFLAGS@
LDADD += @SOFA_LIBS@

mopsr_dbsigproc_LDADD = $(LDADD) sigproc/libsigproc.la


endif

#############################################################################
#
# Depends on GSL and SOFA
#
#############################################################################

if HAVE_GSL

if HAVE_SOFA

AM_CPPFLAGS += @GSL_CFLAGS@
LDADD += @GSL_LIBS@

bin_PROGRAMS += mopsr_solve_delays mopsr_solve_delays_spectral mopsr_phase_differencer_slope_test mopsr_phase_differencer_slope_test2 mopsr_test_delays_skew
# depends on GSL 2+
# mopsr_phase_differencer_slope_test3
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

LDADD += @RDMACM_LIBS@
CFLAGS += @RDMACM_CFLAGS@

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
                mopsr_test_ant_sum mopsr_test_transpose_TS_to_ST mopsr_test_calib \
                mopsr_test_bfst
libmopsr_la_SOURCES += mopsr_cuda.cu mopsr_delays_cuda.cu mopsr_delays_cuda.h \
                       mopsr_calib_cuda.cu mopsr_calib_cuda.h mopsr_delays_cuda_hires.cu \
                       mopsr_delays_cuda_hires.h

AM_CFLAGS += @CUDA_CFLAGS@
LDADD += @CUDA_LIBS@ @CUFFT_LIBS@


if HAVE_XGPU

bin_PROGRAMS += mopsr_dbxgpu mopsr_xgpu_convert
mopsr_dbxgpu_CFLAGS = $(AM_CFLAGS) @XGPU_CFLAGS@
mopsr_dbxgpu_LDFLAGS = $(LDADD) @XGPU_LIBS@
mopsr_xgpu_convert_CFLAGS = $(AM_CFLAGS) @XGPU_CFLAGS@ -g
mopsr_xgpu_convert_LDFLAGS = $(LDADD) -static

endif

if HAVE_CUB

AM_CPPFLAGS += @CUB_CFLAGS@

endif

if HAVE_SOFA

bin_PROGRAMS += mopsr_aqdsp mopsr_aqdsp_hires mopsr_bfdsp mopsr_test_delays \
                mopsr_test_delays_hires mopsr_test_bfdsp \
                mopsr_dbdelaydb mopsr_test_bfdsp_tb

endif

endif

