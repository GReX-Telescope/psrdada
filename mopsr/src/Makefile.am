
SUBDIRS = sigproc

lib_LTLIBRARIES = libmopsr.la

mopsr_hdr = mopsr_udp.h mopsr_def.h mopsr_util.h

libmopsr_la_SOURCES = $(mopsr_hdr) mopsr_udp.c mopsr_util.c

INCLUDES = -I$(top_srcdir)/src -I$(HOME)/opt/cub-1.3.2

# add pgplot and fftw configuration 

AM_CFLAGS = @PGPLOT_CFLAGS@ @FFTW_CFLAGS@ -D$(ARCH)

LDADD = libmopsr.la $(top_builddir)/src/libpsrdada.la @PGPLOT_LIBS@ @FFTW_LIBS@
#libmopsr_la_LIBADD = $(top_builddir)/src/libpsrdada.la @PGPLOT_LIBS@ @FFTW_LIBS@

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
							 mopsr_dbintegratedb mopsr_dbrescaledb \
							 mopsr_test_sk_kernel mopsr_dadatestct mopsr_dadasftsplit mopsr_pulsedada \
							 mopsr_dadatfssplit mopsr_dbreblockdb mopsr_dumpplot mopsr_plot_cands \
               mopsr_dbsigproc mopsr_maskplot mopsr_dadahistplot

mopsr_dbsigproc_LDADD = $(LDADD) sigproc/libsigproc.la

#############################################################################
#
# Depends on SOFA library
#
#############################################################################

if HAVE_SOFA

mopsr_hdr += mopsr_delays.h
libmopsr_la_SOURCES += mopsr_delays.c

bin_PROGRAMS += mopsr_phase_differencer mopsr_getmd mopsr_getns \
								mopsr_dadacorrplot_pfb mopsr_getlst mopsr_test_signal_paths \
								mopsr_test_delays_plot mopsr_phase_diff_transit \
							  mopsr_plot_acs

AM_CFLAGS += @SOFA_CFLAGS@
#libmopsr_la_LIBADD += @SOFA_LIBS@
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
#libmopsr_la_LIBADD += @GSL_LIBS@
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

LDADD += $(RDMACM_LIBS)
#libmopsr_la_LIBADD += @RDMACM_LIBS@


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
                mopsr_dbxgpu mopsr_xgpu_convert 

libmopsr_la_SOURCES += mopsr_cuda.cu mopsr_delays_cuda.cu mopsr_delays_cuda.h \
                       mopsr_calib_cuda.cu mopsr_calib_cuda.h

#libmopsr_la_LIBADD += @CUDA_LIBS@ @CUFFT_LIBS@
AM_CFLAGS += @CUDA_CFLAGS@
LDADD += @CUDA_LIBS@ @CUFFT_LIBS@

mopsr_dbxgpu_CFLAGS = $(AM_CFLAGS) -I$(HOME)/xGPU/src/
mopsr_dbxgpu_LDFLAGS = $(LDADD) -L$(HOME)/xGPU/src -lxgpu
mopsr_xgpu_convert_CFLAGS = $(AM_CFLAGS) -I$(HOME)/xGPU/src/ -g
mopsr_xgpu_convert_LDFLAGS = $(LDADD) -static

if HAVE_SOFA

bin_PROGRAMS += mopsr_aqdsp mopsr_bfdsp mopsr_test_delays mopsr_test_bfdsp \
                mopsr_dbdelaydb mopsr_test_bfdsp_tb

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

