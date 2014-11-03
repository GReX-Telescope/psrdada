
lib_LTLIBRARIES = libmopsr.la

mopsr_hdr = mopsr_udp.h mopsr_def.h mopsr_util.h

libmopsr_la_SOURCES = $(mopsr_hdr) mopsr_udp.c mopsr_util.c mopsr_delays.c

bin_PROGRAMS = mopsr_udpheader mopsr_udpdb_thread mopsr_dbsplitdb \
               mopsr_udpdb mopsr_udpdb_16 mopsr_udpdb_dual mopsr_dbresampdb mopsr_dbsumdb \
               mopsr_udpdump mopsr_dbjoindb mopsr_dbchandb \
							 mopsr_test_signal_paths mopsr_dbnum mopsr_dadatxt \
               mopsr_dadacorrplot mopsr_dadacorrplot_mb mopsr_dadacorrplot_orig \
               mopsr_dadacorrplot_st mopsr_udplongprbs mopsr_dadawaterfallplot \
	    			   mopsr_dadacorrplot_pfb mopsr_fringeplot mopsr_solve_delays \
               mopsr_corr_sum mopsr_corr_fsum mopsr_plot_baseline mopsr_plot_baselines \
               mopsr_dbtransposedb mopsr_dbselectantsdb mopsr_udpdb_selants \
							 mopsr_plot_acs mopsr_udphist_16bit mopsr_phase_differencer \
               mopsr_dadapowermonitor mopsr_getlst mopsr_getmd mopsr_udphist_8bit \
               mopsr_print_drifts mopsr_dadacorrplot_pfb_raw mopsr_dadapearsoncoeff \
							 mopsr_corr_fscr mopsr_solve_delays_spectral mopsr_plot_baselines_vivek \
							 mopsr_udpcorr mopsr_test_cpubf mopsr_dbTBdb


INCLUDES = -I$(top_srcdir)/src -I$(HOME)/opt/sla/include -I$(HOME)/linux_64/include
LDADD = libmopsr.la $(top_builddir)/src/libpsrdada.la $(HOME)/opt/sla/lib/libsla.a -L$(HOME)/linux_64/lib -lfftw3f @PGPLOT_LIBS@
AM_CPPFLAGS = 

mopsr_dbresampdb_LDADD = $(LDADD) @PGPLOT_LIBS@
mopsr_udpdb_CFLAGS = -D$(ARCH)
mopsr_udpdb_selants_CFLAGS = -D$(ARCH)
mopsr_udpdb_16_CFLAGS = -D$(ARCH)
mopsr_udphist_8bit_CFLAGS = -D$(ARCH)
mopsr_udpcorr_CFLAGS = -D$(ARCH)
mopsr_udphist_16bit_CFLAGS = -D$(ARCH)
mopsr_test_cpubf_CFLAGS = -D$(ARCH)
mopsr_solve_delays_LDADD = $(LDADD) -lgsl -lgslcblas
mopsr_solve_delays_spectral_LDADD = $(LDADD) -lgsl -lgslcblas
mopsr_plot_baselines_LDADD = $(LDADD) -lgsl -lgslcblas
mopsr_plot_baselines_vivek_LDADD = $(LDADD) -lgsl -lgslcblas
mopsr_print_drifts_LDADD = $(LDADD) -lgsl -lgslcblas

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
# Depends on PGPLOT
#
#############################################################################

if HAVE_PGPLOT

bin_PROGRAMS += mopsr_dbplotbp mopsr_dbmonitor mopsr_dumpplot mopsr_rxplot \
								mopsr_dadaplot mopsr_pmplot mopsr_test_delays_plot

AM_CPPFLAGS += @PGPLOT_CFLAGS@

endif

if HAVE_CUDA

include $(top_srcdir)/cuda.mk

bin_PROGRAMS += mopsr_dbdelaydb mopsr_dbrephasedb mopsr_dbcalib mopsr_test_kernel \
                mopsr_test_delays mopsr_aqdsp mopsr_test_transpose_FT_to_TF \
                mopsr_test_ant_sum mopsr_test_transpose_TS_to_ST mopsr_test_calib \
                mopsr_bfdsp mopsr_test_bfdsp

libmopsr_la_SOURCES += mopsr_cuda.cu mopsr_delays_cuda.cu mopsr_delays_cuda.h \
                       mopsr_calib_cuda.cu mopsr_calib_cuda.h

AM_CPPFLAGS += @CUDA_CFLAGS@

mopsr_dbdelaydb_LDADD      = $(LDADD) $(CUDA_LIBS)
mopsr_dbdelaydb_LDFLAGS    = $(CUDA_LDFLAGS)
mopsr_dbrephasedb_LDADD    = $(LDADD) $(CUDA_LIBS)
mopsr_dbrephasedb_LDFLAGS  = $(CUDA_LDFLAGS)
mopsr_dbcalib_LDADD        = $(LDADD) $(CUDA_LIBS) $(CUFFT_LIBS)
mopsr_dbcalib_LDFLAGS      = $(CUDA_LDFLAGS)  
mopsr_test_calib_LDADD     = $(LDADD) $(CUDA_LIBS) $(CUFFT_LIBS)
mopsr_test_calib_LDFLAGS   = $(CUDA_LDFLAGS)
mopsr_test_kernel_LDADD    = $(LDADD) $(CUDA_LIBS)
mopsr_test_kernel_LDFLAGS  = $(CUDA_LDFLAGS) 
mopsr_test_delays_LDADD    = $(LDADD) $(CUDA_LIBS)
mopsr_test_delays_LDFLAGS  = $(CUDA_LDFLAGS) 
mopsr_test_transpose_FT_to_TF_LDADD    = $(LDADD) $(CUDA_LIBS)
mopsr_test_transpose_FT_to_TF_LDFLAGS  = $(CUDA_LDFLAGS) 
mopsr_test_transpose_TS_to_ST_LDADD    = $(LDADD) $(CUDA_LIBS)
mopsr_test_transpose_TS_to_ST_LDFLAGS  = $(CUDA_LDFLAGS)
mopsr_test_ant_sum_LDADD    = $(LDADD) $(CUDA_LIBS)
mopsr_test_ant_sum_LDFLAGS  = $(CUDA_LDFLAGS) 
mopsr_aqdsp_LDADD    = $(LDADD) $(CUDA_LIBS)
mopsr_aqdsp_LDFLAGS  = $(CUDA_LDFLAGS) 
mopsr_bfdsp_LDADD    = $(LDADD) $(CUDA_LIBS)
mopsr_bfdsp_LDFLAGS  = $(CUDA_LDFLAGS) 
mopsr_test_bfdsp_LDADD    = $(LDADD) $(CUDA_LIBS)
mopsr_test_bfdsp_LDFLAGS  = $(CUDA_LDFLAGS) 


endif

#############################################################################
#
# Depends on 3rd Party Timers
#
#############################################################################

if COMPILE_THIRD

bin_PROGRAMS += mopsr_udpgen mopsr_udpplot mopsr_udpfftplot mopsr_udpfftplot_16 mopsr_udpwaterfall mopsr_udphist mopsr_udptimeseries mopsr_udpundersampleplot mopsr_dadafftplot mopsr_numdb

mopsr_udpgen_CFLAGS = -D$(ARCH)
mopsr_udpplot_CFLAGS = -D$(ARCH)
mopsr_udpwaterfall_CFLAGS = -D$(ARCH)
mopsr_udpfftplot_CFLAGS = -D$(ARCH) -I$(HOME)/linux_64/include
mopsr_udpfftplot_16_CFLAGS = -D$(ARCH) -I$(HOME)/linux_64/include
mopsr_udpundersampleplot_CFLAGS = -D$(ARCH) -I$(HOME)/linux_64/include
mopsr_udphist_CFLAGS = -D$(ARCH)
mopsr_udptimeseries_CFLAGS = -D$(ARCH)
mopsr_numdb_CFLAGS = -D$(ARCH)
mopsr_dadafftplot_CFLAGS = -I$(HOME)/linux_64/include -g
mopsr_udpplot_LDADD = $(LDADD) @PGPLOT_LIBS@
mopsr_udpwaterfall_LDADD = $(LDADD) @PGPLOT_LIBS@
mopsr_udpfftplot_LDADD = $(LDADD) @PGPLOT_LIBS@
mopsr_udpfftplot_16_LDADD = $(LDADD) @PGPLOT_LIBS@
mopsr_udpundersampleplot_LDADD = $(LDADD) @PGPLOT_LIBS@
mopsr_udphist_LDADD = $(LDADD) @PGPLOT_LIBS@
mopsr_udptimeseries_LDADD = $(LDADD) @PGPLOT_LIBS@
mopsr_dadafftplot_LDADD = $(LDADD) @PGPLOT_LIBS@

INCLUDES += -I$(top_srcdir)/3rdparty
LDADD += $(top_builddir)/3rdparty/libtimers.la

endif

