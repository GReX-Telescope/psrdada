# Makefile for write_read_adjustplot
#LOCAL_ROOT = ../../..
LOCAL_ROOT = ..
include ${LOCAL_ROOT}/include/Makefile.standard

#sardinia
#LCPGPLOT = -L${PGPLOT_DIR} -L/nfs/cluster/psr/packages/linux_64/pgplot -lcpgplot -lpgplot -lpng -lgfortran -L/usr/X11R6/lib -lX11 -lm
#green
#LCPGPLOT = -L$(PGPLOT_DIR) -lcpgplot -lpgplot  -L/usr/X11R6/lib -lX11 -lpng -L/usr/lib/gcc/x86_64-redhat-linux/3.4.6 -lgfortran $(LF77)
#apsr
LCPGPLOT = -L/usr/lib64 -lX11 -lpng -L$(PGPLOT_DIR) -lgfortran -lcpgplot -lpgplot


LIBRARY = libplot4mon.a

LIB_SRCS = get_commandline.c check_file_exists.c display_help.c \
        compute_extremes.c compute_margin.c strings_compare.c \
        read_params.c read_stream.c create_xaxis.c set_array_dim.c \
        create_outputname.c create_labels.c four1.c realft.c do_powerspec.c \
        work_on_data.c whiten.c selectmed.c plot_stream_1D.c plot_stream_2D.c 

LDFLAGS += -lm -lplot4mon $(LCPGPLOT) 

PROGRAMS = plot4mon.c

include $(LOCAL_ROOT)/include/Makefile.extended
