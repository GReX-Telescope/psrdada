LOCAL_ROOT = ../../..
include $(LOCAL_ROOT)/include/Makefile.standard

LIBRARY = libpic.a
LIB_SRCS =  pic.c
INCFLAGS += -I/opt/pic -I/opt/EDTpcd
LDFLAGS += -L/opt/EDTpcd -ldada -ledt -lpic -lpthread

PROGRAMS = puma2_dmadb.c testpic.c

include $(LOCAL_ROOT)/include/Makefile.extended

