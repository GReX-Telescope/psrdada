LOCAL_ROOT = ../../..
include $(LOCAL_ROOT)/include/Makefile.standard

LIBRARY = libdada.a

LIB_SRCS = dada_db.c

include $(LOCAL_ROOT)/include/Makefile.extended

