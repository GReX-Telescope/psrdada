LOCAL_ROOT = ../../..
include ${LOCAL_ROOT}/include/Makefile.standard

PROGRAMS := dada_db.c dada_writer.c dada_reader.c dada_dbdisk.c

LDFLAGS += -ldada -lpthread
CPPFLAGS += -D_REENTRANT

include ${LOCAL_ROOT}/include/Makefile.extended

