
lib_LTLIBRARIES = libpic.la

libpic_la_SOURCES = pic.c

bin_PROGRAMS = puma2_dmadb.c

INCLUDES = -I/opt/pic -I/opt/EDTpcd
LDADD = ../src/libdada.la libpic.la -L/opt/EDTpcd -ledt

