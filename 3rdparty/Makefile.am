
lib_LTLIBRARIES = libtimers.la

libtimers_la_SOURCES = StopWatch.c RealTime.c Statistics.c

include_HEADERS = arch.h StopWatch.h RealTime.h Statistics.h

ARCH = @ARCH@
libtimers_la_CFLAGS  = -Wall -O4 -D$(ARCH)

