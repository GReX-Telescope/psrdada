bin_PROGRAMS = vsib_proxy vsib_recv vsib_send 

lib_LTLIBRARIES = libtimers.la

libtimers_la_SOURCES = stopwatch.c mach_gettime.c

include_HEADERS = stopwatch.h

if HAVE_LINUX

libtimers_la_LIBADD = -lrt

endif
