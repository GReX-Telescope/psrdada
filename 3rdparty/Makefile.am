
noinst_LTLIBRARIES = libtimers.la

nobase_include_HEADERS = stopwatch.h

libtimers_la_SOURCES = stopwatch.c mach_gettime.c

bin_PROGRAMS = vsib_proxy vsib_recv vsib_send 

#if HAVE_LINUX

libtimers_la_LIBADD = -lrt

#endif
