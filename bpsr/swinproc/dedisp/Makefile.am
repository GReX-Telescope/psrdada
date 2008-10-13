
lib_LTLIBRARIES = libdedisp.la

libdedisp_la_SOURCES = getDMtable.C

bin_PROGRAMS = dedisperse_all

dedisperse_all_SOURCES = dedisperse_all.C dedisperse_header.c

LDADD = ../libswinproc.la libdedisp.la

dedisperse_all_LDADD = $(LDADD)

INCLUDES = -I../

AM_CXXFLAGS = -fopenmp
