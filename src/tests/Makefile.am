
check_PROGRAMS = \
  test_dada_nexus\
  test_ipcbuf \
  test_multilog \
  $(TESTS)

TESTS = \
  test_command_parser \
  test_uint64_t \
  test_disk_array \
  test_ascii_header

bin_PROGRAMS = \
  test_command_parse_server \
  test_ipcio \
  test_multilog_server

AM_CFLAGS = -I$(top_srcdir)/src
LDADD = $(top_builddir)/src/libpsrdada.la
