
lib_LTLIBRARIES = libdada.la

libdada_la_SOURCES = ascii_header.c multilog.c sock.c			\
	ipcutil.c ipcbuf.c ipcio.c disk_array.c node_array.c		\
	command_parse.c command_parse_server.c string_array.c		\
	diff_time.c nexus.c filesize.c fileread.c utc.c daemon.c	\
	dada_hdu.c dada_pwc.c dada_pwc_nexus.c dada_ni.c		\
	dada_pwc_main.c dada_client.c					\
	dada_pwc_nexus_config.c dada_pwc_nexus_header_parse.c		\
	multilog_server.c

include_HEADERS = ascii_header.h multilog.h sock.h		\
	ipcutil.h ipcbuf.h ipcio.h disk_array.h node_array.h	\
	command_parse.h command_parse_server.h string_array.h	\
	diff_time.h nexus.h futils.h utc.h daemon.h dada_def.h	\
	dada_hdu.h dada_pwc.h dada_pwc_nexus.h dada_ni.h	\
	dada_pwc_main.h dada_client.h

bin_PROGRAMS = dada_db dada_write_test dada_dbdisk dada_dbnic dada_nicdb \
	dada_install_header dada_pwc_command

check_PROGRAMS = test_multilog test_ipcbuf test_ipcio loop_ipcio	\
	test_command_parser test_multilog_server telnecho		\
	test_command_parse_server test_dada_nexus test_dada_pwc		\
	dada_writer dada_reader $(TESTS)

TESTS = test_uint64_t test_disk_array

LDADD = libdada.la
