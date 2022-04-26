
set(tests
    test_ascii_header.c
    test_command_parser.c
    test_disk_array.c
    test_ipcbuf.c
    test_median_smooth_zap.c
    test_multilog.c
    test_uint64_t.c
)

set(manual_tests
    test_command_parse_server.c
    test_dada_nexus.c
    test_ipcio.c
    test_multilog_server.c
)

foreach(testsourcefile IN LISTS tests)
    string(REPLACE ".c" "" testname ${testsourcefile})
    add_executable(${testname} ${testsourcefile})
    target_link_libraries(${testname} psrdada)
    add_test(${testname} ${testname})
endforeach(testsourcefile IN LISTS tests)

foreach(testsourcefile IN LISTS manualtests)
    string(REPLACE ".c" "" testname ${testsourcefile})
    add_executable(${testname} ${testsourcefile})
    target_link_libraries(${testname} psrdada)
endforeach(testsourcefile IN LISTS manualtests)

