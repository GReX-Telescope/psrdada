
# executables to build

#file( GLOB APP_SOURCES *.c )
#foreach( testsourcefile ${APP_SOURCES} )
#    # I used a simple string replace, to cut off .cpp.
#    string( REPLACE ".cpp" "" testname ${testsourcefile} )
#    add_executable( ${testname} ${testsourcefile} )
#    # Make sure YourLib is linked to each app
#    target_link_libraries( ${testname} YourLib )
#endforeach( testsourcefile ${APP_SOURCES} )

add_executable(dada_db dada_db.c)
add_executable(dada_dbcopydb dada_dbcopydb.c)
add_executable(dada_dbdisk dada_dbdisk.c)
add_executable(dada_dbevent dada_dbevent.c)
add_executable(dada_dbmeminfo dada_dbmeminfo.c)
add_executable(dada_dbmergedb dada_dbmergedb.c)
add_executable(dada_dbmetric dada_dbmetric.c)
add_executable(dada_dbmonitor dada_dbmonitor.c)
add_executable(dada_dbNdb dada_dbNdb.c)
add_executable(dada_dbnic dada_dbnic.c)
add_executable(dada_dbnull dada_dbnull.c)
add_executable(dada_dbnum dada_dbnum.c)
add_executable(dada_dboverflow dada_dboverflow.c)
add_executable(dada_dbrecover dada_dbrecover.c)
add_executable(dada_dbscrubber dada_dbscrubber.c)
add_executable(dada_dbxferinfo dada_dbxferinfo.c)
add_executable(dada_diskdb dada_diskdb.c)
add_executable(dada_edit dada_edit.c)
add_executable(dada_header dada_header.c)
add_executable(dada_install_header dada_install_header.c)
add_executable(dada_junkdb dada_junkdb.c)
add_executable(dada_mem_test dada_mem_test.c)
add_executable(dada_nicdb dada_nicdb.c)
add_executable(dada_numdb dada_numdb.c)
add_executable(dada_pwc_command dada_pwc_command.c)
add_executable(dada_pwc_demo dada_pwc_demo.c)
add_executable(dada_write_block_test dada_write_block_test.c)
add_executable(dada_write_test dada_write_test.c)
add_executable(dada_zerodb dada_zerodb.c)
add_executable(dspsr_start_time dspsr_start_time.c)
add_executable(load_test load_test.c)
add_executable(slow_rm slow_rm.c)
add_executable(test_disk_perf test_disk_perf.c)

target_link_libraries(dada_db PUBLIC psrdada)
target_link_libraries(dada_dbcopydb PUBLIC psrdada)
target_link_libraries(dada_dbdisk PUBLIC psrdada)
target_link_libraries(dada_dbevent PUBLIC psrdada)
target_link_libraries(dada_dbmeminfo PUBLIC psrdada)
target_link_libraries(dada_dbmergedb PUBLIC psrdada)
target_link_libraries(dada_dbmetric PUBLIC psrdada)
target_link_libraries(dada_dbmonitor PUBLIC psrdada)
target_link_libraries(dada_dbNdb PUBLIC psrdada)
target_link_libraries(dada_dbnic PUBLIC psrdada)
target_link_libraries(dada_dbnull PUBLIC psrdada)
target_link_libraries(dada_dbnum PUBLIC psrdada)
target_link_libraries(dada_dboverflow PUBLIC psrdada)
target_link_libraries(dada_dbrecover PUBLIC psrdada)
target_link_libraries(dada_dbscrubber PUBLIC psrdada)
target_link_libraries(dada_dbxferinfo PUBLIC psrdada)
target_link_libraries(dada_diskdb PUBLIC psrdada)
target_link_libraries(dada_edit PUBLIC psrdada)
target_link_libraries(dada_header PUBLIC psrdada)
target_link_libraries(dada_install_header PUBLIC psrdada)
target_link_libraries(dada_junkdb PUBLIC psrdada)
target_link_libraries(dada_mem_test PUBLIC psrdada)
target_link_libraries(dada_nicdb PUBLIC psrdada)
target_link_libraries(dada_numdb PUBLIC psrdada)
target_link_libraries(dada_pwc_command PUBLIC psrdada)
target_link_libraries(dada_pwc_demo PUBLIC psrdada)
target_link_libraries(dada_write_block_test PUBLIC psrdada)
target_link_libraries(dada_write_test PUBLIC psrdada)
target_link_libraries(dada_zerodb PUBLIC psrdada)
target_link_libraries(dspsr_start_time PUBLIC psrdada)
target_link_libraries(load_test PUBLIC psrdada)
target_link_libraries(slow_rm PUBLIC psrdada)
target_link_libraries(test_disk_perf PUBLIC psrdada)

install(
    TARGETS
        dada_db
        dada_dbcopydb
        dada_dbdisk
        dada_dbevent
        dada_dbmeminfo
        dada_dbmergedb
        dada_dbmetric
        dada_dbmonitor
        dada_dbNdb
        dada_dbnic
        dada_dbnull
        dada_dbnum
        dada_dboverflow
        dada_dbrecover
        dada_dbscrubber
        dada_dbxferinfo
        dada_diskdb
        dada_edit
        dada_header
        dada_install_header
        dada_junkdb
        dada_mem_test
        dada_nicdb
        dada_numdb
        dada_pwc_command
        dada_pwc_demo
        dada_write_block_test
        dada_write_test
        dada_zerodb
        dspsr_start_time
        load_test
        slow_rm
        test_disk_perf
    RUNTIME DESTINATION
        bin
)

if ((RDMACM_FOUND) AND (IBVERBS_FOUND))
    add_executable(dada_dbib dada_dbib.c)
    add_executable(dada_dbud dada_dbud.c)
    add_executable(dada_ibdb dada_ibdb.c)
    add_executable(dada_udgen dada_udgen.c)
    add_executable(dada_udheader dada_udheader.c)

    target_link_libraries(dada_dbib PUBLIC psrdada)
    target_link_libraries(dada_dbud PUBLIC psrdada)
    target_link_libraries(dada_ibdb PUBLIC psrdada)
    target_link_libraries(dada_udgen PUBLIC psrdada)
    target_link_libraries(dada_udheader PUBLIC psrdada)

    install(
        TARGETS
            dada_dbib
            dada_ibdb
            dada_dbud
            dada_udgen
            dada_udheader
        RUNTIME DESTINATION
            bin
    )
endif()

if(CUDAToolkit_FOUND)
    add_executable(dada_dbgpu dada_dbgpu.c)
    target_link_libraries(dada_dbgpu PUBLIC psrdada)
    install(
        TARGETS
            dada_dbgpu
        RUNTIME DESTINATION
            bin
    )
endif()
