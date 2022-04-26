
set(public_headers
    ascii_header.h
    command_parse.h
    command_parse_server.h
    dada_affinity.h
    dada_client.h
    dada_def.h
    dada_generator.h
    dada_hdu.h
    dada_msg.h
    dada_ni.h
    dada_pwc.h
    dada_pwc_main.h
    dada_pwc_main_multi.h
    dada_pwc_nexus.h
    dada_udp.h
    daemon.h
    diff_time.h
    disk_array.h
    futils.h
    ipcbuf.h
    ipcio.h
    ipcutil.h
    median_smooth_zap.h
    monitor.h
    multilog.h
    nexus.h
    node_array.h
    sock.h
    string_array.h
    tmutil.h
)

set(sources
    ascii_header.c
    command_parse.c
    command_parse_server.c
    dada_affinity.c
    dada_client.c
    dada_generator.c
    dada_hdu.c
    dada_ni.c
    dada_pwc.c
    dada_pwc_main.c
    dada_pwc_main_multi.c
    dada_pwc_nexus.c
    dada_pwc_nexus_config.c
    dada_pwc_nexus_header_parse.c
    dada_udp.c
    daemon.c
    diff_time.c
    disk_array.c
    fileread.c
    filesize.c
    ipcbuf.c
    ipcio.c
    ipcutil.c
    monitor.c
    multilog.c
    multilog_server.c
    nexus.c
    node_array.c
    sock.c
    string_array.c
    tmutil.c
)

if((RDMACM_FOUND) AND (IBVERBS_FOUND))
    list(APPEND optional_sources
        dada_ib.c
        dada_ib_datagram.c
    )
    list(APPEND optional_public_headers
        dada_ib.h
        dada_ib_datagram.h
    )
endif()

if(CUDAToolkit_FOUND)
    list(APPEND optional_sources
        dada_cuda.cu
        ipcbuf_cuda.cu
        ipcio_cuda.cu
        ipcutil_cuda.cu
    )

    list(APPEND optional_public_headers
        dada_cuda.h
        ipcbuf_cuda.h
        ipcio_cuda.h
        ipcutil_cuda.h
    )
endif()

set(private_headers
)

add_library(psrdada
    SHARED
    ${sources}
    ${optional_sources}
    ${private_headers}
    ${public_headers}
    ${optional_public_headers}
)

target_include_directories(
    psrdada
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    psrdada
    PUBLIC
        m
)

# Add required libraries to the psrdada target if they are available

if(CUDAToolkit_FOUND)
    target_link_libraries(psrdada PUBLIC CUDA::cudart)
endif()

if(HWLOC_FOUND)
    target_link_libraries(psrdada PUBLIC hwloc)
endif()

if((RDMACM_FOUND) AND (IBVERBS_FOUND))
    target_link_libraries(psrdada PUBLIC rdmacm ibverbs)
endif()

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()

install(
    TARGETS
        psrdada
    LIBRARY DESTINATION
        lib
)

install(
    FILES
        ${public_headers}
    DESTINATION
        include
)
