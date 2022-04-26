
set(public_headers
    stopwatch.h
    mach_gettime.h
)

set(sources
    mach_gettime.c
    stopwatch.c
)

add_library(3rdparty
    SHARED
    ${sources}
    ${private_headers}
    ${public_headers}
)

target_include_directories(
    3rdparty
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/3rdparty>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    3rdparty
    PUBLIC
)

install(
    TARGETS
        3rdparty
    LIBRARY DESTINATION
        lib
)

install(
    FILES
        ${public_headers}
    DESTINATION
        include
)

