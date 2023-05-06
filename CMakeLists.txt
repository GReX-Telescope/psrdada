cmake_minimum_required(VERSION 3.13.4)

file(STRINGS version.txt PSRDada_VERSION)

# Project configuration, specifying version, languages,
project(PSRDada LANGUAGES C VERSION ${PSRDada_VERSION})

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  if(DEFINED ENV{PSRHOME})
    SET(CMAKE_INSTALL_PREFIX $ENV{PSRHOME})
  endif(DEFINED ENV{PSRHOME})
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

message(STATUS "Building PSRDada ${PSRDada_VERSION} into ${CMAKE_INSTALL_PREFIX}")

set(CMAKE_PROJECT_HOMEPAGE_URL http://psrdada.sourceforge.net/)
set(CMAKE_PROJECT_DESCRIPTION "PSRDADA: Distributed Data Acquisition")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(dependencies)
include(CTest)

find_dependencies()
find_package(CUDAToolkit)
find_package(rdmacm)
find_package(HWLOC)

if (RDMACM_FOUND)
  set(HAVE_RDMACM on)
endif()

if (CUDAToolkit_FOUND)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60;75;80")
  endif()
  enable_language(CUDA)
  set(HAVE_CUDA on)
endif()
if (HWLOC_FOUND)
  set(HAVE_HWLOC on)
endif()
set(HAVE_SYS_STATVFS_H on)
set(HAVE_SYS_VFS_H on)

configure_file(cmake_config.h.in config.h)

include(GNUInstallDirs)

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(Management)

# ---------------------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------------------
export(TARGETS psrdada FILE "${CMAKE_CURRENT_BINARY_DIR}/PSRDadaConfig.cmake")
install(EXPORT PSRDadaConfig DESTINATION "${CMAKE_INSTALL_DATADIR}/PSRDada/cmake")
