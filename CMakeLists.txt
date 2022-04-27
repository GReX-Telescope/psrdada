cmake_minimum_required(VERSION 3.13.4)

file(STRINGS version.txt PSRDada_VERSION)
message(STATUS "Building PSRDada version ${PSRDada_VERSION}")

# Project configuration, specifying version, languages,
project(PSRDada LANGUAGES C VERSION ${PSRDada_VERSION})
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
    set(CMAKE_CUDA_ARCHITECTURES "35;50;72")
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

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(Management)

# ---------------------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------------------

set(export_dest_dir "${CMAKE_INSTALL_PREFIX}/cmake/psrdada")

# Install cmake config + version + target files
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/PSRDadaConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PSRDadaConfig.cmake"
  INSTALL_DESTINATION
    share/ska-pst-recv/cmake
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PSRDadaVersion.cmake"
  COMPATIBILITY
    AnyNewerVersion
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PSRDadaConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PSRDadaVersion.cmake
  DESTINATION
    "${export_dest_dir}"
)

#install(EXPORT PSRDadaTargets
#  FILE
#    PSRDadaTargets.cmake
#  DESTINATION
#    share/ska/pst/recv/cmake
#)

