cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)
# ENABLE ADIOS2_ROOT
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

project(nwchem-sort-trajectory C CXX)

find_package(MPI REQUIRED)
find_package(ADIOS2 REQUIRED)

option(USE_TIMERS "Use profiling timers")
if(USE_TIMERS)
  message(STATUS "Enabling profiling timers")
  add_definitions(-DENABLE_TIMERS)
endif()

# We are not using the C++ API of MPI, this will stop the compiler look for it
add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)

add_executable(nwchem-sort-trajectory-local
  nwchem-sort-trajectory-local.cpp
)
target_link_libraries(nwchem-sort-trajectory-local adios2::adios2 MPI::MPI_C)

add_executable(nwchem-sort-trajectory-global
  nwchem-sort-trajectory-global.cpp
)
target_link_libraries(nwchem-sort-trajectory-global adios2::adios2 MPI::MPI_C)

add_executable(nwchem-dump-streamer-local
  nwchem-dump-streamer-local.cpp
)
target_link_libraries(nwchem-dump-streamer-local adios2::adios2 MPI::MPI_C)

