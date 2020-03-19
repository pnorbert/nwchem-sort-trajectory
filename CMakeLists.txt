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

add_executable(nwchem-sort-trajectory
  nwchem-sort-trajectory.cpp
)
target_link_libraries(nwchem-sort-trajectory adios2::adios2 MPI::MPI_C)

add_executable(testStream
  testStream.cpp
)
target_link_libraries(testStream adios2::adios2 MPI::MPI_C)

