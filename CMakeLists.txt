cmake_minimum_required(VERSION 3.16)

project(tsp_qlearning_sa LANGUAGES CXX)
include(CheckLanguage)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TSP_ENABLE_OPENMP "Enable OpenMP parallel algorithms" ON)
option(TSP_ENABLE_CUDA "Enable CUDA parallel algorithms" ON)
# MPI + OpenMP hybrid backend is an optional engineering extension. It is ON by
# default, but gracefully disabled when MPI is not installed, so it never affects
# the default OpenMP/CUDA build.
option(TSP_ENABLE_MPI "Enable MPI backend" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(TSP_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
endif()

set(TSP_CUDA_SOURCES "")
if(TSP_ENABLE_CUDA)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        unset(CMAKE_CUDA_COMPILER CACHE)
        message(STATUS "CUDA language skipped for Visual Studio generator unless CUDA VS toolset is installed; cuda mode will fall back to serial multi-chain execution")
    else()
        check_language(CUDA)
    endif()
    if(CMAKE_CUDA_COMPILER AND NOT CMAKE_GENERATOR MATCHES "Visual Studio")
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)
        set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures")
        set(TSP_CUDA_SOURCES src/cuda_kernels.cu)
        message(STATUS "CUDA enabled")
    elseif(NOT CMAKE_GENERATOR MATCHES "Visual Studio")
        message(STATUS "CUDA compiler not found; cuda mode will fall back to serial multi-chain execution")
    endif()
endif()

add_library(tsp_core
    src/tsplib_parser.cpp
    src/distance_matrix.cpp
    src/tour.cpp
    src/rng.cpp
    src/timer.cpp
    src/sa.cpp
    src/qlsa.cpp
    src/parallel.cpp
    src/cuda.cpp
    src/metrics.cpp
    ${TSP_CUDA_SOURCES}
)

target_include_directories(tsp_core PUBLIC include)

if(TSP_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(tsp_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(tsp_core PUBLIC TSP_HAS_OPENMP=1)
    message(STATUS "OpenMP enabled")
else()
    message(STATUS "OpenMP not enabled or not found; parallel mode will fall back to serial multi-chain execution")
endif()

if(TSP_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    target_compile_definitions(tsp_core PUBLIC TSP_HAS_CUDA=1)
    set_source_files_properties(src/cuda_kernels.cu PROPERTIES LANGUAGE CUDA)
    target_compile_options(tsp_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>:-O3;-DNDEBUG>>
    )
endif()

if(MSVC)
    target_compile_options(tsp_core PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2 /DNDEBUG>)
else()
    target_compile_options(tsp_core PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -march=native -DNDEBUG>)
endif()

set(TSP_EXTRA_LINK_LIBS "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    list(APPEND TSP_EXTRA_LINK_LIBS stdc++fs)
endif()

if(TSP_ENABLE_MPI)
    find_package(MPI QUIET)
    if(MPI_CXX_FOUND)
        message(STATUS "MPI enabled")
    else()
        message(STATUS "TSP_ENABLE_MPI=ON but MPI was not found; hybrid backend disabled gracefully")
    endif()
endif()

add_executable(tsp_sa src/main.cpp)
target_link_libraries(tsp_sa PRIVATE tsp_core ${TSP_EXTRA_LINK_LIBS})

if(TSP_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(tsp_sa PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(tsp_sa PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
else()
    target_compile_options(tsp_sa PRIVATE $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>)
endif()

if(TSP_ENABLE_MPI AND MPI_CXX_FOUND)
    add_executable(tsp_sa_mpi
        src/mpi_main.cpp
        src/mpi_parallel.cpp
    )
    target_link_libraries(tsp_sa_mpi PRIVATE tsp_core MPI::MPI_CXX ${TSP_EXTRA_LINK_LIBS})
    target_compile_definitions(tsp_sa_mpi PRIVATE TSP_HAS_MPI=1)

    if(TSP_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(tsp_sa_mpi PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(MSVC)
        target_compile_options(tsp_sa_mpi PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
    else()
        target_compile_options(tsp_sa_mpi PRIVATE $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>)
    endif()
endif()

enable_testing()
set_property(DIRECTORY PROPERTY TEST_INCLUDE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CTestSetDefaultConfig.cmake")
add_subdirectory(tests)
