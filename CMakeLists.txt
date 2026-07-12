cmake_minimum_required(VERSION 3.24)

project(tsp_qlearning_sa LANGUAGES CXX)
include(CheckLanguage)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TSP_ENABLE_OPENMP "Enable OpenMP parallel algorithms" ON)
option(TSP_REQUIRE_OPENMP "Fail configuration when requested OpenMP support is unavailable" OFF)
option(TSP_ENABLE_CUDA "Enable CUDA parallel algorithms" ON)
option(TSP_REQUIRE_CUDA "Fail configuration when requested CUDA support is unavailable" OFF)
# MPI + OpenMP hybrid backend is an optional engineering extension. It is ON by
# default, but gracefully disabled when MPI is not installed, so it never affects
# the default OpenMP/CUDA build.
option(TSP_ENABLE_MPI "Enable MPI backend" ON)

# CMake 3.24 introduced the `native` CUDA architecture value. It avoids baking
# the former fixed sm_70 target into every local build. For a distributable or
# CI artifact, pin this to the intended device set (for example `86;89`).
set(TSP_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architectures: native, all-major, all, or a semicolon-separated list such as 86;89")
set_property(CACHE TSP_CUDA_ARCHITECTURES PROPERTY STRINGS native all-major all)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(TSP_REQUIRE_OPENMP AND NOT TSP_ENABLE_OPENMP)
    message(FATAL_ERROR "TSP_REQUIRE_OPENMP=ON requires TSP_ENABLE_OPENMP=ON.")
endif()

if(TSP_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP enabled")
    elseif(TSP_REQUIRE_OPENMP)
        message(FATAL_ERROR
            "TSP_REQUIRE_OPENMP=ON, but no usable C++ OpenMP runtime was found. "
            "Install/configure the compiler OpenMP support, or configure with -DTSP_REQUIRE_OPENMP=OFF.")
    else()
        message(WARNING
            "OpenMP was requested but is unavailable; parallel mode will fall back to serial multi-chain execution. "
            "Set -DTSP_REQUIRE_OPENMP=ON to make this a configuration error.")
    endif()
else()
    message(STATUS "OpenMP disabled (TSP_ENABLE_OPENMP=OFF)")
endif()

set(TSP_CUDA_SOURCES "")
set(TSP_CUDA_ENABLED OFF)
if(TSP_REQUIRE_CUDA AND NOT TSP_ENABLE_CUDA)
    message(FATAL_ERROR "TSP_REQUIRE_CUDA=ON requires TSP_ENABLE_CUDA=ON.")
endif()

if(TSP_ENABLE_CUDA)
    # Let check_language probe both Ninja and Visual Studio generators. A VS
    # CUDA preset therefore uses the installed CUDA VS toolset when available
    # instead of silently forcing a CPU fallback.
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER AND NOT CMAKE_CUDA_COMPILER MATCHES "-NOTFOUND$")
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)

        if(TSP_CUDA_ARCHITECTURES STREQUAL "" OR TSP_CUDA_ARCHITECTURES STREQUAL "OFF")
            message(FATAL_ERROR
                "TSP_CUDA_ARCHITECTURES must name at least one CUDA architecture when CUDA is enabled. "
                "Use native, all-major, all, or a list such as 86;89.")
        endif()
        # FORCE also replaces the stale sm_70 value in existing build caches.
        set(CMAKE_CUDA_ARCHITECTURES "${TSP_CUDA_ARCHITECTURES}" CACHE STRING
            "CUDA architectures selected by TSP_CUDA_ARCHITECTURES" FORCE)
        set(TSP_CUDA_SOURCES src/cuda_kernels.cu)
        set(TSP_CUDA_ENABLED ON)
        message(STATUS "CUDA enabled (${CMAKE_CUDA_COMPILER}; architectures: ${CMAKE_CUDA_ARCHITECTURES})")
    elseif(TSP_REQUIRE_CUDA)
        message(FATAL_ERROR
            "TSP_REQUIRE_CUDA=ON, but CMake could not enable a CUDA compiler for the ${CMAKE_GENERATOR} generator. "
            "Install the CUDA toolkit/toolset or configure with -DTSP_REQUIRE_CUDA=OFF.")
    else()
        message(WARNING
            "CUDA was requested but no CUDA compiler is available; CUDA mode will fall back to serial multi-chain execution. "
            "Set -DTSP_REQUIRE_CUDA=ON to make this a configuration error.")
    endif()
else()
    message(STATUS "CUDA disabled (TSP_ENABLE_CUDA=OFF)")
endif()

add_library(tsp_core
    src/tsplib_parser.cpp
    src/distance_matrix.cpp
    src/tour.cpp
    src/rng.cpp
    src/timer.cpp
    src/sa.cpp
    src/qlsa.cpp
    src/island.cpp
    src/parallel.cpp
    src/cuda.cpp
    src/metrics.cpp
    ${TSP_CUDA_SOURCES}
)

target_include_directories(tsp_core PUBLIC include)

if(TSP_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(tsp_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(tsp_core PUBLIC TSP_HAS_OPENMP=1)
endif()

if(TSP_CUDA_ENABLED)
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
