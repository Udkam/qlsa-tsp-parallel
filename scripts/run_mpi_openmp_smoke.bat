@echo off
REM MPI + OpenMP hybrid backend smoke test.
REM This machine has no MPI installed, so the script detects MPI first and only
REM runs when it is available; otherwise it reports gracefully without failing
REM the rest of the pipeline. See docs/dev/mpi_openmp_design.md.

where mpiexec >nul 2>&1
if errorlevel 1 (
    echo [info] mpiexec not found; MPI hybrid backend is design-only on this machine.
    echo [info] See docs/dev/mpi_openmp_design.md for the rank-level design.
    exit /b 0
)

echo [info] mpiexec found. Configure with -DTSP_ENABLE_MPI=ON and build tsp_sa_mpi first.
if not exist build-cuda-ninja\tsp_sa_mpi.exe (
    echo [info] tsp_sa_mpi.exe not built yet; build the MPI target then re-run this script.
    exit /b 0
)

mpiexec -n 2 build-cuda-ninja\tsp_sa_mpi.exe --input tests\fixtures\square4.tsp --parallel mpi-omp --chains 8 --threads 2
