# CUDA Nsight profiling analysis

- Instance: `a280`
- Iterations: `20000`
- Candidate policy: `hybrid`
- Executable: `E:\OneDrive\MOSS\4_c_er\学习记录\Proj\parallel-algorithm\build-cuda-ninja\tsp_sa.exe`

- Nsight Systems: `not found`
- Nsight Compute: `C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\ncu.BAT`

## Nsight Systems

- Tool not found in PATH; no Systems trace was captured.

## Nsight Compute

- Command: `C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\ncu.BAT --force-overwrite --set speedOfLight --target-processes all --export E:\OneDrive\MOSS\4_c_er\学习记录\Proj\parallel-algorithm\results\logs\nsight\cuda_candidate_a280_hybrid_ncu E:\OneDrive\MOSS\4_c_er\学习记录\Proj\parallel-algorithm\build-cuda-ninja\tsp_sa.exe --input E:\OneDrive\MOSS\4_c_er\学习记录\Proj\parallel-algorithm\data\a280.tsp --parallel cuda --cuda_mode candidate --cuda_candidate_policy hybrid --cuda_reversal_mode parallel --cuda_candidates_per_iter 128 --chains 64 --cuda_block_size 128 --iterations 20000 --seed 1 --init nn --csv-only`
- Exit code: `0`
- Output prefix: `E:\OneDrive\MOSS\4_c_er\学习记录\Proj\parallel-algorithm\results\logs\nsight\cuda_candidate_a280_hybrid_ncu`

## Interpretation boundary

- This file records profiler availability and generated reports.
- If metrics are missing or the tool is unavailable, do not report occupancy, bandwidth, or CUDA advantage claims.
- Runtime and quality conclusions should still come from CSV experiments.
