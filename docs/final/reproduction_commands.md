# 复现命令

以下命令均在项目根目录执行。Windows 环境建议使用 `py` 启动 Python 脚本；Ubuntu VM 中使用 `python3`。报告仅使用 VM1/VM2 代称，本地 IP、用户名和 SSH 细节不进入提交包。

## 构建与测试

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake -S . -B build-cuda-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DTSP_ENABLE_CUDA=ON -DTSP_ENABLE_MPI=ON"
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake --build build-cuda-ninja -j"
ctest --test-dir build-cuda-ninja --output-on-failure
```

## 默认参数 OpenMP 实验

```powershell
py scripts\run_step5_experiments.py --instances berlin52 eil51 st70 eil76 rat99 eil101 --iterations 1000000 --repeat 3 --chains 32 --threads 8 --no-cuda --output results\raw\step5_multi_cpu_raw.csv
```

## 调优与定向增强

```powershell
scripts\run_tuned_validation.bat
scripts\run_targeted_quality.bat
```

## QLSA 论文机制对齐变体

```powershell
.\build-cuda-ninja\tsp_sa.exe --qlsa --qlsa_variant paper --input tests\fixtures\square4.tsp --iterations 500 --seed 1 --init random --csv-only
.\build-cuda-ninja\tsp_sa.exe --qlsa --qlsa_variant paper-sb --input data\berlin52.tsp --iterations 100000 --seed 1 --init nn --csv-only
py scripts\run_qlsa_variant_experiments.py --instances berlin52 eil76 rat99 eil101 --iterations 300000 --repeat 3 --chains 32 --threads 8 --output results\raw\qlsa_variant_alignment_raw.csv
py scripts\analyze_qlsa_variant_experiments.py --input results\raw\qlsa_variant_alignment_raw.csv --output results\summary\qlsa_variant_alignment_summary.csv --markdown docs\dev\qlsa_variant_alignment_analysis.md --figure figures\fig_qlsa_variant_alignment.png
```

## CUDA candidate / QLSA candidate

```powershell
py scripts\run_cuda_candidate_experiments.py --instances berlin52 eil101 ch130 a280 --algorithms sa qlsa --iterations 500000 --repeat 3 --chains 64 --block-sizes 128 --candidates-per-iter 128 --reversal-modes serial parallel --output results\raw\cuda_qlsa_candidate_raw.csv
py scripts\analyze_cuda_qlsa_candidate.py --input results\raw\cuda_qlsa_candidate_raw.csv --output results\summary\cuda_qlsa_candidate_summary.csv --markdown docs\final\cuda_qlsa_candidate_analysis.md --figure figures\fig21_cuda_qlsa_candidate.png
py scripts\analyze_cuda_reversal.py --input results\raw\cuda_qlsa_candidate_raw.csv --output results\summary\cuda_reversal_summary.csv --markdown docs\final\cuda_reversal_analysis.md --figure figures\fig22_cuda_parallel_reversal.png
```

CUDA 候选策略 quick 对照：

```powershell
py scripts\run_cuda_candidate_experiments.py --instances berlin52 a280 --algorithms sa qlsa --iterations 100000 --repeat 1 --chains 32 --block-sizes 128 --candidates-per-iter 128 --reversal-modes parallel --candidate-policies best random hybrid --output results\raw\cuda_candidate_hybrid_quick_raw.csv
py scripts\analyze_cuda_candidate.py --input results\raw\cuda_candidate_hybrid_quick_raw.csv --output results\summary\cuda_candidate_hybrid_quick_summary.csv --markdown docs\dev\cuda_candidate_hybrid_quick_analysis.md --figure figures\fig_cuda_candidate_hybrid_quick.png
```

Nsight 检测入口：

```powershell
py scripts\run_cuda_nsight_profile.py --instance a280 --iterations 100000 --policy hybrid --ncu-set basic --markdown docs\dev\cuda_nsight_profile_deep_analysis.md
```

如果 Nsight Systems 在中文项目路径下报路径编码错误，可将 `tsp_sa.exe` 与 `a280.tsp` 临时复制到 ASCII 路径后运行同一条 CUDA 命令；本报告使用该方式取得 Systems 记录。Nsight Compute 需要当前用户具备 NVIDIA GPU performance counters 权限。

## 大实例 OpenMP / CUDA

```powershell
py scripts\run_large_openmp.py --tier L1 --iterations 1000000 --repeat 3 --chains 64 --threads 8 --output results\raw\large_openmp_l1_raw.csv
py scripts\analyze_large_openmp.py --input results\raw\large_openmp_l1_raw.csv --output results\summary\large_openmp_l1_summary.csv

py scripts\run_large_cuda.py --instances ch130 a280 lin318 rat575 --iterations 500000 --repeat 3 --chains 64 --block-size 128 --candidates-per-iter 128 --output results\raw\large_cuda_formal_raw.csv
py scripts\analyze_large_cuda.py --input results\raw\large_cuda_formal_raw.csv --output results\summary\large_cuda_formal_summary.csv
```

## MPI + OpenMP VM 实验

在 VM1 上执行：

```bash
cd ~/parallel-algorithm
export PATH=$HOME/ompi-4.1.2/bin:$PATH
export LD_LIBRARY_PATH=$HOME/ompi-4.1.2/lib:${LD_LIBRARY_PATH:-}
python3.8 scripts/run_large_mpi_vm.py --instances ch130 a280 --np 1 2 --threads 2 4 --chains 64 --iterations 300000 --repeat 3 --hostfile mpi_hosts.local --output results/raw/large_mpi_vm_formal_aggressive_raw.csv --summary results/summary/large_mpi_vm_formal_aggressive_summary.csv --timeout 600
```

Windows 侧分析：

```powershell
py scripts\analyze_large_mpi_vm.py --input results\raw\large_mpi_vm_formal_aggressive_raw.csv --output results\summary\large_mpi_vm_formal_aggressive_summary.csv --markdown docs\final\large_mpi_vm_analysis.md --figure figures\fig18_large_mpi_vm_scaling.png
```

## 报告检查

```powershell
py scripts\check_report_format.py docs\final\final_report_course.md
py scripts\check_report_assets.py
py scripts\check_privacy_and_encoding.py
```
