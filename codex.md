# Codex 项目交接记录

> 说明：旧版 `codex.md` 在本地清理前未被 Git 跟踪，当前仓库中没有可直接 `git restore` 的历史版本。本文件依据当前工作区、最终报告、结果索引和最近执行记录重建为完整交接版，用于后续继续维护项目。

## 0. 当前状态总览

- 项目路径：`E:\OneDrive\MOSS\4_c_er\学习记录\Proj\parallel-algorithm`
- 当前分支：`main`
- `feature/hpc-aggressive-mpi-cuda` 状态：已确认被 `main` 包含，并已删除本地分支。
- 当前保留报告入口：课程提交版。
- 当前保留提交包：`submission/course/`。
- 当前核心结论：
  - OpenMP multi-chain 是主性能结论。
  - QLSA 在部分 hard instances 上提升解质量，但不能写成总是优于 SA。
  - CUDA chain / candidate / QLSA candidate / parallel reversal 已完成工程验证，但不能写成优于 OpenMP。
  - MPI + OpenMP 已完成真实双 VM `mpirun` 证据链，但不能写成生产 HPC benchmark。
  - 大实例实验用于工程可扩展性验证，不用于声称所有实例达到 BKS。

## 1. 当前保留入口

### 1.1 报告与提交包

- 课程报告 Markdown：`docs/final/final_report_course.md`
- 课程报告 PDF：`docs/final/final_report_course.pdf`
- 已知限制：`docs/final/known_limitations.md`
- 复现命令：`docs/final/reproduction_commands.md`
- 课程提交包：`submission/course/`
- 提交包说明：`submission/README.md`

### 1.2 关键结果入口

- 关键结果索引：`results/final/RESULTS_INDEX.md`
- 论文参考表：`results/reference/paper_table8_runtime.csv`
- 论文 hard-instance quality：`results/reference/paper_hard_instance_quality.csv`
- 最终汇总结果：`results/final/final_key_results.csv`
- 论文/本项目对比：`results/final/report_comparison_summary.csv`
- 大实例下载状态：`results/final/large_instance_download_status.csv`
- 大实例 inventory：`results/final/large_instance_inventory.csv`

### 1.3 最终图表目录

- 最终报告图表统一保留在：`figures/final/`
- 历史图、临时图、archive 图已清理。

## 2. 工程模块状态

### 2.1 核心 C++ 模块

保留目录：

- `include/tsp/`
- `src/`
- `tests/`
- `cmake/`
- `configs/`

核心能力：

- TSPLIB95 parser；
- 一维连续 `DistanceMatrix`；
- `Tour` 表示、nearest-neighbor 初始化、合法性检查；
- 2-opt O(1) delta；
- 串行 SA；
- 串行 QLSA；
- OpenMP multi-chain；
- CUDA chain mode；
- CUDA candidate-level 2-opt evaluation；
- CUDA QLSA candidate mode；
- CUDA parallel reversal；
- MPI + OpenMP hybrid backend；
- Python faithful reference baseline。

### 2.2 CMake 状态

- 支持 C++20；
- 支持 OpenMP；
- 支持 CUDA；
- 支持可选 MPI；
- Windows 本机没有 MPI 时，CMake 会 graceful disable MPI target，不影响 CUDA/OpenMP 主程序。

推荐构建命令：

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake -S . -B build-cuda-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DTSP_ENABLE_CUDA=ON -DTSP_ENABLE_MPI=ON"
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake --build build-cuda-ninja -j"
ctest --test-dir build-cuda-ninja --output-on-failure
```

最近一次重新构建和测试结果：

- CUDA compiler：nvcc 12.9.41；
- MSVC：19.44；
- CUDA enabled；
- OpenMP enabled；
- MPI not found on Windows host，hybrid backend disabled gracefully；
- `ctest`：6/6 passed。

测试包括：

- `test_small_instance`
- `test_parallel`
- `test_cuda`
- `test_cuda_candidate`
- `test_cuda_qlsa_candidate`
- `test_qlsa_paper_lite`

测试后已删除临时 `build-cuda-ninja/`，保持项目结构干净。

## 3. 已完成阶段记录

### Step 1：C++ 串行 SA 基线

完成内容：

- C++20 + CMake 工程骨架；
- TSPLIB95 parser；
- DistanceMatrix；
- Tour；
- SA 2-opt；
- 基础测试；
- baseline 脚本和 README。

关键原则：

- 随机算法支持 seed；
- 2-opt 使用 O(1) delta；
- 距离矩阵一维连续存储；
- 不引入 Boost/Eigen/Concorde。

### Step 2：串行 QLSA

完成内容：

- `QLSAParams` / `QLSAResult`；
- Q table；
- epsilon-greedy / softmax；
- 状态/动作离散化；
- reward 设计；
- CLI `--qlsa`；
- QLSA 测试。

限制：

- C++ 主线是论文思想的工程化变体，不等同完整 SB-QLSA。

### Step 3：OpenMP multi-chain

完成内容：

- chain-level OpenMP；
- 每条 chain 独立 RNG、tour、Q table；
- 只读共享 DistanceMatrix；
- chain results 串行归约；
- CLI 支持 `--parallel omp`、`--chains`、`--threads`；
- CSV 字段加入 `algorithm`、`parallel`、`chains`、`threads`。

主结论：

- OpenMP 是当前最稳定的性能加速后端。

### Step 4：CUDA chain mode

完成内容：

- CUDA 后端；
- distance matrix 拷贝到 GPU；
- one block / thread-level chain mode；
- CUDA SA / QLSA smoke；
- CUDA build 使用 Ninja；
- CUDA 不作为主性能结论。

### Step 5：默认参数实验与自动化

完成内容：

- `scripts/run_step5_experiments.py`；
- `scripts/analyze_results.py`；
- berlin52 手动结果归档；
- 多实例 CPU/OpenMP 默认实验；
- CSV/Markdown 分析。

关键结论：

- SA OpenMP 平均 speedup 约 5.46x；
- QLSA OpenMP 平均 speedup 约 4.98x；
- OpenMP 是最终报告主性能证据。

### Step 6A：参数调优

完成内容：

- `scripts/tune_params.py`；
- `scripts/analyze_tuning.py`；
- OpenMP scaling grid；
- 调参搜索。

结论边界：

- 调参搜索结果不能直接当作独立验证结论。

### Step 6B：调优参数独立验证

完成内容：

- `configs/tuned_params.json`；
- `scripts/run_tuned_validation.py`；
- `scripts/analyze_tuned_validation.py`；
- 独立 seed=101，repeat=10。

主要观察：

- eil76：SA tuned 最小 Gap 可到 0%；
- rat99：QLSA quality-first 优于 SA tuned，但独立验证最小 Gap 为 0.083%；
- eil101：没有稳定复现 Step 6A 的 BKS。

### Step 6C：定向增强实验

完成内容：

- `configs/targeted_quality_configs.json`；
- `scripts/run_targeted_quality.py`；
- `scripts/analyze_targeted_quality.py`；
- repeat=5 targeted high-budget。

主要观察：

- eil101：SA/QLSA targeted 均达到 BKS；
- rat99：QLSA targeted 达到 BKS=1211，SA high-budget 最好为 1212；
- QLSA 在 rat99 上形成明确质量案例，但不能推广为总是优于 SA。

### Step 6D：最终实验汇总

完成内容：

- `docs/final_experiment_summary.md` 曾用于汇总；
- `results/final_key_results.csv`；
- 最终主结论整理。

当前清理后：

- 历史 `docs/archive/`、旧版本汇总文档已删除；
- 最终结论保留在 `docs/final/final_report_course.md` 和 `results/final/RESULTS_INDEX.md`。

### Step 7/8：最终报告重构

完成内容：

- 多轮报告重写；
- 图表统一；
- 课程提交版与公开版曾分离；
- 最后按用户要求只保留课程提交版。

当前保留：

- `docs/final/final_report_course.md`
- `docs/final/final_report_course.pdf`

### MPI + OpenMP 双 VM

完成内容：

- `include/tsp/mpi_parallel.hpp`
- `src/mpi_parallel.cpp`
- `src/mpi_main.cpp`
- `scripts/run_mpi_smoke.py`
- `scripts/run_mpi_vm_scaling.py`
- `scripts/run_large_mpi_vm.py`
- `scripts/analyze_large_mpi_vm.py`
- 两台 Ubuntu VM 上真实 `mpirun` smoke/formal/large quick。

报告边界：

- 只能写真实 VM distributed-memory evidence；
- 不能写生产 HPC benchmark；
- 不写 VM IP、用户名、密码、key path。

### CUDA candidate / QLSA candidate / parallel reversal

完成内容：

- `--cuda_mode chain|candidate`
- `--cuda_candidates_per_iter`
- `--cuda_reversal_mode serial|parallel`
- SA CUDA candidate；
- QLSA CUDA candidate；
- block-level candidate reduction；
- parallel reversal；
- tests and experiments。

报告边界：

- candidate mode 是 batch proposal 变体；
- 不替代默认 chain mode；
- 不声称 CUDA 优于 OpenMP。

### 大实例工程压力测试

完成内容：

- `configs/large_tsplib_instances.json`
- `scripts/download_large_tsplib_subset.py`
- `scripts/prepare_large_tsplib.py`
- `scripts/estimate_large_instance_cost.py`
- `scripts/run_large_openmp.py`
- `scripts/analyze_large_openmp.py`
- `scripts/run_large_cuda.py`
- `scripts/analyze_large_cuda.py`
- `scripts/run_large_mpi_vm.py`
- `scripts/analyze_large_mpi_vm.py`

数据状态：

- L1 全部 8 个实例已下载；
- L2 全部 10 个实例已下载；
- L3 中 dsj1000、u1060、vm1084 已下载；
- pr1002、si1032 当前镜像缺失；
- 下载 SHA256 和 edge weight type 在 `results/final/large_instance_download_status.csv`。

实验边界：

- 可写“百万迭代级中等规模 TSPLIB95 实验跑通”；
- 不能写“百万城市级实例跑通”。

## 4. 当前最终结果文件

### 4.1 final

- `results/final/RESULTS_INDEX.md`
- `results/final/final_key_results.csv`
- `results/final/report_comparison_summary.csv`
- `results/final/large_instance_download_status.csv`
- `results/final/large_instance_inventory.csv`

### 4.2 summary

保留的关键 summary 包括：

- `results/summary/step5_multi_cpu_summary.csv`
- `results/summary/tuned_validation_summary.csv`
- `results/summary/targeted_quality_summary.csv`
- `results/summary/policy_comparison_summary.csv`
- `results/summary/openmp_scaling_final_summary.csv`
- `results/summary/openmp_scaling_large_summary.csv`
- `results/summary/cuda_qlsa_candidate_summary.csv`
- `results/summary/cuda_reversal_summary.csv`
- `results/summary/cuda_candidate_sweep_aggressive_summary.csv`
- `results/summary/large_openmp_l1_summary.csv`
- `results/summary/large_openmp_l2_formal_summary.csv`
- `results/summary/large_openmp_l3_quick_summary.csv`
- `results/summary/large_cuda_formal_summary.csv`
- `results/summary/mpi_vm_smoke_summary.csv`
- `results/summary/mpi_vm_scaling_formal_summary.csv`
- `results/summary/large_mpi_vm_formal_aggressive_summary.csv`
- `results/summary/python_reference_summary.csv`

### 4.3 reference

- `results/reference/paper_table8_runtime.csv`
- `results/reference/paper_hard_instance_quality.csv`

## 5. 当前最终图表

课程报告引用的主要图表在 `figures/final/`：

- `fig01_architecture_pipeline.png`
- `fig02_openmp_speedup.png`
- `fig03_openmp_efficiency.png`
- `fig04_default_gap.png`
- `fig05_tuning_curve.png`
- `fig06_policy_comparison.png`
- `fig07_cuda_positioning.png`
- `fig08_paper_runtime_comparison.png`
- `fig09_paper_quality_comparison.png`
- `fig13_mpi_vm_scaling_formal.png`
- `fig14_hpc_hybrid_architecture.png`
- `fig15_cuda_candidate_mode.png`
- `fig16_large_openmp_gap_time.png`
- `fig17_large_cuda_chain_vs_candidate.png`
- `fig18_large_mpi_vm_scaling.png`
- `fig19_openmp_large_scaling.png`
- `fig21_cuda_profiling_breakdown.png`
- `fig21_cuda_qlsa_candidate.png`
- `fig22_cuda_parallel_reversal.png`
- `fig23_cuda_candidate_sweep_tradeoff.png`

## 6. 最近执行的检查

### 6.1 已通过

```powershell
py scripts\check_report_assets.py docs\final\final_report_course.md
py scripts\check_privacy_and_encoding.py
py scripts\check_final_submission.py
py scripts\check_report_format.py docs\final\final_report_course.md
```

结果：

- report asset check passed；
- privacy and UTF-8/mojibake checks passed；
- course submission package check passed；
- report format check passed with 9 warnings。

warnings：

- MPI 结果表有 8 列，偏宽；不影响 Markdown 渲染。

### 6.2 CTest

最近重新构建并执行：

```powershell
ctest --test-dir build-cuda-ninja --output-on-failure
```

结果：

- 6/6 tests passed；
- 测试后已删除临时 `build-cuda-ninja/`。

如需重新验证，需要先重新构建。

## 7. 当前 `.gitignore` 策略

已忽略：

- build 输出；
- Python cache；
- logs/traces/archive；
- data 下本地 TSPLIB `.tsp` 文件；
- VM SSH key；
- profiler 原始输出；
- browser/GPT 临时材料；
- 私有课程提交包 `submission/course/`。

注意：

- `submission/course/` 适合课程提交，但被 `.gitignore` 忽略，防止误推到公开仓库。
- 如果需要把课程提交包放入私有仓库，需要 `git add -f submission/course submission/README.md`。

## 8. 敏感文件状态

根目录仍存在：

- `.ssh_mpi_vm_key`
- `.ssh_mpi_vm_key.pub`

这两个文件：

- 已被 `.gitignore` 命中；
- 当前 ACL 阻止 Codex 删除；
- 所有者显示为 `MOSS\CodexSandboxOffline`；
- 若要彻底删除，需要用户用管理员权限或手动调整 ACL。

禁止提交：

- `.ssh_mpi_vm_key`
- `.ssh_mpi_vm_key.pub`
- `mpi_hosts.local`
- VM IP、用户名、密码、hostfile。

## 9. 当前报告结论边界

详见 `docs/final/known_limitations.md`。

必须避免：

- CUDA 比 OpenMP 更快；
- QLSA 总是优于 SA；
- C++ 完整复刻论文 SB-QLSA；
- MPI VM 等同生产 HPC benchmark；
- 论文时间对比是同平台公平 benchmark；
- 百万城市规模；
- 所有大实例达到 BKS。

可以安全写：

- OpenMP multi-chain 是主性能结论；
- CUDA candidate-level evaluation 已实现并可运行；
- QLSA candidate CUDA 路径已接入；
- parallel reversal 已实现并测试；
- MPI + OpenMP 双 VM `mpirun` 真实跑通；
- 大实例 L1/L2/L3 quick/formal subset 提供工程可扩展性证据；
- rat99 QLSA high-budget 达到 BKS，而 SA high-budget 最好为 1212；
- eil101 targeted 中 SA/QLSA 均达到 BKS。

## 10. 当前建议提交方式

不要使用：

```powershell
git add .
```

建议按类别添加：

```powershell
git add .gitignore CMakeLists.txt README.md
git add include src tests scripts configs cmake python_ref
git add docs/final/final_report_course.md docs/final/final_report_course.pdf docs/final/known_limitations.md docs/final/reproduction_commands.md
git add figures/final
git add results/final results/summary results/reference
```

如果课程提交包需要进入私有仓库：

```powershell
git add -f submission/course submission/README.md
```

谨慎提交：

- `results/raw/`：体量较大，但可作为 evidence；
- `data/*.tsp`：默认忽略，除非课程要求完整数据包；
- `codex.md`：工作交接文件，适合本地持续维护，可按需要提交。

禁止提交：

- `.ssh_mpi_vm_key*`
- `build*/`
- `results/logs/`
- `results/traces/`
- VM IP、用户名、密码、hostfile；
- browser/GPT 临时材料；
- Nsight 原始大 report，除非课程明确要求。

## 11. 后续可做事项

如果还要继续增强：

1. 重新导出 `docs/final/final_report_course.pdf`，确保 PDF 与 Markdown 最新内容完全一致。
2. 修正报告中 MPI 表格过宽 warning。
3. 手动删除 `.ssh_mpi_vm_key*`，或调整 ACL 后删除。
4. 若需要公开仓库版本，重新生成脱敏 public README/report，而不是直接使用 course 报告。
5. 若需要补充论文级严谨性，可增加 softmax 机制对照和统计检验，但不要新增无证据结论。

## 12. 最后一次清理后的项目形态

保留：

- 源码；
- tests；
- scripts；
- configs；
- final docs；
- final figures；
- final/summary/reference results；
- course submission package；
- Python reference；
- TSPLIB 本地数据。

删除：

- 历史报告版本；
- docs/dev 与 docs/archive 过程材料；
- figures/archive；
- results/logs/archive/traces；
- old submission/public/legacy；
- build 目录；
- 临时同步包；
- GPT/browser/screenshot 中间材料。

项目当前定位：课程提交优先、工程证据完整、公开仓库需另行脱敏。
