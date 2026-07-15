# Codex 工作记录

本文件记录 `parallel-algorithm` 项目的阶段性处理结果，便于后续继续维护。所有中文内容应保持 UTF-8 编码；如果发现四个连续问号、Unicode replacement character（U+FFFD）、常见 GBK/UTF-8 错配片段或类似乱码，应立即定位并修复，不应继续在损坏文本上追加内容。

## 2026-07-16：核心性能优化与实验合同闭环

- QLSA softmax 权重改为链内工作区复用，近期增量窗口改为定长环形缓冲；OpenMP island 在全部迁移轮次中复用一个并行团队，并以 barrier + single 保持 chunk/迁移边界。
- `ccc0b57` 与 `a749f9a` 使用同一 MSVC 19.44、Release、Ninja/OpenMP 构建合同完成 20 组共同种子 AB/BA 配对。a280 QLSA 组合优化的配对中位加速比为 1.2971，eil76 ring-island 组合优化为 1.5586；全部非计时输出字段逐对一致。
- 普通 CLI runner 统一读取 14/22/27/28 列输出，新实验严格要求 28 列并核对算法标签、seed 序列、初始化、requested/actual backend、fallback、实际线程数、完成迭代和 deadline。
- MPI + OpenMP 新增 19 列结果合同，记录归约后的实际线程数、完成迭代和 deadline；accepted/improved/iterations/deadline/actual_threads 在所有 rank 上生成一致的全局统计。旧 16 列格式保留历史读取入口。
- CUDA 在设备检测与 CPU fallback 前执行同一请求校验，统一限制 block、候选数、QLSA action 数和 `state_window`；TSPLIB 解析器逐项验证坐标节点 ID 的唯一性与完整性。
- 报告图建立 CSV/PNG SHA-256 manifest，QLSA 机制对齐汇总、说明与图从同一 raw CSV 重新生成；提交索引新增核心优化 raw/summary 配对数据。
- 验证结果：86 个 Python 流水线测试、CPU preset 8 个 CTest、真实 CUDA preset 8 个 CTest 全部通过；`test_cuda` 确认 `cuda_available=true`。严格 MPI preset 在本机按设计报告缺少 MPI C++ 工具链，optional MPI 配置路径正常生成。

## 2026-07-16：报告展示性叙述整理

- 本地课程报告源 `docs/final/report.md` 将边界说明改写为“实现条件—数据来源—数据结论”的展示性叙述：集中说明 Hamming/fixed-city0 历史条件，以及 seeded-NN/edge 的工程设计，删除“报告不做什么”“不能说明什么”一类元叙事。
- QLSA 机制对齐图的文字说明明确为已扫描 policy/threshold 网格中的最小偏差；20 个共同种子的 eil76 配对复核承担固定配置下的统计结论，避免把参数扫描中的最佳值表述为稳定优劣。
- 本轮未生成新实验数据，现有图表保持原始数值与数据来源；报告格式、资源引用、UTF-8 与课程提交资料检查均通过。

## 2026-07-13：报告口径与 QLSA 图表同步

- 本地课程报告源 `docs/final/report.md` 与交付 PDF `小组及个人报告.pdf` 已按小组报告的“结论—证据—解释—边界”结构更新，并重新导出为 35 页 A4 PDF 后逐页抽查版式。两者按 `.gitignore` 约定保留在本地，不作为公开仓库文件提交。
- 明确当前实现的正确性边界：2-opt 的常数时间增量仅适用于对称 TSP；解析器和 `DistanceMatrix` 都会拒绝 ATSP 与非对称 `EXPLICIT/FULL_MATRIX`。多链最近邻初始化现由链 seed 派生起点，而单链仍从 city 0 出发。
- 明确区分 `paper-sb` 的两种状态语义：历史正式样本使用论文兼容的 position-Hamming；当前工程还提供以无向回路边差异定义的 `edge`。历史多链 NN 同时使用固定 city 0 起点，因此不把这些归档数值写成新 `edge`/seeded-NN 默认值的性能结论。
- 修正 `scripts/analyze_qlsa_variant_experiments.py` 的汇总键，将 `diversity_metric` 纳入条件；旧 raw CSV 缺少该字段时仅对 `paper-sb` 显式归一为 `hamming`。这样 edge 与 Hamming 新旧样本不会被错误合并。
- 已从同一份历史 raw CSV 重生成 `results/summary/qlsa_variant_alignment_summary.csv`、`docs/dev/qlsa_variant_alignment_analysis.md` 和 `figures/fig_qlsa_variant_alignment.png`。图内标题与图例均标注 `paper-sb: Hamming`，数值未被修改；其余性能图因没有新配对实验而保持历史图表不变。
- 已通过报告格式/资源/UTF-8/提交资料检查、64 个 Python 测试，以及 CUDA+OpenMP Ninja 预设下的 8 个 CTest。新 seed-NN 或 edge 的效果仍需以共同种子、相同预算的完整实验矩阵重新测量后再更新第 7 节全部相关表图。

## 当前报告定位

当前课程报告重写稿：

- `docs/final/final_report_course_rewrite_v2.md`
- `docs/final/personal_report.md`

保留但不覆盖的旧稿：

- `docs/final/final_report_course.md`
- `docs/final/final_report_course_rewrite.md`

报告主线为“基于多搜索链的 SA/QLSA 并行化实现”。OpenMP 是主要性能结论，CUDA 与 MPI 作为工程扩展和边界实验。报告保持以下边界：

- 不写 CUDA 全面优于 OpenMP。
- 不写 QLSA 总是优于 SA。
- 不写 C++ 主线完整复刻论文 State-Based QLSA。
- 不把双虚拟机 MPI 写成生产 HPC 集群评测。
- 不把百万迭代级实验写成百万城市规模。
- 不把参考论文不同语言、不同硬件下的运行时间写成同平台公平比较。

## 本轮图表筛选

已弃用无实验数据支撑的早期结构示意图，课程报告正文只引用有数据来源的可视化图。当前正文使用：

- `fig_course_01_openmp_speedup.png`
- `fig_course_02_openmp_efficiency.png`
- `fig_course_03_default_gap.png`
- `fig_course_04_targeted_quality.png`
- `fig_course_05_policy_comparison.png`
- `fig_course_06_cuda_boundary.png`
- `fig_course_07_mpi_scaling.png`
- `fig_course_08_large_openmp.png`
- `fig_course_09_paper_quality.png`
- `fig_course_10_openmp_thread_scaling.png`

处理结果：

- 最后一张论文质量对比图已改为蓝、橙、绿配色，不再以灰色作为主要对比色。
- 新增 OpenMP 线程扩展曲线，用于展示 1、2、4、8、12、16 线程下的加速趋势。
- 报告中不再使用“图 x：标题”或“表 x：标题”的独立图表标题样式，改为在正文中自然引入图表，再在图后分析。
- 正文已移除不必要的课程目标宽表，课程要求映射改为段落说明。

## 本轮报告调整

`docs/final/final_report_course_rewrite_v2.md` 已按以下原则重写：

- 正文以中文为主，仅保留 TSP、SA、QLSA、OpenMP、CUDA、MPI、BKS、2-opt、Q-learning、Metropolis 等必要术语。
- 公式使用 `$$...$$` 包裹，保留可渲染的 LaTeX。
- 第 2 至第 9 节增强叙事，不再按文件清单罗列。
- 第 9 节只保留有代表性的实施问题：TSPLIB95 数据准备、CUDA 构建与性能边界、MPI 双虚拟机环境、QLSA 参数敏感性。
- 个人工作说明已单独放入 `docs/final/personal_report.md`，不再作为主报告附录。

## 实验设计与结果分析增强

针对课程报告中“实验设计说明不足、结果分析偏短”的问题，已进一步扩写 `docs/final/final_report_course_rewrite_v2.md` 的第 6、7 节：

- 第 6 节不再只列实验名称，而是说明默认参数实验、调优验证、定向增强、策略对比、CUDA、MPI、大实例压力测试各自回答什么问题。
- 明确默认实验的控制变量：实例集合、迭代次数、搜索链数、线程数、重复次数、串行多链基准和 OpenMP 对比口径。
- 补充调优验证与定向增强的实验逻辑，说明为什么要使用独立随机种子以及为什么不能只报告调参搜索中的最好值。
- 补充 CUDA 候选批量评价和 MPI + OpenMP 双虚拟机实验的设计目的与结论边界。
- 第 7 节已按图表逐组增强分析，包括 OpenMP 加速和效率、默认解质量、rat99/eil101 定向增强、QLSA 策略敏感性、CUDA 质量/时间边界、MPI 通信开销和大实例压力测试。
- 正文补入关键数据，例如 SA/QLSA 平均加速比和效率、rat99 QLSA high-budget 达到 BKS=1211、SA high-budget 最好为 1212、MPI np=2 加速比和通信时间、L1/L2/L3 大实例运行时间与偏差。
- 所有新增分析继续保持保守表述，不把 CUDA 写成主性能结论，不把双虚拟机 MPI 写成生产 HPC 评测，不把 QLSA 写成普遍优于 SA。

## 检查结果

已执行并通过：

```powershell
py scripts\make_course_report_figures.py
py scripts\check_report_assets.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_report_format.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_privacy_and_encoding.py
py -m py_compile scripts\make_course_report_figures.py scripts\check_report_assets.py scripts\check_report_format.py
```

额外复核：

- 主报告不包含 `fig_v2_` 旧图引用。
- 主报告不包含独立的“图 x：”或“表 x：”标题样式。
- 主报告不包含裸 `frac`、`times100` 或 `{sa_avg_speed}` 一类模板残留。
- 主报告不包含“附录 A 个人工作说明”。
- 主报告不包含常见乱码 token。
- 主报告不再残留“部分”一类用户明确不希望出现的弱表述。
- 临时图表总览 `_course_figures_review.png` 已删除。

## 后续建议

提交前建议人工阅读：

1. `docs/final/final_report_course_rewrite_v2.md`
2. `docs/final/personal_report.md`
3. `figures/final/fig_course_*.png`

如果决定采用 v2 作为最终课程报告，可以再复制为正式提交文件名。不要提交 `.ssh_mpi_vm_key*`、构建目录、浏览器/GPT 临时材料或包含隐私的临时记录。

## 全量数据重跑与代表性分析修订

用户进一步要求“全部实例，无论小还是大”都要跑通，同时正文不能把 38 个实例全部堆进分析。已完成以下调整：

- 本地 OpenMP 后端对 `data/` 目录下 38 个可用 `.tsp` 文件完成统一重跑：
  - 1,000,000 次迭代；
  - 64 条搜索链；
  - 8 个线程；
  - 每个实例、每个算法重复 3 次；
  - 原始结果保存为 `results/raw/final_all_data_openmp_raw.csv`。
- 新增 `scripts/analyze_all_data_openmp.py`，用于从全量重跑结果中抽取报告用代表实例，而不是默认使用全部 38 个实例。
- 报告当前采用 10 个代表实例：
  - `berlin52`
  - `eil76`
  - `rat99`
  - `eil101`
  - `a280`
  - `rat575`
  - `rat783`
  - `dsj1000`
  - `u1060`
  - `vm1084`
- 代表实例汇总保存为 `results/summary/final_representative_openmp_summary.csv`。
- 代表实例分析保存为 `docs/dev/final_representative_openmp_analysis.md`。
- 新图保存为 `figures/final/fig_course_11_representative_openmp.png`。

报告修订：

- `docs/final/final_report_course_rewrite_v2.md` 的大实例压力测试段落已改为“后台全量覆盖 + 正文代表分析”的口径。
- 正文不再使用 `fig_course_08_large_openmp.png`，改用 `fig_course_11_representative_openmp.png`。
- 第 6 节实验设计明确说明不同实验采用不同取舍：默认性能实验用 6 个实例，调优分析用困难实例，CUDA/MPI 用对应后端实例，大实例压力测试用 10 个代表实例。
- 清除了旧的“仍需要调参或增加搜索预算”“不能写成性能优于 OpenMP”“生产 HPC 集群性能”“公平竞赛”等生硬表述。
- 定向增强表格已同步更新为 `results/summary/final_hard_targeted_summary.csv` 中的最新复核结果。

最新检查结果：

```powershell
py scripts\check_report_assets.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_report_format.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_privacy_and_encoding.py
py -m py_compile scripts\analyze_all_data_openmp.py scripts\check_report_assets.py scripts\check_report_format.py
ctest --test-dir build-cuda-ninja --output-on-failure
```

以上命令均已通过。其中 `ctest` 结果为 6/6 通过，包括 CUDA chain、CUDA candidate、CUDA QLSA candidate 和 QLSA paper-lite smoke 测试。

## 图表标题与实例说明修订

用户指出课程报告图表不应出现“重跑”一类过程性表述，图题应只描述数据内容。同时报告需要解释不同实验实例的特征，说明为什么这些实例适合作为困难实例或大实例。

已完成：

- `figures/final/fig_course_11_representative_openmp.png` 已重新生成，图题改为：
  - `OpenMP 代表实例结果（64 条搜索链，8 线程）`
- 图内不再出现“重跑”等过程性词语。
- `scripts/analyze_all_data_openmp.py` 已同步更新，后续重新生成该图时仍使用新版标题。
- `docs/final/final_report_course_rewrite_v2.md` 第 6 节新增实例分组说明：
  - 小规模基准实例：`berlin52`、`eil51`、`st70`
  - 默认困难实例：`eil76`、`rat99`、`eil101`
  - 中大规模实例：`a280`、`rat575`、`rat783`
  - 千点级压力实例：`dsj1000`、`u1060`、`vm1084`
- 这些说明解释了各类实例的用途：正确性/稳定性检查、参数敏感性分析、中大规模运行时间和质量变化观察、千点级工程可扩展性验证。
- 第 6 节实例说明改为 Markdown 列表，避免长段落堆叠，提升渲染可读性。
- 第 7.7 节对应改写为“本组数据覆盖 38 个可用实例，正文选取 10 个代表实例”，不再使用“重跑”字样。

已执行并通过：

```powershell
py scripts\analyze_all_data_openmp.py --input results\raw\final_all_data_openmp_raw.csv --output results\summary\final_representative_openmp_summary.csv --markdown docs\dev\final_representative_openmp_analysis.md --figure figures\final\fig_course_11_representative_openmp.png
py scripts\check_report_assets.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_report_format.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_privacy_and_encoding.py
```

## CUDA candidate policy 增强

用户澄清“CUDA 还有能做的就继续实现”，因此在不改变 OpenMP 主结论和默认 CUDA chain/candidate 行为的前提下，补充了 CUDA candidate mode 的候选选择策略。

本轮实现内容：

- `include/tsp/parallel.hpp`
  - 新增 `CudaCandidatePolicy`：
    - `Best`
    - `Random`
  - 新增 `ParallelParams::cuda_candidate_policy`，默认 `Best`。
- `src/main.cpp`
  - 新增 CLI 参数：
    - `--cuda_candidate_policy best|random`
  - 增加参数校验。
  - `random` 模式下 algorithm 字段输出为 `sa-cuda-candidate-random` 或 `qlsa-cuda-candidate-random`。
- `src/cuda_kernels.cu`
  - SA candidate kernel 增加 policy 分支。
  - QLSA candidate kernel 增加 policy 分支。
  - `best` 继续使用块内最小 delta 归约。
  - `random` 从候选批中按可复现随机方式选择一个候选，然后走相同的 Metropolis 接受与路径反转流程。
- `tests/test_cuda_candidate.cpp`
  - 增加 SA candidate random policy 测试。
- `tests/test_cuda_qlsa_candidate.cpp`
  - 增加 QLSA candidate random policy 测试。
- `scripts/run_cuda_candidate_experiments.py`
  - 新增 `--candidate-policies`。
- `scripts/run_cuda_candidate_sweep.py`
  - 新增 `--candidate-policies`。
- `scripts/analyze_cuda_candidate.py`
  - summary 增加 `cuda_candidate_policy` 字段。
  - 兼容旧 CSV：旧 candidate 数据默认视为 `best`。
- `scripts/analyze_cuda_candidate_sweep.py`
  - summary 增加 `cuda_candidate_policy` 字段。
- 新增开发文档：
  - `docs/dev/cuda_candidate_policy_design.md`
  - `docs/dev/cuda_candidate_policy_analysis.md`
- 新增 quick 结果：
  - `results/raw/cuda_candidate_policy_quick_raw.csv`
  - `results/summary/cuda_candidate_policy_quick_summary.csv`
  - `figures/final/fig_cuda_candidate_policy_quick.png`

quick 结果摘要：

- berlin52 上 SA / QLSA 的 chain、candidate-best、candidate-random 均达到 BKS=7542。
- SA 运行时间：
  - chain：112.747 ms
  - candidate-best：288.305 ms
  - candidate-random：184.229 ms
- QLSA 运行时间：
  - chain：200.121 ms
  - candidate-best：459.231 ms
  - candidate-random：269.192 ms
- 解释：
  - `random` policy 比 `best` policy 更快；
  - 但 candidate 模式仍慢于 chain 模式；
  - 该结果支持“CUDA candidate policy 是工程增强和实验对照维度”，不支持“CUDA 优于 OpenMP”。

报告同步更新：

- `docs/final/final_report_course_rewrite_v2.md`
  - 标题压缩为 `TSP 多搜索链 SA/QLSA 并行优化`。
  - CUDA 设计部分加入 `best/random` candidate policy。
  - CUDA 实验部分加入 berlin52 quick policy 对照。
  - 第 9 节“实施过程中遇到的问题”改为更有代表性的四类问题：
    - 链级并行与细粒度并行的取舍；
    - CUDA 质量收益与时间成本之间的矛盾；
    - MPI 结果必须区分真实分布式运行和本地 fallback；
    - QLSA 参数敏感性。
- `docs/final/known_limitations.md`
  - 增加 candidate policy 说明。
- `README.md`
  - 增加 `--cuda_candidate_policy random` 示例。

验证已通过：

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake --build build-cuda-ninja -j"
ctest --test-dir build-cuda-ninja --output-on-failure
py scripts\run_cuda_candidate_experiments.py --instances berlin52 --algorithms sa qlsa --iterations 100000 --repeat 1 --chains 32 --block-sizes 128 --candidates-per-iter 64 --reversal-modes serial --candidate-policies best random --output results\raw\cuda_candidate_policy_quick_raw.csv
py scripts\analyze_cuda_candidate.py --input results\raw\cuda_candidate_policy_quick_raw.csv --output results\summary\cuda_candidate_policy_quick_summary.csv --markdown docs\dev\cuda_candidate_policy_analysis.md --figure figures\final\fig_cuda_candidate_policy_quick.png
py -m py_compile scripts\run_cuda_candidate_experiments.py scripts\analyze_cuda_candidate.py scripts\run_cuda_candidate_sweep.py scripts\analyze_cuda_candidate_sweep.py scripts\check_report_format.py
py scripts\check_report_assets.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_report_format.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_privacy_and_encoding.py
```

风险与边界：

- `best` policy 是批量择优 proposal，`random` policy 更接近随机候选语义，但仍不完全等同原 SA 的逐步单候选采样。
- quick policy 结果只是功能验证和设计对照，不替代正式 CUDA 大实例实验。
- CUDA 仍不作为主性能结论，OpenMP 仍是报告主加速证据。

## CUDA 可实现方向与报告分点优化

用户询问 CUDA 部分是否还有可实现内容，并指出报告中长段落阅读负担较重，需要更多分点和加粗结构。

本轮先核对了当前 CUDA 实现状态：

- 已实现 CUDA 多链模式。
- 已实现 SA CUDA candidate mode。
- 已实现 QLSA CUDA candidate mode。
- 已实现 `--cuda_reversal_mode serial|parallel`。
- CUDA kernel 中使用共享内存保存候选增量和候选下标，并进行块内归约。
- 测试覆盖 `test_cuda_candidate` 和 `test_cuda_qlsa_candidate`。

因此，当前 CUDA 已经具备较完整的工程扩展内容。继续能做的方向主要是优化型，而不是课程提交前必须补齐的功能：

- 更细的路径反转优化；
- 更接近原始 SA 随机采样的候选选择策略；
- 共享内存路径片段缓存；
- Nsight profiling 驱动的 kernel / memory / synchronization 分析；
- 更大实例上的 CUDA 参数搜索。

报告更新：

- 第 4 节“选择链级并行”的长段落改为两点：
  - 搜索层面；
  - 系统层面。
- 第 4 节“并行效率分析”改为三点：
  - SA 链更均匀；
  - QLSA 链更不均匀；
  - CUDA 瓶颈更复杂。
- 第 7.5 节 CUDA 小节改为：
  - 多链模式；
  - 候选批量评价模式；
  - 并行路径反转；
  - 实验结果；
  - 瓶颈来源；
  - 后续可做的 CUDA 优化方向。
- 第 10 节后续工作改为项目符号列表，新增统计检验作为后续方向。

验证已通过：

```powershell
py scripts\check_report_assets.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_report_format.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_privacy_and_encoding.py
```

## QLSA 机制对齐与 CUDA profiling 深化收尾

时间：2026-06-23 01:10

本阶段把报告中原先列为后续工作的两个方向推进为实际实验：QLSA 论文机制对齐实验，以及 CUDA Nsight profiling 深化。

新增与更新：

- `scripts/run_qlsa_variant_experiments.py`
  - 批量运行 `current`、`paper`、`paper-sb` 三种 QLSA 入口。
  - 支持 epsilon-greedy、softmax，以及 paper-sb 的 diversity threshold 扫描。
- `scripts/analyze_qlsa_variant_experiments.py`
  - 生成 QLSA 机制对齐汇总、Markdown 分析和中文图表。
- `results/raw/qlsa_variant_alignment_raw.csv`
  - berlin52、eil76、rat99、eil101，300000 次迭代，32 条链，8 线程，重复 3 次。
- `results/summary/qlsa_variant_alignment_summary.csv`
- `docs/dev/qlsa_variant_alignment_analysis.md`
- `figures/final/fig_qlsa_variant_alignment.png`
- `scripts/run_cuda_nsight_profile.py`
  - 增强 Nsight Systems / Nsight Compute 检测路径。
  - 支持 `--ncu-set`。
  - 增加常见 Windows 安装路径查找。
- `docs/dev/cuda_nsight_profile_deep_analysis.md`
  - 记录 a280 CUDA candidate-hybrid 的 Nsight Systems 结果。
  - 记录 Nsight Compute 因 GPU performance counters 权限阻止采集 occupancy / 带宽指标。
- `docs/final/final_report_course_rewrite_v2.md`
  - 第 7.3 节加入 QLSA `current` / `paper` / `paper-sb` 对齐实验。
  - 第 7.5 节加入 Nsight Systems 分析。
  - 第 9 节移除 MPI 双虚拟机问题项，只保留任务粒度、CUDA 路径操作、QLSA 参数敏感三个代表性问题。
  - 第 10 节后续工作只保留分布式搜索增强与统计检验补充。
- `docs/final/known_limitations.md`
  - 更新 paper / paper-sb 已完成代表实例实验的状态。
- `docs/final/reproduction_commands.md`
  - 增加 QLSA 机制对齐实验与 Nsight profiling 复现命令。
- `results/final/RESULTS_INDEX.md`
  - 增加 QLSA 机制对齐与 Nsight Systems 记录索引。
- `README.md`
  - 增加 QLSA 论文机制对齐结果位置，删除不必要的 hybrid quick 单项路径清单。

关键观察：

- QLSA paper-sb 在 eil76 上达到 BKS=538。
- QLSA paper-sb 在 eil101 上取得 632，最小偏差 0.477%。
- rat99 在 300000 次迭代预算下仍由 current epsilon-greedy 取得较好结果，最短路径为 1231。
- Nsight Systems 在 ASCII 临时路径下成功捕获 a280 candidate-hybrid：
  - Host-to-Device 约 0.314 MB；
  - Device-to-Host 约 0.074 MB；
  - CUDA API 时间主要集中在 `cudaDeviceSynchronize`，约 637 ms；
  - 数据传输不是本次配置的主要代价。
- Nsight Compute 已找到，但当前用户没有 GPU performance counters 权限，因此没有写 occupancy、SM 利用率或内存带宽数值。

验证：

```powershell
py -m py_compile scripts\run_qlsa_variant_experiments.py scripts\analyze_qlsa_variant_experiments.py scripts\run_cuda_nsight_profile.py
py scripts\check_report_assets.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_report_format.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_privacy_and_encoding.py
ctest --test-dir build-cuda-ninja --output-on-failure
```

结果：

- Python 脚本语法检查通过。
- 报告图片检查通过。
- 报告格式检查通过。
- 隐私与 UTF-8 检查通过。
- CTest 6/6 通过。

仍需保持的边界：

- 不把 paper-sb 代表实例实验改写成历史 Step 5/6 主实验。
- 不把 CUDA profiling 写成 CUDA 优于 OpenMP。
- 不报告 Nsight Compute 未能采集的硬件计数器。
- 不把 QLSA 写成所有实例上都优于 SA。

## 最终课程报告主入口强化

时间：2026-06-23 02:20

用户说明最终只会提交一份项目报告和一份个人报告，因此本阶段把当前项目报告整理为正式入口，并重点增强第 4 节“多搜索链并行化方案”。

修改内容：

- `docs/final/final_report_course_rewrite_v2.md`
  - 重写第 4 节，增加并行粒度选择理由、数据归属、OpenMP/CUDA/MPI 三类任务划分、同步位置和代价分析。
  - 说明为什么不把 CPU 端 2-opt 单次移动作为主要并行粒度。
  - 明确 OpenMP 是主性能后端，CUDA candidate 是 GPU 端候选批量评价和质量探索，MPI + OpenMP 是分布式链级划分验证。
- `docs/final/final_report_course.md`
  - 已同步为强化后的最终项目报告入口。
- `docs/final/personal_report.md`
  - 更新不足部分，反映 `paper` / `paper-sb` 已完成代表实例实验。
  - 删除生硬边界式表述，改为更自然的课程个人总结语气。

报告逻辑调整：

- 项目报告围绕“多搜索链 SA/QLSA 并行化实现”展开。
- 第 4 节从以下角度展开：
  - 链级并行的原因；
  - OpenMP 的线程级任务划分；
  - CUDA 多链与候选批量评价；
  - MPI + OpenMP 的进程级与线程级两层划分；
  - 三类后端的任务粒度、同步位置和适用问题对比。
- 实验结果章节继续保留 OpenMP、调优增强、QLSA 机制对齐、CUDA、MPI、大实例压力测试等数据，但正文强调分析而不是清单式列文件。

验证：

```powershell
py scripts\check_report_assets.py docs\final\final_report_course.md
py scripts\check_report_format.py docs\final\final_report_course.md
py scripts\check_privacy_and_encoding.py
ctest --test-dir build-cuda-ninja --output-on-failure
```

结果：

- 报告图片检查通过。
- 报告格式检查通过，0 warning。
- 隐私与 UTF-8 检查通过。
- CTest 6/6 通过。

当前最终提交建议：

- 项目报告：`docs/final/final_report_course.md`
- 个人报告：`docs/final/personal_report.md`

不建议把旧版 rewrite、dev 过程文档、构建目录、日志、私钥、浏览器/GPT 临时材料放入课程最终提交包。

## 项目报告唯一入口与并行方案强化

时间：2026-06-23 03:10

用户要求最终只保留一份项目报告，并继续强化第 4 节并行方案说明，同时优化第 5 节实验流程文字、增强第 6-8 节展示内容。

文件整理：

- 删除旧项目报告草稿：
  - `docs/final/final_report_course_rewrite.md`
  - `docs/final/final_report_course_rewrite_v2.md`
  - `docs/final/final_report_course.pdf`
- 删除旧提交包内重复报告：
  - `submission/course/final_report.md`
- 当前项目报告唯一入口：
  - `docs/final/final_report_course.md`
- 当前个人报告：
  - `docs/final/personal_report.md`
- 更新 `submission/README.md`，说明提交文档以 `docs/final/` 下两份文件为准。
- 更新 `README.md`，删除旧 PDF 入口，增加个人报告入口。

报告强化：

- 第 4 节继续扩充：
  - 加入链级并行开销模型；
  - 说明移动级并行与链级并行的区别；
  - 增加随机种子派生、结果数组写入、内层无锁执行说明；
  - 增加 CUDA 路径反转、共享内存候选信息、最优路径协作复制说明；
  - 增加 MPI 通信量估计和连续链编号分段策略；
  - 增加 OpenMP / CUDA / MPI 三类后端任务粒度对比；
  - 增加与课程并行算法要求的对应关系。
- 第 5 节重写：
  - 将“CSV 和脚本流程”改成工程可复现性设计；
  - 按数据、构建、运行、汇总、检查说明实验流程；
  - 用测试层次说明从单链到多后端的实现顺序。
- 第 6 节增强：
  - 增加实验矩阵表；
  - 增加代表实例分层表。
- 第 7 节增强：
  - 增加成果总览表，展示 OpenMP、QLSA、CUDA、MPI、大实例压力测试各自作用。
- 第 8 节增强：
  - 增加参考论文与本项目的对比表；
  - 明确质量指标和运行时间指标的不同对比口径。

验证：

```powershell
py scripts\check_report_assets.py docs\final\final_report_course.md
py scripts\check_report_format.py docs\final\final_report_course.md
py scripts\check_privacy_and_encoding.py
ctest --test-dir build-cuda-ninja --output-on-failure
```

结果：

- 报告图片检查通过。
- 报告格式检查通过，0 warning。
- 隐私与 UTF-8 检查通过。
- CTest 6/6 通过。

补充说明：

- `docs/final/known_limitations.md` 与 `results/final/RESULTS_INDEX.md` 仍包含若干安全边界说明，例如 CUDA、MPI 和 QLSA 的结论边界；这些不是项目报告正文。
- 项目报告正文已避免旧的生硬表述和旧版报告路径。

## 删除 submission 目录并收敛最终文档入口

时间：2026-06-23 05:15

用户说明最终提交文档在 `docs/` 下，只需要项目报告和个人报告。因此检查 `submission/` 是否有独有重要材料后，删除该目录。

检查结果：

- `submission/course/figures/` 中的图片均为 `figures/final/` 下已有图片的副本或旧版命名副本。
- `submission/course/results_key/` 中的 CSV 均可在 `results/final/`、`results/summary/` 或 `results/reference/` 中找到。
- `submission/course/reproduction_commands.md` 内容由 `docs/final/reproduction_commands.md` 覆盖。
- `submission/README.md` 只是提交包说明，不包含独有实验数据。

执行内容：

- 删除整个 `submission/` 目录。
- 更新 `.gitignore`，移除旧 submission 目录忽略规则。
- 更新 `README.md`，删除 `submission/course/` 入口，只保留：
  - `docs/final/final_report_course.md`
  - `docs/final/personal_report.md`
  - `results/final/RESULTS_INDEX.md`
  - `docs/final/reproduction_commands.md`
- 更新 `docs/final/known_limitations.md`，说明最终课程提交文档只保留两份。
- 更新 `scripts/check_privacy_and_encoding.py`，不再扫描已删除的 `submission/README.md`。
- 更新 `scripts/check_final_submission.py`，改为检查当前最终文档入口和关键结果文件。

验证：

```powershell
Test-Path submission
py scripts\check_privacy_and_encoding.py
py scripts\check_final_submission.py
```

结果：

- `Test-Path submission` 返回 `False`。
- 隐私与 UTF-8 检查通过。
- 最终课程文档检查通过。

当前最终提交文档：

- 项目报告：`docs/final/final_report_course.md`
- 个人报告：`docs/final/personal_report.md`

## QLSA 论文机制对齐与 CUDA 候选策略补强

本阶段针对两个之前列为后续工作的方向继续实现：论文机制对齐，以及 CUDA 端候选策略与 profiling 入口。

源码更新：

- `include/tsp/qlsa.hpp`
  - 新增 `QLSAParams::variant`，支持 `current`、`paper`、`paper-sb`。
  - 新增 `QLSAParams::diversity_threshold`。
- `src/qlsa.cpp`
  - 保留原有工程化 QLSA 为 `current` 变体。
  - 新增 candidate-leader 版本：
    - `paper`：无状态 candidate-leader；
    - `paper-sb`：candidate-leader + Hamming diversity state。
  - candidate leader 动作包含 current、global best、random、double-bridge。
- `src/main.cpp`
  - 新增 `--qlsa_variant current|paper|paper-sb`。
  - 新增 `--diversity_threshold`。
  - CUDA QLSA 明确限制为 `current` 变体，避免误把 CUDA 结果标成 paper-sb。
- `include/tsp/parallel.hpp`、`src/cuda_kernels.cu`
  - `CudaCandidatePolicy` 增加 `Hybrid`。
  - CUDA candidate kernel 支持 `best`、`random`、`hybrid`。
- `tests/test_qlsa_paper_lite.cpp`
  - 覆盖 paper 和 paper-sb 在 square4 上可运行并得到最短路径 40。
- `tests/test_cuda_candidate.cpp`、`tests/test_cuda_qlsa_candidate.cpp`
  - 覆盖 CUDA candidate hybrid 策略。
- `scripts/run_cuda_nsight_profile.py`
  - 新增 Nsight Systems / Nsight Compute 检测与执行入口。

文档更新：

- `docs/dev/paper_mechanism_alignment.md`
  - 记录 current、paper、paper-sb 三种 QLSA 变体的机制差异和报告边界。
- `docs/dev/cuda_candidate_policy_design.md`
  - 更新 best/random/hybrid 策略、共享内存路径操作和 Nsight 状态。
- `docs/dev/cuda_candidate_hybrid_quick_analysis.md`
  - 记录 berlin52、a280 上 SA/QLSA chain、candidate-best、candidate-random、candidate-hybrid 的 quick 对照。
- `docs/dev/cuda_nsight_profile_analysis.md`
  - 记录 Nsight Systems 未在 PATH 中找到，Nsight Compute 已生成 `.ncu-rep`。
- `docs/final/known_limitations.md`
  - 改为：C++ 已提供可选 paper/paper-sb 机制对齐变体，但历史主实验仍主要来自 current 变体。
- `docs/final/reproduction_commands.md`
  - 增加 paper-sb、CUDA hybrid quick 和 Nsight 复现命令。
- `README.md`
  - 增加 QLSA paper-sb 和 CUDA hybrid 示例。
- `docs/final/final_report_course_rewrite_v2.md`
  - 更新后续工作与 CUDA 策略描述，避免把已实现内容继续写成待做。

新增结果：

- `results/raw/cuda_candidate_hybrid_quick_raw.csv`
- `results/summary/cuda_candidate_hybrid_quick_summary.csv`
- `figures/final/fig_cuda_candidate_hybrid_quick.png`
- `results/logs/nsight/cuda_candidate_a280_hybrid_ncu.ncu-rep`
- `results/logs/nsight/cuda_candidate_a280_hybrid_ncu.log`

验证命令：

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake --build build-cuda-ninja --clean-first -j"
ctest --test-dir build-cuda-ninja --output-on-failure
.\build-cuda-ninja\tsp_sa.exe --qlsa --qlsa_variant paper-sb --input tests\fixtures\square4.tsp --iterations 500 --seed 1 --init random --csv-only
.\build-cuda-ninja\tsp_sa.exe --input data\berlin52.tsp --parallel cuda --cuda_mode candidate --cuda_candidate_policy hybrid --cuda_reversal_mode parallel --cuda_candidates_per_iter 128 --chains 32 --cuda_block_size 128 --iterations 100000 --seed 1 --init nn --csv-only
.\build-cuda-ninja\tsp_sa.exe --qlsa --input data\berlin52.tsp --parallel cuda --cuda_mode candidate --cuda_candidate_policy hybrid --cuda_reversal_mode parallel --cuda_candidates_per_iter 128 --chains 32 --cuda_block_size 128 --iterations 100000 --seed 1 --init nn --csv-only
py scripts\run_cuda_candidate_experiments.py --instances berlin52 a280 --algorithms sa qlsa --iterations 100000 --repeat 1 --chains 32 --block-sizes 128 --candidates-per-iter 128 --reversal-modes parallel --candidate-policies best random hybrid --output results\raw\cuda_candidate_hybrid_quick_raw.csv
py scripts\analyze_cuda_candidate.py --input results\raw\cuda_candidate_hybrid_quick_raw.csv --output results\summary\cuda_candidate_hybrid_quick_summary.csv --markdown docs\dev\cuda_candidate_hybrid_quick_analysis.md --figure figures\final\fig_cuda_candidate_hybrid_quick.png
py scripts\run_cuda_nsight_profile.py --instance a280 --iterations 20000 --policy hybrid
```

验证结果：

- clean build 通过。
- `ctest` 6/6 通过。
- `qlsa-paper-sb` 在 square4 上返回 40。
- CUDA SA/QLSA candidate-hybrid 在 berlin52 上均达到 7542。
- a280 quick 中 candidate-best 和 candidate-hybrid 明显降低偏差，但运行时间仍高于 chain 模式。
- Nsight Compute 可用并生成 `.ncu-rep`；Nsight Systems 未在 PATH 中找到。

结论边界：

- 可以写 C++ 已提供 paper/paper-sb 机制对齐入口。
- 不能把历史 Step 5/6 实验改写成 paper-sb 结果。
- 可以写 CUDA 已支持 best/random/hybrid 候选策略、并行路径反转和 Nsight Compute 捕获。
- 不能写 CUDA 性能优于 OpenMP。

## CUDA 与报告并行策略补强

本阶段根据用户反馈，继续推进 CUDA 实现和课程报告表达，重点不再停留在“实验过程”，而是补充并行策略、任务划分、同步方式和性能优化原因。

代码与脚本更新：

- CUDA candidate 模式新增候选选择策略：
  - `--cuda_candidate_policy best`
  - `--cuda_candidate_policy random`
- `best` 策略选择候选批中增量最小的 2-opt 移动，偏向解质量。
- `random` 策略从候选批中可复现随机选择一个移动，用于对照批量择优带来的质量收益和时间成本。
- CUDA candidate kernel 增加块内协作复制最优路径，减少 thread 0 在路径复制阶段的串行循环。
- CUDA QLSA candidate 测试与 SA candidate 测试均覆盖 random 策略、串行路径反转和并行路径反转。
- `scripts/run_cuda_candidate_experiments.py`、`scripts/run_cuda_candidate_sweep.py`、`scripts/analyze_cuda_candidate.py`、`scripts/analyze_cuda_candidate_sweep.py` 已支持 `cuda_candidate_policy` 字段。

新增或更新的 CUDA 实验：

- quick 对照：
  - `results/raw/cuda_candidate_policy_quick_raw.csv`
  - `results/summary/cuda_candidate_policy_quick_summary.csv`
  - `figures/final/fig_cuda_candidate_policy_quick.png`
- formal 对照：
  - `results/raw/cuda_candidate_policy_formal_raw.csv`
  - `results/summary/cuda_candidate_policy_formal_summary.csv`
  - `docs/dev/cuda_candidate_policy_formal_analysis.md`
  - `figures/final/fig_cuda_candidate_policy_formal.png`

formal 实验结论：

- `candidate-best` 在 a280、lin318、rat575 上明显降低 Gap。
- `candidate-random` 比 `candidate-best` 更快，但质量收益明显较弱。
- CUDA candidate 仍以质量探索和 GPU 工程扩展为主，OpenMP 仍是主性能后端。

报告更新：

- `docs/final/final_report_course_rewrite_v2.md` 第 3 节补充复杂度和并行粒度选择关系。
- 第 4 节重写为并行策略说明：
  - 任务集合；
  - OpenMP 线程级链任务划分；
  - CUDA 块级链任务划分和块内候选批量评价；
  - MPI 进程级连续链段划分；
  - 私有数据、只读共享数据、归约和通信方式。
- 第 5 节补充性能优化点：
  - 距离矩阵连续存储；
  - O(1) 2-opt 增量；
  - 链内状态私有化；
  - 固定随机种子派生；
  - CUDA 共享内存候选信息；
  - MPI 低频通信。
- 第 7.1 节补充 OpenMP 加速原因分析：内层循环无锁、距离矩阵只读共享、链结束后归约。
- 第 7.5 节重写 CUDA 分析，加入 best/random 候选策略 formal 数据和 `fig_cuda_candidate_policy_formal.png`。
- 第 7.6 节补充 MPI 通信模型分析。
- 第 7.7 节补充困难实例、大实例、千点级实例的选择理由和内存规模解释。
- 第 9 节去除模板化开头，保留任务粒度、CUDA 路径操作、MPI 真实分布式证据、QLSA 参数敏感四个代表性问题。

图表处理：

- 当前报告引用的所有图均重新检查。
- 主数据系列不再使用灰色配色。
- 网格线由浅灰改为淡蓝，避免视觉上出现灰色主调。
- `fig_cuda_candidate_policy_formal.png` 改为按实例分组，横坐标只保留实例名，避免标签过密。

验证：

```powershell
py scripts\check_report_assets.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_report_format.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_privacy_and_encoding.py
ctest --test-dir build-cuda-ninja --output-on-failure
```

结果：

- 报告图片检查通过。
- 报告格式检查通过。
- 隐私与 UTF-8 检查通过。
- CTest 6/6 通过。

## 报告算法原理与流程补强

用户要求报告能让没读过参考论文或不了解项目的人也能理解完整过程，因此对 `docs/final/final_report_course_rewrite_v2.md` 做了结构性补强：

- 第 2 节增加论文方法拆解：
  - 普通 SA；
  - QLSA；
  - State-Based QLSA；
  - 论文内容与本项目实现的对应关系。
- 第 3 节改为带小标题的算法说明：
  - `3.1 问题建模与数据表示`
  - `3.2 SA 算法描述`
  - `3.3 QLSA 算法描述`
  - `3.4 算法分析`
- 第 3 节补充了：
  - TSP 路径表示；
  - 距离矩阵一维存储；
  - 2-opt O(1) 增量；
  - SA 单链执行步骤；
  - QLSA 状态、动作、奖励和 Q 表更新；
  - 距离矩阵、路径和主循环复杂度分析。
- 第 4 节补充并行设计分析：
  - 为什么选择链级并行；
  - 为什么不优先做 2-opt 内部细粒度 CPU 并行；
  - OpenMP、CUDA 多链、CUDA 候选批量评价、MPI + OpenMP 的粒度、通信数据和主要代价。
- 第 5 节补充运行环境：
  - Windows 主机；
  - CPU：12th Gen Intel(R) Core(TM) i5-12600KF；
  - GPU：NVIDIA GeForce RTX 4070 SUPER；
  - MSVC 19.44 / nvcc 12.9；
  - CMake + Ninja；
  - Release；
  - OpenMP、CUDA、MPI + OpenMP。
- 第 5 节补充项目推进步骤：
  - 数据解析；
  - 串行 SA；
  - QLSA；
  - 串行多链与 OpenMP；
  - CUDA 多链与候选批量评价；
  - MPI + OpenMP；
  - 实验脚本、汇总、图表和报告。

验证已通过：

```powershell
py scripts\check_report_assets.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_report_format.py docs\final\final_report_course_rewrite_v2.md
py scripts\check_privacy_and_encoding.py
```
## 2026-06-23 图片目录清理

本轮目标是清理无用图片，并将课程报告实际使用的图片统一放在 `figures/` 根目录，不再使用 `figures/final/`。

已完成：

- 核对 `docs/final/final_report_course.md` 的图片引用，只保留 12 张当前报告实际使用的图：
  - `fig_course_01_openmp_speedup.png`
  - `fig_course_02_openmp_efficiency.png`
  - `fig_course_03_default_gap.png`
  - `fig_course_04_targeted_quality.png`
  - `fig_course_05_policy_comparison.png`
  - `fig_course_06_cuda_boundary.png`
  - `fig_course_07_mpi_scaling.png`
  - `fig_course_09_paper_quality.png`
  - `fig_course_10_openmp_thread_scaling.png`
  - `fig_course_11_representative_openmp.png`
  - `fig_cuda_candidate_policy_formal.png`
  - `fig_qlsa_variant_alignment.png`
- 将上述图片移动到 `figures/` 根目录。
- 删除未被课程报告引用的历史图片和 `figures/final/` 目录。
- 删除 `results/logs/final_cleanup/course_figures_contact_sheet.png` 旧 contact sheet。
- 更新 `docs/final/final_report_course.md`、`results/final/RESULTS_INDEX.md` 和 `docs/final/reproduction_commands.md` 中的图片路径。
- 更新图表生成/分析脚本默认输出目录，避免以后重新生成到 `figures/final/`。
- 重写 `README.md` 和 `.gitignore`，修复其中的中文乱码。
- 重写 `scripts/check_report_assets.py`、`scripts/check_privacy_and_encoding.py`、`scripts/check_final_submission.py` 和 `scripts/analyze_policy_comparison.py`，恢复 UTF-8 正常中文，并加强对旧图片目录和乱码的检查。

验证结果：

```powershell
py scripts\check_report_assets.py docs\final\final_report_course.md
py scripts\check_report_format.py docs\final\final_report_course.md
py scripts\check_privacy_and_encoding.py
py scripts\check_final_submission.py
py -m py_compile scripts\check_report_assets.py scripts\check_final_submission.py scripts\check_privacy_and_encoding.py scripts\analyze_policy_comparison.py scripts\make_report_figures.py scripts\make_course_report_figures.py scripts\analyze_qlsa_variant_experiments.py scripts\analyze_large_mpi_vm.py scripts\analyze_large_cuda.py scripts\run_mpi_vm_scaling.py scripts\analyze_large_openmp.py scripts\run_python_reference_comparison.py scripts\analyze_cuda_candidate.py scripts\analyze_cuda_reversal.py scripts\analyze_all_data_openmp.py scripts\analyze_cuda_candidate_sweep.py scripts\analyze_cuda_qlsa_candidate.py scripts\make_hpc_architecture_figure.py
```

结果均通过。当前全仓库图片扫描仅剩 `figures/` 下 12 张课程报告图片，`figures/final/` 不存在。

## 2026-06-23 报告收尾调整

本轮根据用户对最终课程报告的反馈，完成三项定向修改：

- 重新生成 `figures/fig_course_10_openmp_thread_scaling.png`：
  - berlin52 SA、berlin52 QLSA、eil101 SA、eil101 QLSA 四条实验曲线分别使用蓝、橙、绿、红四种颜色；
  - 理想线使用紫色虚线，避免和实验曲线混淆。
- 更新 `docs/final/final_report_course.md` 第 7.7 节：
  - 将 dsj1000、u1060、vm1084 的千点级最小偏差改为表格呈现；
  - 补充说明同样百万迭代预算下，千点级实例偏差较高的原因，以及 QLSA 与 SA 在该组实例上的差异。
- 扩充第 10 节总结：
  - 从链级并行设计、OpenMP 性能、QLSA 解质量、CUDA/MPI 工程扩展和整体课程完成度五个角度收束；
  - 保持结论边界，不扩大 CUDA、MPI 或 QLSA 的实验含义。

同时修复：

- `scripts/check_report_assets.py` 和 `scripts/check_report_format.py` 支持 Markdown 图片与 HTML `<img>` 两种引用方式；
- `scripts/check_report_format.py` 恢复 UTF-8 正常中文检查规则；
- 报告中少量“第二/第三”重复编号表达。

验证命令：

```powershell
py scripts\check_report_assets.py docs\final\final_report_course.md
py scripts\check_report_format.py docs\final\final_report_course.md
py scripts\check_privacy_and_encoding.py
py scripts\check_final_submission.py
py -m py_compile scripts\check_report_assets.py scripts\check_report_format.py scripts\make_course_report_figures.py
```

结果均通过。

## 2026-07-11 Island 与公平实验报告整合

在三方只读审计确认本轮实现能够明显增强课程报告的算法设计深度、实验公平性和证据可信度后，将 `codex/fair-experiments-island-migration` 快进合并到本地 `main`，并在干净的 `main@5ccaf3c` 上重新执行正式实验。

### 干净版本重跑

- 公平配对实验：eil76、4 种算法、20 个共同种子，**80/80 jobs**；
  - manifest 记录 `git_commit=5ccaf3c`、`git_dirty=false`；
  - SA 与 paper-sb 平均最优长度均为 `540.6`，BKS 命中率为 `10%/25%`；
  - Friedman `p=0.1092`，Holm 校正后无两两比较达到 0.05。
- OpenMP island 消融：eil76、2 种算法、10 个共同种子、3 种拓扑、2 个迁移周期，**120/120 jobs**；
  - manifest 记录 `git_commit=5ccaf3c`、`git_dirty=false`；
  - SA 在 10k 周期下 ring/global 相对匹配 independent 的平均差为 `-3.5/-4.0`；
  - 原始 sign-test `p=0.0215`，Holm 后 `p=0.0859`，报告按改善趋势表述；
  - paper-sb 未观察到稳定迁移收益。
- 第一份公平重跑在 70/80 时遇到 Windows 对 `manifest.json.tmp` 的短暂文件占用，目录已重命名为带 `incomplete_winerror5` 后缀并保留在 `tmp/`；正式数据只采用完整 v2 目录。

### 正式摘要归档

新增以下可公开、无本机绝对路径的汇总文件：

- `results/summary/fair_paired_eil76_summary.csv`
- `results/summary/fair_paired_eil76_pairwise.csv`
- `results/summary/fair_paired_eil76_friedman.csv`
- `results/summary/island_eil76_summary.csv`
- `results/summary/island_eil76_paired_comparisons.csv`
- `results/final/fair_island_clean_run_provenance.csv`

同步更新 `results/final/RESULTS_INDEX.md` 与 `docs/final/known_limitations.md`，明确单机 OpenMP island、单实例覆盖和多重校正边界。

### 本地课程报告

`docs/final/report.md` 与根目录 `小组及个人报告.pdf` 属于 `.gitignore` 排除的本地课程交付物，未发布到远端。正文按原有“结论—数据—原因—边界”格式加入：

- OpenMP island 的可续跑状态、同步边界与三种拓扑；
- requested/actual backend、实际线程/迭代、deadline 和 total/kernel 分层计时；
- 20 个共同种子的四算法公平复核；
- 120-job island 迁移消融；
- 统计与结论边界、实施问题、总结、后续方向和个人附录同步更新。

PDF 采用 A4、中文无衬线字体、现有标题层级和灰色表格样式重新生成，共 30 页；已检查全部页面缩略图及新增内容页，未发现文字裁切、表格越界、图片缺失或乱码。

主要验证命令：

```powershell
python scripts\check_report_format.py docs\final\report.md
python scripts\check_report_assets.py docs\final\report.md
python scripts\check_final_submission.py
python scripts\check_privacy_and_encoding.py
ctest --test-dir tmp\build-measurement-release --output-on-failure
ctest --test-dir tmp\build-measurement-cpu-release --output-on-failure
python -S -m unittest tests.test_fair_experiment_pipeline tests.test_fair_runner_validation tests.test_fair_analyzer_integrity tests.test_island_ablation_pipeline tests.test_island_runner_hardening tests.test_island_analyzer_hardening
```

## 2026-07-11 公平实验、限时搜索与 OpenMP island migration

本轮工作在独立分支 `codex/fair-experiments-island-migration` 上完成，目标是补齐课程项目在“实验公平性、后端真实性、统计可信度和可协作多链搜索”上的工程缺口；未修改最终报告正文。

### 核心实现

- 将 SA 与 QLSA（`current`、`paper`、`paper-sb`）重构为 `initialize / run_chunk / finalize` 状态接口：
  - 状态完整保存 RNG、温度、当前/最优路径、Q 表、动作计数和滚动窗口；
  - 分块续跑与一次性运行保持确定性等价；
  - 支持相对 time limit 和多 worker 共享的绝对 deadline；
  - 结果记录实际完成迭代数与 deadline 状态。
- 新增 OpenMP island 模型：
  - 支持 `independent`、`ring`、`global` 三种拓扑；
  - 迁移仅发生在同步 chunk 边界，避免 worker 读取正在变化的跨岛状态；
  - 记录迁移轮次、尝试数、采纳数和真实 OpenMP team size。
- 扩展 CLI：
  - `--time-limit-ms`
  - `--migration-topology disabled|independent|ring|global`
  - `--migration-interval`
- CSV 保留原 14 列前缀，并追加 total/kernel 计时、requested/actual backend、fallback 原因、实际迭代、deadline、迁移统计和 `actual_threads`。

### 测量与后端真实性

- `elapsed_ms` 与 `total_elapsed_ms` 统一为 end-to-end solver/backend 时间。
- CUDA 另行记录 `cuda_kernel_elapsed_ms`；total 覆盖首次设备探测、分配、H2D、kernel、D2H、释放和结果校验。
- CUDA/OpenMP fallback 不再伪装成请求后端，结构化记录 actual backend 与原因。
- OpenMP 在真实 parallel region 内记录 `omp_get_num_threads()`，实验 runner 会拒绝实际线程数与请求不一致的样本。
- Release 测试目标显式取消 `NDEBUG`，所有 assertion-based 测试增加编译期哨兵。

### 公平实验流水线

- 新增 `configs/fair_experiment_matrix.json`、`scripts/run_fair_experiments.py` 和 `scripts/analyze_paired_experiments.py`：
  - 20 个 paired seeds；
  - 等搜索迭代与固定 solver wall time 两种互补预算；
  - 移除与等迭代命令完全重复、无法代表真实等工作的旧 `equal-proposals` 预算；
  - seed 内使用可复现循环顺序平衡算法执行位置；
  - 正式运行必须显式指定 executable；
  - 严格核对算法/变体、预算、链数、初始化、requested/actual backend 和 actual threads；
  - 记录环境、编译器、GPU、Git 状态和 config/input/executable/environment SHA-256；
  - 目录分析前校验 manifest、raw、config snapshot、日志和 checksum sidecar。
- 统计分析默认要求四算法完整共同 seed block，输出 bootstrap CI、配对 Wilcoxon、精确 sign test、Friedman 与 Holm 校正；不同 condition/provenance 不会混组。

### Island 消融流水线

- 新增 `configs/island_ablation_matrix.json`、`scripts/run_island_ablation.py` 和 `scripts/analyze_island_ablation.py`：
  - 默认 3 个实例、SA 与 `paper-sb`、10 个 paired seeds；
  - 10k/100k 两个 cadence；
  - 每个 cadence 都运行匹配的 independent/ring/global，避免把 chunk/barrier 开销误算成迁移收益；
  - 默认矩阵 360 jobs，每 island 1,000,000 iterations；
  - 六个 algorithm-condition 位置按 seed 循环轮换；
  - 默认要求每 interval 三拓扑的 seed 集完全一致；
  - 输出迁移采纳率、paired differences、精确 sign test 和 condition 内 Holm 校正。

### 验证结果

- Release + MSVC + CUDA 12.9 + OpenMP：CTest **8/8**。
- Release CPU-only（关闭 CUDA/OpenMP/MPI）：CTest **8/8**，覆盖真实 fallback。
- Python 标准库流水线测试：**60/60**。
- UTF-8、隐私与 mojibake 检查通过。
- 30 秒 fixed-time 实测：
  - solver total `30000.048 ms`；
  - process wall `30012.714 ms`；
  - `deadline_reached=true`；
  - actual OpenMP threads `8/8`。
- eil76 等迭代正式样本：4 algorithms × 20 seeds = **80/80 jobs**；完整 seed blocks=20，83 个 run artifact 通过 SHA-256 校验。
  - SA 与 paper-sb 平均最优长度均为 `540.6`；
  - BKS 命中率分别为 `10%` 与 `25%`；
  - Friedman `p=0.1092`，Holm 校正后的 pairwise 比较均未达到 0.05，不能宣称显著普遍优势。
- eil76 island 正式消融：2 algorithms × 10 seeds × 6 matched conditions = **120/120 jobs**。
  - SA 在 10k cadence 下 ring/global 相对 independent 的平均最优长度差为 `-3.5/-4.0`；
  - 两者原始 sign-test `p=0.0215`，Holm 后均为 `0.0859`，属于强趋势而非校正后显著结论；
  - paper-sb 未显示稳定迁移收益。

### 当前边界与后续实验

- 尚未执行 5 实例 × 20 seeds 的完整公平矩阵；本轮正式结果只覆盖 eil76 等迭代口径。
- fixed-time 已验证真实 30 秒样本和严格数据契约，但未跑完全部算法/实例/seed。
- island 正式消融目前只完成 eil76；rat99、eil101 仍需按默认矩阵执行。
- CUDA 暂不支持 mid-kernel time limit，island migration 当前为单机 OpenMP；MPI 跨节点迁移仍属于后续工作。
- 课程报告正文按用户要求未在本分支更新。

主要验证命令：

```powershell
ctest --test-dir tmp\build-measurement-release --output-on-failure -C Release
ctest --test-dir tmp\build-measurement-cpu-release --output-on-failure -C Release
python -S -m unittest tests.test_fair_experiment_pipeline tests.test_fair_runner_validation tests.test_fair_analyzer_integrity tests.test_island_ablation_pipeline tests.test_island_runner_hardening tests.test_island_analyzer_hardening
python scripts\check_privacy_and_encoding.py
```

## 2026-06-23 个人报告同步更新

根据当前最终课程报告 `docs/final/report.md`，重写了 `docs/final/personal_report.md`。

更新重点：

- 个人报告主线对齐课程报告：
  - 多搜索链作为统一并行粒度；
  - OpenMP 是主性能后端；
  - CUDA 是 GPU 端候选批量评价和路径操作工程扩展；
  - MPI + OpenMP 是双虚拟机分布式链级划分验证；
  - 大实例压力测试用于说明工程可扩展性。
- 保留单人团队信息、姓名、学号和学院专业。
- 按个人承担工作、关键实现内容、实验与分析、问题处理、收获与不足组织，而不是简单罗列文件。
- 避免扩大 CUDA、MPI 或 QLSA 结论。

验证命令：

```powershell
py scripts\check_privacy_and_encoding.py
py scripts\check_final_submission.py
py scripts\check_report_assets.py docs\final\report.md
py scripts\check_report_format.py docs\final\report.md
```

结果均通过。

## 2026-06-23 个人报告并入小组报告

根据课程要求“成员报告以附录形式附在小组报告后”，将个人报告内容并入 `docs/final/report.md` 末尾，新增：

- `## 附录 A：个人报告`
- 基本信息；
- 个人承担的作用；
- 主要完成工作；
- 实验和分析工作；
- 遇到的问题和处理；
- 收获与不足。

为避免重复提交内容，`docs/final/personal_report.md` 已改为说明文件，提示个人报告已并入 `docs/final/report.md` 的附录 A，不再保存另一份重复正文。

同步更新：

- `README.md` 中的最终入口说明。

验证命令：

```powershell
py scripts\check_report_assets.py docs\final\report.md
py scripts\check_report_format.py docs\final\report.md
py scripts\check_privacy_and_encoding.py
py scripts\check_final_submission.py
```

结果均通过。

## 2026-06-23 个人报告更新

根据当前最终项目状态，重写 `docs/final/personal_report.md`，使其与最新课程报告和实验状态一致。

更新重点：

- 明确单人团队下本人承担选题、论文阅读、算法实现、并行后端、实验脚本、结果分析和报告整理全部工作。
- 补充当前已完成的关键实现：
  - SA/QLSA 串行搜索内核；
  - OpenMP 多搜索链并行；
  - CUDA 多链、候选批量评价、候选策略、并行路径反转和 QLSA candidate；
  - `paper` / `paper-sb` 论文机制对齐入口；
  - MPI + OpenMP 双虚拟机分布式实验；
  - 大实例压力测试与图表整理。
- 将实验分析写成个人工作过程：
  - OpenMP 默认实验作为主性能结论；
  - rat99、eil101 等困难实例用于说明 QLSA 的质量收益；
  - CUDA 和 MPI 分别作为 GPU 与分布式工程扩展；
  - 大实例测试用于说明工程可扩展性。
- 修复 `README.md` 和 `scripts/check_final_submission.py` 中发现的文件内容级乱码。
- 加强 `scripts/check_privacy_and_encoding.py` 对“锛/歚”等乱码模式的检测。

验证命令：

```powershell
py scripts\check_privacy_and_encoding.py
py scripts\check_final_submission.py
py scripts\check_report_assets.py
py scripts\check_report_format.py
```

结果均通过。

## 2026-07-13 优先正确性与可复现性改进

本轮先处理会影响算法有效性、实验可解释性和本地构建复现的问题，未重写课程报告或重跑正式结果。

### 对称 TSP 问题域保护

- 解析器现在明确拒绝 `TYPE: ATSP`，并拒绝非对称的 `EXPLICIT/FULL_MATRIX`；`DistanceMatrix` 对手工构造的 `Instance` 也执行同样校验。
- 原因是当前 O(1) 2-opt 增量公式以对称距离为前提。继续在 ATSP 上运行会得到不正确的长度增量，不能用“可运行”替代“算法正确”。
- 新增 ATSP、非对称矩阵和对称矩阵 fixture/回归测试，错误消息会指出不支持的输入和原因。

### 多链初始化、QLSA 状态与实验语义

- 多链 `--init nn` 现在从每条链的既有 seed 派生最近邻起点；单链仍从 city 0 启动。CPU/OpenMP、MPI、island 和 CUDA chain/candidate 路径均使用同一规则，避免所有链因固定 NN 起点而过早同质化。
- `paper-sb` 新增 `--diversity_metric edge|hamming`：默认 `edge` 用无向回路边集差异，循环移位或反向表示同一对称 TSP 回路时多样性为零；`hamming` 保留论文和历史 CSV 的 position-Hamming 定义。
- `edge` 的 best-tour 邻接表在每条 QLSA state 初始化或全局最优改进时复用，不会在每次迭代构造 `unordered_set`，避免把多样性判断变成高频堆分配热点。
- 公平实验和 island 消融的保存配置显式固定 `hamming`，避免与新的默认 `edge` 样本混合。迁移接受外来解后会清除 `current` QLSA 的近期增量窗口，并按变体/metric 重算学习状态。
- `run_qlsa_variant_experiments.py` 同时修复了扩展后的 28 列程序 CSV 被静默忽略、导致“wrote 0 rows”的问题；输出会记录 paper-sb 的 diversity metric。
- CUDA QLSA 的公开库入口与 CLI 一致，只接受 `current` 变体；即使 CUDA 未编译或运行时无 GPU，也不会静默将 `paper`/`paper-sb` 切换成 CPU fallback。

### CMake 路径与后端复现

- 清理遗留的旧 OneDrive 源目录缓存后，重新生成了本地 Ninja CUDA 构建目录。
- CMake 最低版本提高到 3.24；CUDA 架构默认采用 `native`，替代陈旧的固定 `sm_70`。可通过 `TSP_CUDA_ARCHITECTURES=86;89` 等值固定可分发/CI 目标。
- 新增 `CMakePresets.json`：Ninja 与 VS2022 的 CPU+OpenMP、CUDA+OpenMP Release 预设；严格 preset 缺少 CUDA/OpenMP 会在配置阶段失败，手工配置仍可选择降级。
- QLSA variant runner 和公平/island 配置的 executable candidates 已纳入 Ninja/VS preset 输出目录；使用严格 preset 后不会因仍只查找旧 `build-cuda-ninja` 而找不到程序。

### 本轮验证

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake --preset ninja-cuda-release"
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake --build --preset build-ninja-cuda-release --parallel"
ctest --preset test-ninja-cuda-release --output-on-failure
ctest --preset test-ninja-cpu-release --output-on-failure
py -S -m unittest -v tests.test_fair_experiment_pipeline tests.test_fair_runner_validation tests.test_fair_analyzer_integrity tests.test_island_ablation_pipeline tests.test_island_runner_hardening tests.test_island_analyzer_hardening tests.test_qlsa_variant_runner
py scripts\check_privacy_and_encoding.py
py scripts\check_report_format.py docs\final\report.md
py scripts\check_report_assets.py
py scripts\check_final_submission.py
```

- CUDA/OpenMP preset 和 CPU/OpenMP preset 均为 CTest **8/8** 通过；Python 流水线回归为 **62/62** 通过。
- `tsp_sa --input tests/fixtures/atsp3.tsp` 以明确的对称 TSP 限制错误退出。

### 结果边界

- 既有多链 NN 结果来自修复前固定 city 0 的初始化；旧 `paper-sb` 结果使用 Hamming metric。它们可作为历史记录，但不能与新默认行为直接合并统计。
- 如需发布新默认行为的性能或质量结论，应使用固定 seed、匹配预算和完整 paired-seed 设计重跑，而不是在旧 CSV 上追加样本。
