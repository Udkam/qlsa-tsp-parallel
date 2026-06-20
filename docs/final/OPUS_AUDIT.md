# OPUS 审计记录

本文档记录对 `parallel-algorithm` 项目的一次完整只读审计，作为后续整理与报告重写的依据。审计时间口径为 2026-06，审计范围覆盖报告入口、结果数据、图表路径、隐私、编码、目录结构与最终修改计划。

## 1. 当前报告入口是否清晰

不清晰，存在三处互相矛盾的“主入口”定义：

- `README.md` 指向 `docs/final/final_report_master_v2.md`（行 132、154）。
- `docs/final/REPORT_MANIFEST.md` 与 `docs/final/PROJECT_STRUCTURE.md` 指向 `docs/final/final_report_master.md` 与 course/public 派生版。
- `scripts/check_report_assets.py` 的 `DEFAULT_REPORTS` 第一顺位也是 `final_report_master_v2.md`。

同时存在两个 master：`final_report_master.md`（全中文、10 节、`#` 一级标题）与 `final_report_master_v2.md`（中英混排、英文为主、10 节）。两者结构都与 course/public 的 12 节结构（摘要 + 基本信息 + 12 节 + 附录、`##` 二级标题）不一致，`check_report_format.py` 为此专门用 `MASTER_HEADINGS` 与 `REQUIRED_HEADINGS` 两套规则区别对待。这是入口混乱的根因。

## 2. 哪些报告应保留

- `docs/final/final_report_master.md`：保留并重写为唯一 12 节主报告母版。
- `docs/final/final_report_course.md`：保留为课程派生版（含姓名学号）。
- `docs/final/final_report_public.md`：保留为脱敏公开派生版。
- `docs/final/personal_report_appendix.md`、`reproduction_commands.md`、`known_limitations.md`、`REPORT_MANIFEST.md`、`PROJECT_STRUCTURE.md`：保留。

## 3. 哪些报告应归档

- `docs/final/final_report_master_v2.md`：中英混排、与目标结构不一致，移入 `docs/archive/`，不再作为入口。
- `docs/archive/` 下既有的 `final_report.md`、`final_report_v2/v3/v4.md`、`final_report_v5_package/`、`final_submission_v2/`、`final_report_archived.md` 等：维持归档状态，不动。

## 4. 结果数据是否可追溯

整体可追溯，但发现一处**数据陈旧**问题：

- 三份报告（master、course、public）的“定向增强关键结果”表中 `Mean ms` 列为 `1677.495 / 3348.545 / 329.022 / 1649.518`。
- 权威来源 `results/final/final_key_results.csv` 与 `results/summary/targeted_quality_summary.csv` 中，对应 best-quality 行的 `elapsed_ms_mean` 实为：
  - eil101-SA：1867.987（报告写 1677.495，错误）
  - eil101-QLSA：3348.545（正确）
  - rat99-SA：1804.426（报告写 329.022，错误）
  - rat99-QLSA：3424.631（报告写 1649.518，错误）

其余主表（默认参数 OpenMP speedup/efficiency、policy、论文对比）均与 `step5_multi_cpu_summary.csv`、`policy_comparison_summary.csv`、`report_comparison_summary.csv` 一致，已逐行核对。重写时必须改用权威 CSV 值；修正后 QLSA 在 rat99 上达到 BKS=1211 但耗时约为 SA（1212）的 1.9 倍，这是更诚实、也更有说服力的“质量—时间权衡”叙事。

`RESULTS_INDEX.md` 已正确登记 final/summary/reference/raw/archive 的来源与用途，可作为数据索引保留。

## 5. 图表路径是否一致

一致。`figures/final/` 下存在 `fig01`–`fig10`：

- 三份 docs 报告均以 `../../figures/final/figNN_*.png` 引用 `fig01`–`fig09`，路径正确。
- `submission/course` 与 `submission/public` 报告以 `figures/figNN_*.png` 引用（包内自带 figures 子目录），路径正确。
- `fig10_openmp_scaling_threads.png` 存在但不在主报告正文，符合“fig10 仅作附录补充”的约束。
- `figures/archive/` 保留旧命名图，不影响最终报告。

## 6. 是否存在私人信息泄漏风险

无公开侧泄漏。

- 扫描 `README.md`、`docs/final/final_report_public.md`、`submission/public/**` 未命中 `陈乐浚`、`22361054`、QQ/SYSU 邮箱、`22XXXXXX` 学号样式。
- 姓名学号仅出现在 course 侧：`final_report_course.md`、`personal_report_appendix.md`、`submission/course/**`。
- `.gitignore` 已忽略 `submission/course/`（`git ls-files submission/course` 为空），public 包仍可入库。隔离正确。

## 7. 是否存在乱码或编码问题

未发现。扫描公开文本未命中 `????`、`�`、`锟斤拷` 或拉丁乱码片段。`scripts/check_*.py` 均以 `encoding="utf-8"` 读写，符合约束。

## 8. 当前报告最大问题

不是缺内容，而是：

1. **分析深度不足**：第 8 节多为“给图—列表—下结论”，缺少“为什么 5x 可信、scaling 为何下降、QLSA 为何效率更低、质量收益的时间代价”等机制级分析。
2. **master 弱于派生版**：master 只有 10 节、无摘要/基本信息/个人报告附录，反而比 course/public 单薄，违背“master 是母版”的定位。
3. **工程问题章节偏浅**：现为 7 行，且部分偏运维（下载、Python alias），技术深度不够；课程要求按“问题类型/表现/技术原因/解决方案/对结果影响”展开、至少 9 类真实工程问题。
4. **数据陈旧**：见第 4 点。
5. **入口与结构不统一**：见第 1 点。

## 9. 是否需要重新整理目录

不需要大改。`docs/{final,dev,archive}`、`figures/{final,archive}`、`results/{final,raw,summary,reference,archive}`、`submission/{course,public}` 分层已符合目标。只需小幅收敛：

- 归档 `final_report_master_v2.md`。
- 统一 README / MANIFEST / PROJECT_STRUCTURE / 检查脚本的入口为 `final_report_master.md`。

## 10. 修改计划

1. 归档 `docs/final/final_report_master_v2.md` → `docs/archive/`。
2. 重写 `docs/final/final_report_master.md` 为唯一 12 节主报告母版：摘要 + 1 基本信息（脱敏占位）+ 2 预期目标与实际完成情况 + 3 参考论文方法与实现差异 + 4 实施方案设计 + 5 并行算法设计 + 6 实施过程中解决的问题（≥9 类，五列表）+ 7 实验设计 + 8 实验结果与分析（每图必有机制级分析）+ 9 与论文结果对比 + 10 工程难度与完成质量 + 11 局限性 + 12 总结 + 附录 A 个人报告。采用“结论 → 方案 → 数据 → 分析 → 局限”叙事，修正定向增强表数据，固定 fig01–fig09 顺序。
3. 由 master 派生 `final_report_course.md`（真实姓名学号）与 `final_report_public.md`（Redacted）。
4. 同步重生成 `submission/course/final_report.md` 与 `submission/public/final_report_public.md`（图片用 `figures/...` 路径）。
5. 统一检查脚本到新结构：`check_report_format.py` 单一 12 节 `REQUIRED_HEADINGS`；`check_report_assets.py` 的 `DEFAULT_REPORTS` 去掉 master_v2、`REQUIRED_CONTENT` 改为新章节关键词。
6. 更新 `README.md`、`REPORT_MANIFEST.md`、`PROJECT_STRUCTURE.md` 的入口与命令为 `final_report_master.md`。
7. 运行 `check_privacy_and_encoding.py`、`check_report_assets.py`、`check_report_format.py`（master/course/public）、`check_final_submission.py` 与 `ctest`，失败即修复重跑。
