# 已知限制与结论边界

本文档记录最终课程报告中必须保持克制的结论边界。它用于防止把工程证据、补充实验或不同平台对比误写成过强结论。

## 1. OpenMP 是主要性能结论

默认参数多实例实验表明，OpenMP multi-chain 是当前最稳定、最可解释的性能提升来源。报告可以写 OpenMP 在多个 TSPLIB95 实例上取得约 5x 的平均加速，并且并行方式不改变 SA/QLSA 的核心接受准则。

不能写成所有后端都优于 OpenMP。CUDA 与 MPI 结果主要用于说明工程扩展和不同并行层次的可运行性。

## 2. CUDA 后端的边界

CUDA chain mode、SA/QLSA candidate mode、candidate policy 和 parallel reversal 已完成，并能通过测试。candidate mode 使用 block 内线程并行评价多个 2-opt 候选 move，提高了 GPU 内部计算密度，也在部分实例上改善了解质量。candidate policy 当前支持 `best`、`random` 和 `hybrid`：`best` 选择候选批中的最小 delta，`random` 按可复现随机方式从候选批中选择一个候选，`hybrid` 在两者之间交替。

该模式不是默认后端，也不等同于原始 SA 的单候选 proposal。由于每轮 proposal 从单候选变为 batch candidate proposal，报告中必须单独标记为 CUDA candidate-level evaluation。

当前实验不支持“CUDA 比 OpenMP 更快”的结论。candidate mode 在小中规模实例上仍受同步、shared memory reduction、tour reversal、global memory 和 kernel overhead 限制。`random` 和 `hybrid` policy 能提供候选选择策略对照，但没有改变 CUDA 的整体定位。CUDA 可写为工程扩展和质量探索，不应写成主性能结果。

## 3. QLSA 与论文 SB-QLSA 的差异

C++ 主线实现了 QLSA 的工程化版本，支持状态/动作离散化、Q table、epsilon-greedy、softmax，以及 CUDA QLSA candidate 路径。可选 `--qlsa_variant paper` 和 `--qlsa_variant paper-sb` 保留 candidate-leader 与两阶段 Metropolis 接受流程；`paper-sb` 支持 `--diversity_metric hamming|edge`。`hamming` 保持论文的 position-Hamming 定义，`edge`（默认）以无向回路边集差异表示多样性，因此对对称 TSP 中的循环移位和反向表示不敏感。

不能把历史 Step 5/6 主实验改写成 paper-sb 结果。已有默认参数、调优和定向增强结论仍主要来自 `current` 变体；历史 `paper-sb` CSV 使用 Hamming metric，复跑时配置已显式固定为 `hamming`，不得与新的 `edge` metric 样本合并。CUDA QLSA 当前仍只支持 `--qlsa_variant current`。

为兼容既有严格流水线，主程序的 28 列 CSV 不新增 metric 字段；使用 `--csv-only` 收集的 `paper-sb` 原始行必须与命令日志、配置快照或 manifest 一起保存。公平/island runner 和 variant runner 已分别保存这些 provenance，脱离这些工件手工合并 CSV 不具备 metric 可比性。

QLSA 在 rat99 等部分实例上显示出更好的解质量，但不能写成 QLSA 总是优于 SA。

## 4. MPI + OpenMP 的边界

MPI + OpenMP hybrid 后端已在两台 Ubuntu VM 上通过真实 `mpirun` 完成 smoke、formal scaling 和 large quick/formal subset。它证明了 rank-level chain decomposition、rank 内 OpenMP、rank 间 reduction/gather 的工程链路。

该实验环境是 VMware NAT 双 VM，不是生产 HPC 集群。报告可以写“真实双 VM MPI 运行证据”和“分布式内存扩展路径可行”，不能写成生产级 HPC benchmark。

SA/QLSA 已提供可暂停并继续的 chunked search state，单机 OpenMP 已实现 `independent`、`ring`、`global` 三种 island 拓扑，并在同步 chunk 边界完成迁移。当前正式消融只覆盖 eil76，且 SA 在 10k 周期下的改善趋势经过 Holm 校正后未达到 0.05，因此不能写成普遍显著优势。

MPI 跨节点 island migration 仍未完成。当前迁移协调器依赖单进程共享内存和 OpenMP 同步边界，不能把单机结果写成分布式迁移证据。

## 5. 大实例实验的边界

大实例实验用于验证工程可扩展性、运行时间趋势、Gap 趋势和后端边界。L1/L2/L3 的目标不是证明所有实例达到 BKS。

当前结果支持“130-280 城市 L1 formal 可运行”“300-600 城市 L2 subset 可运行”“约 1000 城市 L3 quick 可运行”这类工程结论。不能写成“百万城市规模”或“所有大实例均达到最优”。

## 6. 与论文对比的边界

参考论文使用 Python + Xeon 平台，本项目使用 Windows + C++20 + OpenMP/CUDA/MPI。语言、硬件和实现均不同，所以论文 Table 8 时间只能作为参考对比，不是同平台公平 benchmark。

可以写“在共同 TSPLIB95 实例和共同 BKS 指标下，本工程展示了 C++ 化和并行化后的时间优势与质量竞争力”。不能写“严格全面超过论文所有性能”。

## 7. 输入问题域边界

当前 2-opt 增量公式只支持对称 TSP。解析器会拒绝 `TYPE: ATSP`，并拒绝非对称的 `EXPLICIT/FULL_MATRIX`；手工构造 `Instance` 时 `DistanceMatrix` 也会重复校验。若需要支持 ATSP，必须实现有向 2-opt 的完整增量公式和对应的 CUDA/MPI 路径，不能绕过此保护。

## 8. 数据与提交边界

报告结论必须追溯到 `results/final/`、`results/summary/`、`results/reference/` 中的 CSV 或最终图表。公平配对与 island 结果分别由 `fair_paired_eil76_*.csv` 和 `island_eil76_*.csv` 提供正式摘要。`results/logs/`、构建目录、VM 私钥、VM hostfile 和浏览器/GPT 临时材料不应提交。

本轮修复使多链 `--init nn` 按链 seed 选择起点，并把 `paper-sb` 默认多样性度量改为 `edge`。因此既有多链 NN 或未显式记录 metric 的历史样本不能直接视为新代码的可复现基线；复用旧结论时必须使用保存的历史配置（其中公平/island 配置已固定为 `hamming`），需要比较新默认行为时应重新运行完整 paired-seed 实验。

最终课程提交文档只保留两份：`docs/final/report.md` 和 `docs/final/personal_report.md`。历史提交包副本已删除，避免多个报告版本并存。
