# 已知限制

1. CUDA 后端已完成构建和运行验证，但当前小规模 TSPLIB95 实例上不优于 OpenMP，因此不作为主要加速结论。
2. 当前 QLSA 是基于论文思想的工程化变体，不是完整复刻论文 SB-QLSA candidate-leader 与 diversity-state 机制。
3. policy comparison 比较的是本实现中的 epsilon-greedy 与 Softmax，不等同于论文 Softmax 实验。
4. 与论文 Table 8 的运行时间对比涉及不同硬件、不同语言和不同实现，只能作为参考对比。
5. 预算扫描曲线不是逐迭代 trace，不能表述为真实逐迭代收敛曲线。
6. 实验实例主要集中在中小规模 TSPLIB95，CUDA 和更大规模实例仍需进一步验证。
