# a280 CUDA candidate-hybrid Nsight 摘要

## 配置

- 实例：a280
- 迭代次数：100000
- 搜索链：64
- CUDA block：128
- 每轮候选数：128
- 候选策略：hybrid
- 路径反转：parallel
- Nsight Systems：2025.1.3
- Nsight Compute：2025.2.0

## Nsight Systems 结果

| CUDA API / 传输 | 总时间 | 次数 | CUDA API 时间占比 |
|---|---:|---:|---:|
| `cudaDeviceSynchronize` | 637.331 ms | 1 | 79.6% |
| `cudaMalloc` | 158.876 ms | 3 | 19.9% |
| `cudaLaunchKernel` | 3.161 ms | 1 | 0.4% |
| `cudaMemcpy` | 0.439 ms | 3 | 0.1% |
| Host-to-Device | 0.0239 ms / 0.314 MB | 1 | — |
| Device-to-Host | 0.0071 ms / 0.074 MB | 2 | — |

本次运行的数据传输量较小，CUDA API 时间主要集中在设备同步等待和一次较长的设备内存分配。候选批量评价的主要代价来自设备端执行与同步，而不是主机—设备传输规模。

## 采集条件

项目路径包含中文字符时，Nsight Systems 的报告序列化出现编码错误；采集命令在 ASCII 临时目录执行，并将统计摘要归档到本文件。CPU context switch 与 sampling 需要管理员权限，因此本次 Systems 统计不包含这两类事件。

Nsight Compute 能连接目标进程，GPU performance counters 的访问被驱动以 `ERR_NVGPUCTRPERM` 拒绝。本次分析采用 Systems API/传输统计和正式 CSV 的运行时间、解质量数据。
