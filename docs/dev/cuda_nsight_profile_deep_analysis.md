# CUDA Nsight profiling 深化记录

## 配置

- 实例：`a280`
- 迭代次数：`100000`
- 搜索链：`64`
- CUDA block：`128`
- 候选数：`128`
- 候选策略：`hybrid`
- 路径反转：`parallel`

## Nsight Systems

当前仓库路径包含中文目录，Nsight Systems 直接在项目目录下运行时会出现 `bad conversion` 或 UTF-8 序列化错误。为排除路径编码干扰，已将 `tsp_sa.exe` 和 `a280.tsp` 临时复制到 `C:\pa_nsys_stage` 后运行。

执行结果：

- Nsight Systems 运行成功。
- 已生成：
  - `results/logs/nsight/cuda_candidate_a280_hybrid_nsys.nsys-rep`
  - `results/logs/nsight/cuda_candidate_a280_hybrid_nsys.sqlite`
  - `results/logs/nsight/cuda_candidate_a280_hybrid_nsys_ascii.log`

摘要信息：

- `cudaDeviceSynchronize` 约占 CUDA API 时间的 79.6%，总计约 637 ms。
- `cudaMalloc` 约占 19.9%，三次调用中有一次较长。
- `cudaLaunchKernel` 约 3.16 ms。
- Host-to-Device 传输约 0.314 MB，Device-to-Host 传输约 0.074 MB。

这些结果说明本次 a280 candidate-hybrid 运行中，主时间不在数据传输规模，而在 kernel 执行等待和设备端计算/同步。它与前面的 CSV 实验结论一致：候选批量评价改善搜索质量，但路径操作、块内同步和设备端执行时间仍是主要代价。

## Nsight Compute

Nsight Compute 已找到：

```text
C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\ncu.BAT
```

使用 `--set basic` 运行时，工具能连接到进程，但驱动拒绝访问 NVIDIA GPU Performance Counters：

```text
ERR_NVGPUCTRPERM
```

因此本轮不报告 occupancy、SM 利用率、内存吞吐等硬件计数器指标。若后续需要这些指标，需要在 NVIDIA 控制面板或驱动权限设置中允许当前用户访问 GPU performance counters，再重新运行相同命令。

## 结论边界

- 可以写 Nsight Systems 已在 ASCII 临时路径下捕获 CUDA candidate 运行，并显示数据传输量很小、时间主要集中在设备同步等待。
- 可以写 Nsight Compute 已安装，但硬件性能计数器权限阻止采集 occupancy 和带宽指标。
- 不能写 occupancy、带宽、SM 利用率等具体数值。
- 不能据此写 CUDA 性能优于 OpenMP。
