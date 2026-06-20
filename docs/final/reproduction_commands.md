# 复现命令

以下命令均在项目根目录执行。Windows 环境建议使用 `py` 启动 Python 脚本；如果本机 `python` 命令可用，也可以替换为 `python`。

## 构建与测试

```bat
cmake -S . -B build-cuda-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DTSP_ENABLE_CUDA=ON
cmake --build build-cuda-ninja -j
ctest --test-dir build-cuda-ninja --output-on-failure
```

## 默认参数多实例实验

```bat
py scripts\run_step5_experiments.py --instances berlin52 eil51 st70 eil76 rat99 eil101 --iterations 1000000 --repeat 3 --chains 32 --threads 8 --no-cuda --output results\raw\step5_multi_cpu_raw.csv
```

## 调优参数独立验证

```bat
scripts\run_tuned_validation.bat
```

## 定向增强实验

```bat
scripts\run_targeted_quality.bat
```

## 图表生成

```bat
py scripts\make_report_figures.py
```

## 提交前检查

```bat
py scripts\check_privacy_and_encoding.py
py scripts\check_report_assets.py
py scripts\check_report_format.py docs\final\final_report_master.md
py scripts\check_report_format.py docs\final\final_report_course.md
py scripts\check_report_format.py docs\final\final_report_public.md
py scripts\check_final_submission.py
ctest --test-dir build-cuda-ninja --output-on-failure
```
