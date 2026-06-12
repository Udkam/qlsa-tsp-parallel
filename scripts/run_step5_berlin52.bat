@echo off
setlocal
cd /d "%~dp0\.."
py scripts\run_step5_experiments.py --instances berlin52 --iterations 1000000 --repeat 3 --chains 32 --threads 8 --cuda-block-size 128 --output results\step5_berlin52_raw.csv
echo.
echo Step 5 berlin52 summary: results\step5_berlin52_summary.csv
echo Step 5 berlin52 analysis: docs\step5_berlin52_analysis.md
