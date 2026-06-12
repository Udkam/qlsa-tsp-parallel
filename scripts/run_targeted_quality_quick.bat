@echo off
setlocal
cd /d "%~dp0.."

py scripts/run_targeted_quality.py --quick --output results/targeted_quality_raw.csv
if errorlevel 1 exit /b %errorlevel%

py scripts/analyze_targeted_quality.py --input results/targeted_quality_raw.csv
if errorlevel 1 exit /b %errorlevel%

echo Targeted quality quick summary: results\targeted_quality_summary.csv
echo Targeted quality quick analysis: docs\step6C_targeted_quality_analysis.md
