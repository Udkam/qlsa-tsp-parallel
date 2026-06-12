@echo off
setlocal
cd /d "%~dp0.."

py scripts/run_tuned_validation.py --repeat 10 --seed 101 --output results/tuned_validation_raw.csv
if errorlevel 1 exit /b %errorlevel%

py scripts/analyze_tuned_validation.py --input results/tuned_validation_raw.csv
if errorlevel 1 exit /b %errorlevel%

echo Tuned validation summary: results\tuned_validation_summary.csv
echo Tuned validation analysis: docs\step6B_tuned_validation_analysis.md
