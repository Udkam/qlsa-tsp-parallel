@echo off
setlocal
cd /d "%~dp0.."

py scripts/run_tuned_validation.py --quick --output results/tuned_validation_raw.csv
if errorlevel 1 exit /b %errorlevel%

py scripts/analyze_tuned_validation.py --input results/tuned_validation_raw.csv
if errorlevel 1 exit /b %errorlevel%

echo Tuned validation quick summary: results\tuned_validation_summary.csv
echo Tuned validation quick analysis: docs\step6B_tuned_validation_analysis.md
