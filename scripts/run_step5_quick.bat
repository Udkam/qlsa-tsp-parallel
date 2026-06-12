@echo off
setlocal
cd /d "%~dp0\.."
py scripts\run_step5_experiments.py --quick
