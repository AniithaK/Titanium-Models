@echo off
setlocal
cd /d "%~dp0"
python "code\train_model_a_production_v5_titanium_core_optimized.py"
if errorlevel 1 exit /b %errorlevel%
endlocal
