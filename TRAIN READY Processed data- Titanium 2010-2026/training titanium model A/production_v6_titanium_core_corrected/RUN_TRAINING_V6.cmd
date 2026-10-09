@echo off
setlocal
cd /d "%~dp0"
python "code\train_model_a_production_v6_titanium_core_corrected.py"
if errorlevel 1 exit /b %errorlevel%
endlocal
