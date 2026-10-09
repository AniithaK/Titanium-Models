@echo off
setlocal
cd /d "%~dp0"
python prepare_model_a_v3_data.py
if errorlevel 1 exit /b %errorlevel%
python code\train_model_a_production_v3.py
exit /b %errorlevel%
