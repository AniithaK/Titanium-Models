@echo off
setlocal
cd /d "%~dp0"
python run_prepare_improved.py
if errorlevel 1 exit /b %errorlevel%
python run_training_improved.py
exit /b %errorlevel%
