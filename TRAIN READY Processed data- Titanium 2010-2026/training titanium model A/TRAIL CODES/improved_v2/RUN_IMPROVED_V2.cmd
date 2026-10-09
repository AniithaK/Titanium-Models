@echo off
setlocal
cd /d "%~dp0"
python prepare_improved_training_data.py
if errorlevel 1 exit /b %errorlevel%
python code\train_titanium_model_a_improved_v2.py
exit /b %errorlevel%
