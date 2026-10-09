@echo off
setlocal
cd /d "%~dp0"
python code\generate_model_a_response.py
exit /b %errorlevel%
