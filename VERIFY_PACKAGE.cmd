@echo off
setlocal
cd /d "%~dp0"
python verify_package.py
if errorlevel 1 exit /b %errorlevel%
endlocal
