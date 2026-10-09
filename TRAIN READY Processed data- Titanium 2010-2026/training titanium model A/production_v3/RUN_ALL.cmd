@echo off
setlocal
cd /d "%~dp0"
call RUN_PRODUCTION_V3.cmd
if errorlevel 1 exit /b %errorlevel%
call RUN_PREDICTION.cmd
exit /b %errorlevel%
