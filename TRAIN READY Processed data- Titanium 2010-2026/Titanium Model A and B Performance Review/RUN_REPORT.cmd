@echo off
python "%~dp0code\build_combined_performance_dashboard.py"
if errorlevel 1 exit /b %errorlevel%
echo.
echo Combined dashboard generated successfully.
