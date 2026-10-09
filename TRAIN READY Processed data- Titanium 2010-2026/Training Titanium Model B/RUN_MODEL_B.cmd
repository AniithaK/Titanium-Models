@echo off
setlocal
cd /d "%~dp0"
python "code\prepare_titanium_model_b_data.py"
if errorlevel 1 exit /b %errorlevel%
python "code\train_titanium_model_b.py"
if errorlevel 1 exit /b %errorlevel%
echo.
echo Titanium Model B completed successfully.
echo Dashboard: %CD%\visualizations\titanium_model_b_dashboard.html
endlocal
