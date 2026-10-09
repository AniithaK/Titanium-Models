@echo off
setlocal
python "%~dp0code\train_titanium_model_a_eruc.py"
if errorlevel 1 exit /b %errorlevel%
echo.
echo Training complete. Open visualizations\titanium_model_a_eruc_dashboard.html
