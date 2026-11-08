@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Setup has not been run. Please execute setup_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python app.py
pause
