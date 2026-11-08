@echo off
setlocal
cd /d "%~dp0"
echo =====================================================
echo AI Cognitive Rehabilitation System - Windows Setup
echo =====================================================
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher was not found. Install Python 3.11 64-bit and select Add Python to PATH.
  pause
  exit /b 1
)
py -3.11 -m venv .venv
if errorlevel 1 goto failed
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto failed
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo Downloading FaceNet VGGFace2 model into models\facenet ...
python download_facenet_model.py
if errorlevel 1 goto failed
echo.
echo Testing FaceNet model...
python test_facenet.py
if errorlevel 1 goto failed
echo.
echo Setup completed. Run run_windows.bat to start the application.
pause
exit /b 0
:failed
echo.
echo Setup failed. Read the error above, check your internet connection, and run setup_windows.bat again.
pause
exit /b 1
