@echo off
setlocal
set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PROJECT_ROOT=%~dp0.."
set "VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" python -m venv "%PROJECT_ROOT%\.venv"
if errorlevel 1 exit /b %errorlevel%

"%VENV_PYTHON%" -m pip install -r "%PROJECT_ROOT%\requirements.txt"
if errorlevel 1 exit /b %errorlevel%

"%VENV_PYTHON%" -X utf8 "%PROJECT_ROOT%\scripts\verify_setup.py"
exit /b %errorlevel%
