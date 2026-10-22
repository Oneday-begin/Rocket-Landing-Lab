@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.ps1 first. See README.md for setup instructions.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m rocket_landing.webapp --open-browser
if errorlevel 1 pause
