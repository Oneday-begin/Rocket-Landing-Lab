@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Run setup.ps1 first. See README.md.
    pause
    exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" -m rocket_landing.panel
