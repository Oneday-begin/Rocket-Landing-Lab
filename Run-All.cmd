@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.ps1 first. See README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m rocket_landing experiment --skip-existing
if errorlevel 1 (
    echo The experiment stopped. Read the error above and README.md.
) else (
    start "" "artifacts\report.html"
)
pause
