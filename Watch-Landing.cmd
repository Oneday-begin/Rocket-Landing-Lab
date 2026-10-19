@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.ps1 first. See README.md.
    pause
    exit /b 1
)
set "ROCKET_MODEL=runs\baseline\seed_0\final_model.zip"
if not exist "%ROCKET_MODEL%" set "ROCKET_MODEL=runs\baseline\seed_0\best_model.zip"
if not exist "%ROCKET_MODEL%" (
    echo Baseline model missing. See README.md for training instructions.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m rocket_landing watch --model "%ROCKET_MODEL%" --human
if errorlevel 1 pause
