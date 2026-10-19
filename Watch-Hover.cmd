@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.ps1 first. See README.md.
    pause
    exit /b 1
)
if not exist "runs\hover\seed_0\final_model.zip" (
    echo Hover model missing. See README.md for training instructions.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m rocket_landing watch --model runs\hover\seed_0\final_model.zip --human
if errorlevel 1 pause
