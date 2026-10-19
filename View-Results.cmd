@echo off
cd /d "%~dp0"
if exist "artifacts\report.html" (
    start "" "artifacts\report.html"
) else (
    echo Results are not ready yet. See README.md to generate results.
    pause
)
