@echo off
cd /d "%~dp0"
powershell -NoProfile -Command "$r = Get-ChildItem -LiteralPath artifacts -Filter weather-comparison.png -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($r) { Start-Process (Join-Path $r.DirectoryName report.html) }"
