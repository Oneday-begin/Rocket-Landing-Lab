@echo off
cd /d "%~dp0"
powershell -NoProfile -Command "$report = Get-ChildItem -LiteralPath 'artifacts' -Filter 'quality-gate.html' -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($report) { Start-Process -FilePath $report.FullName } else { Write-Host 'Run the quality check first.'; Read-Host 'Press Enter' }"
