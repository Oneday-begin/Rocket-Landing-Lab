$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 or newer is required.' }
}
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $projectPython -m pip --isolated install --timeout 120 --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip.' }
& $projectPython -m pip --isolated install --timeout 120 swig
if ($LASTEXITCODE -ne 0) { throw 'Could not install SWIG.' }
$env:PATH = (Join-Path $PSScriptRoot '.venv\Scripts') + ';' + $env:PATH
& $projectPython -m pip --isolated install --timeout 120 -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install dependencies. See README troubleshooting.' }
& $projectPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Verification failed.' }
Write-Host 'Ready. Run: .\.venv\Scripts\python.exe -m rocket_landing experiment'
