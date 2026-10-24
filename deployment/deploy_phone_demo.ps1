param([switch]$Publish,[string]$GitleaksPath='')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$exportPath = Join-Path $projectRoot '.cache\phone-demo\export'
$publishPath = Join-Path $projectRoot '.cache\phone-demo\publish'
$remoteUrl = 'https://github.com/Oneday-begin/Rocket-Landing-Lab.git'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Run setup.ps1 first.' }
if (-not $GitleaksPath) {
    $localScanner = Join-Path $projectRoot '.cache\security\gitleaks.exe'
    $installedScanner = Get-Command gitleaks -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $localScanner) { $GitleaksPath = $localScanner }
    elseif ($installedScanner) { $GitleaksPath = $installedScanner.Source }
    else { throw 'Install Gitleaks from its official release, or pass -GitleaksPath. Publishing is blocked until the scanner is available.' }
}
function Check-Exit([string]$Message) { if ($LASTEXITCODE -ne 0) { throw $Message } }
& $projectPython -X utf8 (Join-Path $PSScriptRoot 'build_phone_demo.py') --project-root $projectRoot --output $exportPath
Check-Exit 'Export failed.'
& $projectPython -X utf8 (Join-Path $PSScriptRoot 'check_publish.py') $exportPath
Check-Exit 'Export security check failed.'
& $GitleaksPath dir $exportPath --redact=100 --ignore-gitleaks-allow --max-archive-depth=3 --no-banner --no-color
Check-Exit 'Secret scanner failed or detected a possible secret. Nothing was published.'
if (-not $Publish) {
    Write-Host "Checked export ready at $exportPath. Run again with -Publish after reviewing it."
    exit 0
}
if (-not (Test-Path -LiteralPath $publishPath)) {
    & git clone --single-branch --branch gh-pages $remoteUrl $publishPath
    Check-Exit 'Could not open the deployment branch.'
}
$configuredRemote = & git -C $publishPath remote get-url origin
Check-Exit 'Could not inspect deployment remote.'
$configuredBranch = & git -C $publishPath branch --show-current
if ($configuredRemote -ne $remoteUrl -or $configuredBranch -ne 'gh-pages') { throw 'Deployment checkout must use the expected repository and gh-pages branch.' }
$pending = & git -C $publishPath status --porcelain
if ($pending) { throw 'The deployment checkout has unfinished changes. Review them before retrying.' }
& git -C $publishPath pull --ff-only origin gh-pages
Check-Exit 'Deployment branch changed; inspect it before retrying.'
Get-ChildItem -LiteralPath $exportPath -Force | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $publishPath -Recurse -Force }
& $projectPython -X utf8 (Join-Path $PSScriptRoot 'check_publish.py') $publishPath
Check-Exit 'Deployment checkout failed its security check.'
& $GitleaksPath dir $publishPath --redact=100 --ignore-gitleaks-allow --max-archive-depth=3 --no-banner --no-color
Check-Exit 'Secret scanner failed. Nothing was published.'
& git -C $publishPath add -- .gitattributes .nojekyll README.md index.html assets data docs
Check-Exit 'Could not stage the demo.'
& git -C $publishPath diff --cached --quiet
if ($LASTEXITCODE -eq 0) { Write-Host 'The published demo already matches this export.'; exit 0 }
if ($LASTEXITCODE -ne 1) { throw 'Could not inspect staged changes.' }
& git -C $publishPath -c user.name=Oneday-begin -c user.email=189585052+Oneday-begin@users.noreply.github.com commit -m 'Update checked phone interview demo'
Check-Exit 'Could not save deployment changes.'
& git -C $publishPath push origin gh-pages
Check-Exit 'Push failed. Inspect the deployment checkout before retrying.'
Write-Host 'Uploaded. Wait for GitHub Pages deployment to finish, then verify https://oneday-begin.github.io/Rocket-Landing-Lab/'
