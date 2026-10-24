# Publishing the phone interview demo

The main branch contains the full Python application. The gh-pages branch serves a read-only export at https://oneday-begin.github.io/Rocket-Landing-Lab/.

From PowerShell in the project folder, first run tests, then create and scan an export:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
powershell -ExecutionPolicy Bypass -File .\deployment\deploy_phone_demo.ps1
```

Review the export in `.cache/phone-demo/export`. To upload it after reviewing:

```powershell
powershell -ExecutionPolicy Bypass -File .\deployment\deploy_phone_demo.ps1 -Publish
```

The script requires Gitleaks. It accepts `-GitleaksPath` or uses `.cache/security/gitleaks.exe` or an installed `gitleaks` command. Download the scanner only from https://github.com/gitleaks/gitleaks/releases and verify its SHA-256 against the matching official checksum list. The checked local scanner is kept in the ignored cache, never committed.

The publisher blocks recognizable secret patterns, credential filenames, unexpected output paths and symlinks. Gitleaks must also succeed. It uses your existing Git sign-in and never reads or saves a token itself. It updates only gh-pages without force-pushing and preserves the full source branch. A scanner lowers risk; a clean scan cannot prove that every possible secret is absent.

If a push fails, review the ignored deployment checkout before retrying. Do not delete unreviewed work or force-push. If an actual secret is found in history, revoke/rotate it first, then plan history cleanup; deleting a current file does not undo exposure.
