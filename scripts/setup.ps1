$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    python -m venv (Join-Path $ProjectRoot ".venv")
}

& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
& $VenvPython (Join-Path $ProjectRoot "scripts\verify_setup.py")
