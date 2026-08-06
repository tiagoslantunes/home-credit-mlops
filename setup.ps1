# One-shot setup script for the home-credit-mlops project (Windows / PowerShell).
#
# Creates a fresh virtual environment, installs every dependency, and verifies
# that the core imports work. Run from the project root:
#
#     .\setup.ps1
#
# Re-running the script is safe: it will REPLACE the existing .venv.

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "[1/5] Creating fresh virtual environment in .venv ..." -ForegroundColor Cyan
if (Test-Path ".venv") {
    Write-Host "      Removing existing .venv ..." -ForegroundColor DarkGray
    try {
        Remove-Item -Recurse -Force .venv -ErrorAction Stop
    }
    catch {
        Write-Host ""
        Write-Host "Could not delete .venv - a process is holding a file open." -ForegroundColor Red
        Write-Host "Close any open Python / Jupyter / uvicorn process and your IDE" -ForegroundColor Red
        Write-Host "Python interpreter (PyCharm: File > Invalidate Caches; VS Code:" -ForegroundColor Red
        Write-Host "kill the Python language server), then run setup.ps1 again." -ForegroundColor Red
        Write-Host ""
        Write-Host "If the .venv was already populated by a previous successful run," -ForegroundColor Yellow
        Write-Host "you can just activate it instead of rebuilding:" -ForegroundColor Yellow
        Write-Host "  . .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
        exit 1
    }
}
python -m venv .venv

Write-Host "[2/5] Activating .venv ..." -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1

Write-Host "[3/5] Upgrading pip ..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

Write-Host "[4/5] Installing dependencies (this takes ~5 minutes) ..." -ForegroundColor Cyan
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"

Write-Host "[5/5] Verifying imports ..." -ForegroundColor Cyan
python -c "import pandas, sklearn, lightgbm, kedro, fastapi, evidently, shap, mlflow, optuna, great_expectations; print('core stack ok')"

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "  1. Drop the 8 Kaggle CSVs into data/01_raw/"
Write-Host '  2. $env:MPLBACKEND="Agg"  (matplotlib backend)'
Write-Host "  3. kedro run                (runs the full pipeline)"
Write-Host "  4. uvicorn app.main:app --port 8000   (start the API)"
Write-Host "  5. Optional Hopsworks client: create a separate environment and install requirements-feature-store.txt"
