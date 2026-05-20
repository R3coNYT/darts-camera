# DartsCamera - Installation (Windows PowerShell)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== DartsCamera - Installation ===" -ForegroundColor Cyan

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python 3 est requis. Telechargez-le sur https://python.org"
    exit 1
}

# Create venv
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "Environnement virtuel cree (.venv)" -ForegroundColor Green
}

# Activate & install
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "=== Installation terminee ===" -ForegroundColor Green
Write-Host ""
Write-Host "Lancement :"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python app.py"
Write-Host ""
Write-Host "Ouvrez http://localhost:5000 dans votre navigateur."
