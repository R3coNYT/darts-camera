#Requires -Version 5.1
<#
.SYNOPSIS
    DartsCamera - Update Script (Windows)
.DESCRIPTION
    Tire le dernier code depuis GitHub, preserve config.yaml,
    et met a jour les dependances Python dans le venv existant.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -- Console helpers ------------------------------------------------------------

function Write-Log  { param([string]$Msg) Write-Host "[+] $Msg" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Msg) Write-Host "[OK] $Msg" -ForegroundColor Green }
function Write-Warn { param([string]$Msg) Write-Host "[!] $Msg" -ForegroundColor Yellow }
function Write-Err  { param([string]$Msg) Write-Host "[X] $Msg" -ForegroundColor Red }
function Write-Info { param([string]$Msg) Write-Host "[i] $Msg" -ForegroundColor Gray }

# -- Locate install directory ---------------------------------------------------

function Find-InstallDir {
    $ScriptDir = Split-Path -Parent $MyInvocation.ScriptName
    if ($ScriptDir -and (Test-Path "$ScriptDir\app.py") -and (Test-Path "$ScriptDir\.git")) {
        return $ScriptDir
    }
    if ($PSScriptRoot -and (Test-Path "$PSScriptRoot\app.py") -and (Test-Path "$PSScriptRoot\.git")) {
        return $PSScriptRoot
    }
    $Cwd = Get-Location
    if ((Test-Path "$Cwd\app.py") -and (Test-Path "$Cwd\.git")) {
        return $Cwd
    }
    Write-Err "Impossible de localiser le dossier DartsCamera."
    Write-Err "Lancez ce script depuis le dossier d'installation."
    exit 1
}

# -- Backup config --------------------------------------------------------------

function Backup-Config {
    param([string]$InstallDir)

    $Timestamp  = Get-Date -Format "yyyyMMdd_HHmmss"
    $BackupDir  = Join-Path $InstallDir "backups\$Timestamp"

    Write-Log "Sauvegarde de la config -> $BackupDir"
    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null

    $ConfigPath = Join-Path $InstallDir "config.yaml"
    if (Test-Path $ConfigPath) {
        Copy-Item $ConfigPath -Destination $BackupDir
        Write-Ok "config.yaml sauvegarde"
    } else {
        Write-Warn "Aucun config.yaml trouve - rien a sauvegarder"
    }

    $EnvPath = Join-Path $InstallDir ".env"
    if (Test-Path $EnvPath) {
        Copy-Item $EnvPath -Destination $BackupDir
        Write-Ok ".env sauvegarde"
    }

    Write-Ok "Sauvegarde terminee -> $BackupDir"
    return $BackupDir
}

# -- Git pull -------------------------------------------------------------------

function Update-Git {
    param([string]$InstallDir, [string]$BackupDir)

    Write-Log "Recuperation du dernier code depuis GitHub"
    Set-Location $InstallDir

    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Err "git n'est pas installe ou absent du PATH"
        Write-Err "Telechargez Git : https://git-scm.com/download/win"
        exit 1
    }

    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    $StashResult = (git stash --quiet 2>&1) -join " "
    if ($StashResult -and $StashResult -notmatch "No local changes") {
        Write-Info "Modifications locales mises en stash : $StashResult"
    }

    git fetch origin 2>&1 | Out-Null

    $Branch = (git rev-parse --abbrev-ref HEAD 2>&1) -join ""
    git reset --hard "origin/$Branch" 2>&1 | Out-Null

    $ErrorActionPreference = $prev

    # Restaurer config.yaml depuis la sauvegarde (ne pas l'ecraser avec la version depot)
    $ConfigBackup = Join-Path $BackupDir "config.yaml"
    $ConfigTarget = Join-Path $InstallDir "config.yaml"
    if (Test-Path $ConfigBackup) {
        Copy-Item $ConfigBackup -Destination $ConfigTarget -Force
        Write-Ok "config.yaml restaure (vos reglages sont preserves)"
    }

    $EnvBackup = Join-Path $BackupDir ".env"
    $EnvTarget = Join-Path $InstallDir ".env"
    if (Test-Path $EnvBackup) {
        Copy-Item $EnvBackup -Destination $EnvTarget -Force
        Write-Ok ".env restaure"
    }

    $NewCommit = (git rev-parse --short HEAD 2>&1) -join ""
    Write-Ok "Code mis a jour -> commit $NewCommit (branche : $Branch)"
}

# -- Update Python deps ---------------------------------------------------------

function Update-Deps {
    param([string]$InstallDir)

    Write-Log "Mise a jour des dependances Python"

    $VenvPip = Join-Path $InstallDir ".venv\Scripts\pip.exe"
    if (-not (Test-Path $VenvPip)) {
        Write-Warn "Venv introuvable (.venv). Creation en cours..."
        Set-Location $InstallDir
        python -m venv .venv
        Write-Ok "Venv cree"
        $VenvPip = Join-Path $InstallDir ".venv\Scripts\pip.exe"
    }

    & $VenvPip install --upgrade pip --quiet
    & $VenvPip install -r (Join-Path $InstallDir "requirements.txt") --quiet
    Write-Ok "Dependances Python a jour"
}

# -- Purge old backups (keep last 5) --------------------------------------------

function Prune-Backups {
    param([string]$InstallDir)

    $BackupRoot = Join-Path $InstallDir "backups"
    if (-not (Test-Path $BackupRoot)) { return }

    $Dirs = Get-ChildItem $BackupRoot -Directory | Sort-Object Name
    if ($Dirs.Count -gt 5) {
        $ToDelete = $Dirs | Select-Object -First ($Dirs.Count - 5)
        foreach ($d in $ToDelete) {
            Remove-Item $d.FullName -Recurse -Force
            Write-Info "Ancienne sauvegarde supprimee : $($d.Name)"
        }
    }
}

# -- Main -----------------------------------------------------------------------

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "   DartsCamera - Mise a jour" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

$InstallDir = Find-InstallDir
Write-Info "Dossier d'installation : $InstallDir"

$BackupDir  = Backup-Config $InstallDir
Update-Git      $InstallDir $BackupDir
Update-Deps     $InstallDir
Prune-Backups   $InstallDir

Write-Host ""
Write-Host "====================================" -ForegroundColor Green
Write-Host "   Mise a jour terminee !" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""
Write-Info "Pour lancer l'application :"
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python app.py" -ForegroundColor White
Write-Host ""
