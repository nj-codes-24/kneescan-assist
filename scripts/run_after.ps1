# run_after.ps1: wait until a file exists, then start one training config (a one-line job queue for overnight runs).
#
# What:  checks every 2 minutes for -WaitFor (e.g. the last fold's done.json of the previous run), then starts
#        scripts\train_reader_cv.py with -Config (and optionally only -Folds) as its own detached process,
#        logging to logs\<name>.out.log / .err.log.
# Why:   the GPU can only run a few big jobs well at a time. This starts the next experiment the moment the previous
#        one ends, even if no one is watching. It's resumable: if the PC restarts, run this again (training resumes from last.pt).
# Alternatives: Windows Task Scheduler or a proper job queue (e.g. Slurm on a cluster). Overkill for one PC.
#
# Usage:
#   powershell -File scripts\run_after.ps1 -WaitFor runs\reader_r18_fbn\fold4\done.json -Config configs\reader_cnxt.yaml
#   powershell -File scripts\run_after.ps1 -WaitFor <file> -Config <yaml> -Folds 1      # one fold only (a quick screen)
param(
    [Parameter(Mandatory = $true)][string]$WaitFor,
    [Parameter(Mandatory = $true)][string]$Config,
    [string]$Folds = ""                         # empty = all 5 folds
)
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
while (-not (Test-Path $WaitFor)) { Start-Sleep -Seconds 120 }
$name = [System.IO.Path]::GetFileNameWithoutExtension($Config)
$suffix = if ($Folds -ne "") { "_fold$Folds" } else { "" }
$arguments = @("-u", "scripts\train_reader_cv.py", "--config", $Config)
if ($Folds -ne "") { $arguments += @("--folds", $Folds) }
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
Start-Process -FilePath "$repo\.venv\Scripts\python.exe" -ArgumentList $arguments `
    -WorkingDirectory $repo -WindowStyle Hidden `
    -RedirectStandardOutput "$repo\logs\$name$suffix.out.log" -RedirectStandardError "$repo\logs\$name$suffix.err.log"
