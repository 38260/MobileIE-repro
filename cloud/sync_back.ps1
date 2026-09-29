<#
.SYNOPSIS
  Pull the cloud run's checkpoints + logs back to the laptop on a timer, so a
  pre-empted or accidentally destroyed instance cannot cost you the training.
  Run it in a SECOND terminal on the laptop while the instance trains.

.USAGE
  powershell -File cloud\sync_back.ps1 -User root -Address region-x.autodl.com -Port 12345 `
           -RemoteDir /root/MobileIE -IntervalSec 300
  powershell -File cloud\sync_back.ps1 -User root -Address ... -Port ... -Once
#>
param(
    [Parameter(Mandatory)][string]$User,
    [Parameter(Mandatory)][string]$Address,
    [Parameter(Mandatory)][string]$Port,
    [string]$RemoteDir = "/root/MobileIE",
    [int]$IntervalSec = 300,
    [string]$LocalDir = "",
    [switch]$Once
)

$ErrorActionPreference = "Continue"
if (-not $LocalDir) { $LocalDir = Join-Path $PSScriptRoot "remote" }
New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
$dest = "$User@$Address"

function Pull([string]$rel) {
    $local = Join-Path $LocalDir ($rel -replace '/', '\')
    New-Item -ItemType Directory -Force -Path (Split-Path $local) | Out-Null
    Write-Host "  scp -P $Port -r ${dest}:$RemoteDir/$rel -> $local" -ForegroundColor DarkGray
    scp -P $Port -q -r "${dest}:$RemoteDir/$rel" $LocalDir 2>&1 | Out-Null
}

function Report {
    Get-ChildItem -Path (Join-Path $LocalDir "runs") -Filter metrics.jsonl -Recurse -ErrorAction SilentlyContinue |
        ForEach-Object {
            $lines = @(Get-Content $_.FullName)
            if ($lines.Count -eq 0) { return }
            $last = $lines[-1] | ConvertFrom-Json
            Write-Host ("  {0,-22} {1,-10} epoch {2,4}  loss {3:F4}  val_psnr {4:F3}" -f `
                        $_.Directory.Name, $last.phase, $last.epoch, $last.train_loss, $last.val_psnr) -ForegroundColor Green
        }
}

while ($true) {
    Write-Host ("[{0}] syncing from {1}:{2}" -f (Get-Date -Format "HH:mm:ss"), $dest, $RemoteDir) -ForegroundColor Cyan
    Pull "repro/runs"
    Pull "repro/logs"
    Report
    # checkpoints are what actually cost you the run: warn if nothing new appeared
    $newest = Get-ChildItem -Path (Join-Path $LocalDir "runs") -Filter "state_last.pt" -Recurse -ErrorAction SilentlyContinue |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($newest) {
        $age = ((Get-Date) - $newest.LastWriteTime).TotalMinutes
        Write-Host ("  newest state_last.pt is {0:F1} min old" -f $age) -ForegroundColor DarkCyan
    } else {
        Write-Host "  no state_last.pt pulled yet - is the training actually running?" -ForegroundColor Yellow
    }
    if ($Once) { break }
    Start-Sleep -Seconds $IntervalSec
}

Write-Host "`nlocal copy under: $LocalDir" -ForegroundColor Green
