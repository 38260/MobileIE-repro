<#
.SYNOPSIS
  Build the uploadable bundle for cloud training: repro/ + the official repo + the
  cloud configs + the verified LOLv1, minus everything machine-specific or oversized.

.USAGE
  powershell -File cloud\pack.ps1 -DryRun      # show the plan + what the gates would see
  powershell -File cloud\pack.ps1              # -> cloud\dist\MobileIE-cloud.tar.gz
  powershell -File cloud\pack.ps1 -NoData      # leave LOLv1 out (re-download on the instance)
  powershell -File cloud\pack.ps1 -WithAllData -MaxMB 3000   # include UIEB/ZRR too

.NOTES
  robocopy's /XD only matches a bare directory NAME or a FULL PATH - a relative
  fragment like "data\_downloads" silently matches nothing.  That bug produced an
  11 GB bundle once, so the prune list below is a single source of truth, is passed
  to robocopy as full paths, is re-applied to the staged tree, and is finally
  checked by two gates (MB and file count) before tar runs.
#>
param(
    [switch]$DryRun,
    [switch]$NoData,
    [switch]$WithRuns,
    [switch]$WithAllData,
    [int]$MaxMB = 600,
    [int]$MaxFiles = 3000
)

$ErrorActionPreference = "Stop"
$Cloud = $PSScriptRoot
$BUNDLE = Split-Path -Parent $Cloud
$Repro = Join-Path $BUNDLE "repro"
$Official = Join-Path $BUNDLE "MobileIE-main"
$Dist = Join-Path $Cloud "dist"
$Stage = Join-Path $Dist "MobileIE"

# ---- single source of truth: bare dir names to drop anywhere, and specific paths
$bareDrop = @("__pycache__", ".ipynb_checkpoints", ".cache")   # .cache = huggingface pointer files, ~1k of them
if (-not $WithRuns) { $bareDrop += "runs" }
$pruneRel = @("data\_downloads", "third_party")          # relative to repro/
if (-not $WithAllData) { $pruneRel += @("data\UIEB", "data\ZRR") }
if ($NoData) { $pruneRel += "data" }
$xdArgs = @()
foreach ($n in $bareDrop) { $xdArgs += "/XD"; $xdArgs += $n }
foreach ($r in $pruneRel) { if ($r) { $xdArgs += "/XD"; $xdArgs += (Join-Path $Repro $r) } }
$xdArgs += "/XD"; $xdArgs += (Join-Path $Official "MobileIE-main\onnx2tflite\build")
$xdArgs += "/XD"; $xdArgs += (Join-Path $Official "MobileIE-main\onnx2tflite\dist")

function Staged([string]$rel) { Join-Path $Stage (Join-Path "repro" $rel) }
function Pruned($fullPath) {
    if (-not $fullPath.StartsWith($Repro)) { return $false }
    $rel = $fullPath.Substring($Repro.Length).TrimStart('\')
    foreach ($p in $pruneRel) { if ($p -and ($rel -eq $p -or $rel.StartsWith("$p\"))) { return $true } }
    return $false
}
function Plan() {
    $files = Get-ChildItem $Repro -Recurse -File -ErrorAction SilentlyContinue |
             Where-Object { -not (Pruned $_.FullName) }
    # a bare drop name matches any path SEGMENT, which is robocopy /XD semantics
    foreach ($n in $bareDrop) {
        $files = @($files | Where-Object {
            ($_.FullName.Substring($Repro.Length) -split '[\\/]') -notcontains $n
        })
    }
    [pscustomobject]@{
        MB    = [math]::Round(((($files | Measure-Object Length -Sum).Sum / 1MB)), 1)
        Files = @($files).Count
    }
}

$plan = Plan
Write-Host ("bundle plan: {0} files, {1} MB (gates: <= {2} files, <= {3} MB)" -f `
            $plan.Files, $plan.MB, $MaxFiles, $MaxMB) -ForegroundColor Cyan
if ($DryRun) {
    Write-Host "  dropped anywhere : $($bareDrop -join ', ')"
    Write-Host "  dropped in repro : $($pruneRel -join ', ')"
    Write-Host "  plus MobileIE-main ($((Get-ChildItem $Official -Recurse -File | Measure-Object).Count) files) and cloud/configs -> repro/configs"
    if ($plan.Files -gt $MaxFiles -or $plan.MB -gt $MaxMB) { Write-Host "  GATE WOULD FAIL - pass -WithAllData/-MaxMB or check the prune list" -ForegroundColor Yellow }
    else { Write-Host "  gates would pass; nothing written (-DryRun)" -ForegroundColor Green }
    exit 0
}
if ($plan.Files -gt $MaxFiles -or $plan.MB -gt $MaxMB) {
    throw "refusing to build: plan is $($plan.Files) files / $($plan.MB) MB, over the gate ($MaxFiles / ${MaxMB}MB)"
}

# ---- stage
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

function Robo($src, $dst, [string[]]$extraXd = @()) {
    $a = @($src, $dst, "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NP") + $extraXd
    & robocopy @a | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE) copying $src" }   # 0-7 == success
    $global:LASTEXITCODE = 0
}
Write-Host "staging -> $Stage"
Robo $Repro (Join-Path $Stage "repro") $xdArgs
Robo $Official (Join-Path $Stage "MobileIE-main") @("/XD", "build", "/XD", "dist", "/XD", "__pycache__")
Robo $Cloud (Join-Path $Stage "cloud") @("/XD", "dist")

# belt and braces: prune the staged tree by path, then re-verify against the gates
foreach ($r in $pruneRel) {
    if (-not $r) { continue }
    $p = Staged $r
    if (Test-Path $p) { Remove-Item -Recurse -Force $p }
}
Get-ChildItem (Join-Path $Stage "repro") -Recurse -Directory |
    Where-Object { $bareDrop -contains $_.Name } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $Cloud "configs\*.yaml") (Join-Path $Stage "repro\configs\") -Force

$staged = Get-ChildItem $Stage -Recurse -File
$actualMB = [math]::Round((($staged | Measure-Object Length -Sum).Sum / 1MB), 1)
Write-Host ("staged: {0} files, {1} MB" -f $staged.Count, $actualMB)
if ($staged.Count -gt $MaxFiles -or $actualMB -gt $MaxMB) {
    $worst = $staged | Group-Object { $_.Directory.FullName.Replace($Stage, '') } |
             Sort-Object { ($_.Group | Measure-Object Length -Sum).Sum } -Descending | Select-Object -First 5
    Write-Host "STAGED TREE OVER GATE, largest folders:" -ForegroundColor Red
    $worst | ForEach-Object { Write-Host ("  {0,8:N1} MB  {1,6} files  {2}" -f `
                 ((($_.Group | Measure-Object Length -Sum).Sum / 1MB), $_.Count, $_.Name)) -ForegroundColor Red }
    throw "refusing to tar an over-gate bundle; staging kept at $Stage for inspection"
}

# ---- tar
$tar = Join-Path $Dist "MobileIE-cloud.tar.gz"
if (Test-Path $tar) { Remove-Item -Force $tar }
Push-Location $Dist
try {
    & tar -czf "MobileIE-cloud.tar.gz" "MobileIE"
    if ($LASTEXITCODE -ne 0) { throw "tar failed ($LASTEXITCODE)" }
} finally { Pop-Location }
# staging is pure scratch; keep only the tarball
Remove-Item -Recurse -Force $Stage
$mb = [math]::Round((Get-Item $tar).Length / 1MB, 1)
Write-Host "`nbundle: $tar  ($mb MB)" -ForegroundColor Green
Write-Host @"

On the instance:
  scp -P <port> cloud/dist/MobileIE-cloud.tar.gz root@<host>:/root/
  ssh  -p <port> root@<host>
  cd /root && tar xzf MobileIE-cloud.tar.gz && bash MobileIE/cloud/bootstrap.sh

From a second terminal on the laptop, keep pulling the checkpoints back:
  powershell -File cloud\sync_back.ps1 -User root -Address <host> -Port <port> -RemoteDir /root/MobileIE
"@ -ForegroundColor Yellow
