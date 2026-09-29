<#
.SYNOPSIS
  One-shot environment setup for the MobileIE reproduction, for moving the project
  to another machine (desktop dGPU / WSL / CPU-only all supported).

.USAGE
  powershell -ExecutionPolicy Bypass -File repro\setup.ps1
  powershell -ExecutionPolicy Bypass -File repro\setup.ps1 -TorchIndex cu121
  powershell -ExecutionPolicy Bypass -File repro\setup.ps1 -SkipDownload -PypiMirror cn

.NOTES
  The venv is NOT portable between machines - always re-run this script after a copy.
  data/, runs/ and results/ ARE portable and need no re-download (-SkipDownload).
#>
param(
    [ValidateSet("auto", "cu128", "cu126", "cu121", "cu118", "cu11", "cpu")]
    [string]$TorchIndex = "auto",
    [string]$PythonVersion = "3.11",
    [ValidateSet("", "cn")]
    [string]$PypiMirror = "",
    [switch]$SkipDownload,
    [switch]$SkipCheck
)

$ErrorActionPreference = "Stop"
$Repro = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $Repro
$Venv = Join-Path $Root ".venv"
$PyRel = if ($env:OS -eq "Windows_NT") { "Scripts\python.exe" } else { "bin/python" }
$Py = Join-Path $Venv $PyRel

function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "    OK   $m" -ForegroundColor Green }
function Bad($m)  { Write-Host "    FAIL $m" -ForegroundColor Red }

function Choose-Index {
    # Returns (index, reason). cu128 covers sm_70..sm_120, but needs a CUDA 12.8-capable
    # driver; RTX 50-series needs cu128 or newer no matter what.
    if ($TorchIndex -ne "auto") { return @($TorchIndex, "explicitly requested") }
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { return @("cpu", "nvidia-smi not found") }
    try {
        $line = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) | Select-Object -First 1
    } catch { return @("cpu", "nvidia-smi query failed") }
    if (-not $line) { return @("cpu", "no GPU reported") }
    $name, $ver = $line -split ","
    # nvidia-smi's CSV puts a space after the comma; without trimming, the version
    # regex below silently fails and a 30/40-series box would be mis-read as "cpu".
    $ver = ("$ver").Trim()
    $major = 0; $build = 0
    if ($ver -match "^(\d+)\.(\d+)") { $major = [int]$Matches[1]; $build = [int]$Matches[2] }
    if ($name -match "RTX 50|B\d{3}|GB") { return @("cu128", "Blackwell (sm_120) needs cu128+: $name") }
    if ($major -gt 570 -or ($major -eq 570 -and $build -ge 62)) { return @("cu128", "driver $ver") }
    if ($major -ge 528) { return @("cu121", "driver $ver too old for cu128") }
    if ($major -ge 450) { return @("cu118", "driver $ver") }
    return @("cpu", "driver $ver too old for any CUDA 11/12 wheel - update the driver")
}

Set-Location $Repro

# ---------------------------------------------------------------- 0. sanity checks
Step "Checking project layout"
foreach ($need in @("requirements.txt", "src/mobileie", "scripts", "..\MobileIE-main\MobileIE-main\model\utils.py")) {
    if (-not (Test-Path (Join-Path $Repro $need))) {
        Bad "missing $need"
        Write-Host @"

The reproduction loads the official re-parameterisation code from
  MobileIE-main/MobileIE-main/model/utils.py
one level ABOVE repro/.  Copy the whole MobileIE folder, not just repro/.
"@ -ForegroundColor Yellow
        exit 1
    }
}
Ok "repro/ and the official repo are both present"

$Index, $Reason = Choose-Index
Step "Torch build to install: $Index ($Reason)"

# ---------------------------------------------------------------- 1. interpreter
Step "Creating virtualenv at $Venv"
$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($uv) {
    & uv venv $Venv --python $PythonVersion
    $UseUv = $true
} else {
    Write-Host "    uv not found - falling back to python -m venv + pip" -ForegroundColor Yellow
    $boot = $null
    foreach ($c in @("python$PythonVersion", "python", "py")) {
        if (Get-Command $c -ErrorAction SilentlyContinue) { $boot = $c; break }
    }
    if (-not $boot) { Bad "no Python 3.10-3.12 interpreter found"; exit 1 }
    if ($boot -eq "py") { & py "-$PythonVersion" -m venv $Venv } else { & $boot -m venv $Venv }
    $UseUv = $false
}
Ok "venv ready"

# ---------------------------------------------------------------- 2. dependencies
Step "Installing PyTorch ($Index)"
$TorchUrls = @{
    "cu128" = "https://download.pytorch.org/whl/cu128"
    "cu126" = "https://download.pytorch.org/whl/cu126"
    "cu121" = "https://download.pytorch.org/whl/cu121"
    "cu118" = "https://download.pytorch.org/whl/cu118"
}
$Pkgs = @("torch", "torchvision")
if ($Index -eq "cpu") {
    if ($UseUv) { & uv pip install --python $Py torch torchvision --index-url https://download.pytorch.org/whl/cpu }
    else { & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu }
} else {
    $url = $TorchUrls[$Index]
    if ($UseUv) { & uv pip install --python $Py ($Pkgs -join " ") --index-url $url }
    else { & $Py -m pip install ($Pkgs -join " ") --index-url $url }
}
Ok "torch installed"

Step "Installing the remaining dependencies"
$Req = Join-Path $Repro "requirements.txt"
$idxArgs = if ($PypiMirror -eq "cn") { @("-i", "https://pypi.tuna.tsinghua.edu.cn/simple") } else { @() }
if ($UseUv) { & uv pip install --python $Py -r $Req @idxArgs }
else { & $Py -m pip install -r $Req @idxArgs }
Ok "dependencies installed"

# ---------------------------------------------------------------- 3. verify GPU path
Step "Verifying CUDA actually works"
$probe = @'
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
if not torch.cuda.is_available():
    print("CUDA UNAVAILABLE - training will run on CPU (very slow for this train graph)")
    raise SystemExit(3)
n = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
free, total = torch.cuda.mem_get_info()
x = torch.nn.functional.conv2d(torch.rand(4, 3, 256, 256, device="cuda"),
                               torch.rand(12, 3, 5, 5, device="cuda"))
print(f"GPU {n}  sm_{cap[0]}{cap[1]}  VRAM total {total/1e9:.1f} GB / free {free/1e9:.1f} GB  conv OK")
bs = 4 if total / 1e9 >= 14 else 2
print(f"SUGGESTED_BATCH={bs}")
'@
$probe | Out-File -Encoding ascii (Join-Path $env:TEMP "aib_probe.py")
& $Py (Join-Path $env:TEMP "aib_probe.py")
$ProbeCode = $LASTEXITCODE
if ($ProbeCode -ne 0) {
    Write-Host @"
    torch built for $Index does not run on this driver.  Retry with a lower line:
      powershell -File setup.ps1 -TorchIndex cu126      (or cu121 / cu118 / cpu)
    Or update the NVIDIA driver and re-run.
"@ -ForegroundColor Yellow
} else { Ok "GPU path verified" }

# ---------------------------------------------------------------- 4. dataset
$Data = Join-Path $Repro "data\LOLv1\our485\low"
if ($SkipDownload) {
    Step "Skipping dataset download (-SkipDownload); expecting an already-copied data/LOLv1"
} elseif (Test-Path $Data) {
    Step "data/LOLv1 already present - skipping download"
} else {
    Step "Downloading LOLv1 via hf-mirror.com (~350 MB)"
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    & $Py (Join-Path $Repro "scripts\download_lolv1.py")
    if ($LASTEXITCODE -ne 0) { Bad "dataset download failed"; exit 1 }
    Ok "LOLv1 complete (485 train pairs + 15 eval pairs)"
}

# ---------------------------------------------------------------- 5. self-check
if (-not $SkipCheck) {
    Step "Running algorithm self-checks (re-parameterisation, IWO, loader)"
    & $Py (Join-Path $Repro "scripts\selfcheck.py")
    if ($LASTEXITCODE -ne 0) { Bad "selfcheck failed on this machine - do not trust runs from it"; exit 1 }
    Ok "all checks passed"
}

# ---------------------------------------------------------------- 6. next steps
Step "Ready"
Write-Host @"
  train stage 1 :  $Py scripts\train.py --config lolv1_stage1
  train stage 2 :  $Py scripts\train.py --config lolv1_stage2_iwo
  both, detached:  cmd /c "start `"`" /min scripts\run_two_stage.cmd"
  measure         :  $Py scripts\evaluate.py --config lolv1_stage1 --run-name lolv1_stage1 --tag stage1
  report          :  $Py scripts\report.py
"@ -ForegroundColor Green
Write-Host "  On a bigger card add  --set optim.batch_size=4  (the release's value) if free VRAM allows."
if ($ProbeCode -ne 0) { exit 1 }
