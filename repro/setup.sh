#!/usr/bin/env bash
# One-shot environment setup for the MobileIE reproduction (Linux / WSL2).
# Windows: use setup.ps1 instead.
#
#   bash setup.sh                     # auto-detect the CUDA wheel line
#   TORCH_INDEX=cu121 bash setup.sh   # force a line
#   SKIP_DOWNLOAD=1 bash setup.sh     # data/LOLv1 already copied over
#
# The venv is not portable between machines - always re-run this after a copy.
set -euo pipefail

REPRO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$REPRO")"
VENV="${VENV:-$ROOT/.venv}"
PYBIN="${PYBIN:-python3.11}"
TORCH_INDEX="${TORCH_INDEX:-auto}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_CHECK="${SKIP_CHECK:-0}"
cd "$REPRO"

step() { printf '\n==> %s\n' "$1"; }

# ---------------------------------------------------------------- 0. layout
step "Checking project layout"
for need in requirements.txt src/mobileie scripts ../MobileIE-main/MobileIE-main/model/utils.py; do
  if [[ ! -e "$need" ]]; then
    echo "FAIL: missing $need"
    echo "The reproduction loads the official re-parameterisation code from"
    echo "  MobileIE-main/MobileIE-main/model/utils.py   (one level ABOVE repro/)"
    echo "Copy the whole MobileIE folder, not just repro/."
    exit 1
  fi
done
echo "OK: repro/ and the official repo are both present"

# ---------------------------------------------------------------- 1. which torch line
if [[ "$TORCH_INDEX" == auto ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    TORCH_INDEX=cpu; reason="nvidia-smi not found"
  else
    line=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)
    drv=$(cut -d, -f2 <<<"$line" | tr -d ' ')
    major=${drv%%.*}
    if grep -qiE 'RTX 50|GB2' <<<"$line"; then TORCH_INDEX=cu128; reason="Blackwell needs cu128+ ($line)"
    elif (( major >= 570 )); then TORCH_INDEX=cu128; reason="driver $drv"
    elif (( major >= 528 )); then TORCH_INDEX=cu121; reason="driver $drv too old for cu128"
    elif (( major >= 450 )); then TORCH_INDEX=cu118; reason="driver $drv"
    else TORCH_INDEX=cpu; reason="driver $drv too old - update it for GPU training"; fi
  fi
else
  reason="explicitly requested"
fi
step "Torch build to install: $TORCH_INDEX ($reason)"

# ---------------------------------------------------------------- 2. venv + deps
# With SKIP_TORCH the venv must see the image's own torch, which lives in the system
# site-packages - a plain venv would not import it.
step "Creating virtualenv at $VENV"
VENV_FLAGS=()
if [[ "${SKIP_TORCH:-0}" == 1 ]]; then VENV_FLAGS+=(--system-site-packages); fi
if command -v uv >/dev/null 2>&1; then
  uv venv "$VENV" --python "$PYBIN" "${VENV_FLAGS[@]+"${VENV_FLAGS[@]}"}"
  pip_install() { uv pip install --python "$VENV/bin/python" "$@"; }
else
  "$PYBIN" -m venv "${VENV_FLAGS[@]+"${VENV_FLAGS[@]}"}" "$VENV"
  pip_install() { "$VENV/bin/python" -m pip install "$@"; }
fi
echo "OK: venv ready"

step "Installing PyTorch"
if [[ "${SKIP_TORCH:-0}" == 1 ]]; then
  echo "SKIP_TORCH=1 - keeping the image's own torch:"
  "$VENV/bin/python" -c "import torch;print(' torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available())" \
    || { echo "FAIL: image torch not importable in this venv; re-run without SKIP_TORCH"; exit 1; }
elif [[ "$TORCH_INDEX" == cpu ]]; then
  pip_install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
  pip_install torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
fi
if [[ "${PYPI_MIRROR:-}" == cn ]]; then
  pip_install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
else
  pip_install -r requirements.txt
fi
echo "OK: dependencies installed"

# ---------------------------------------------------------------- 3. verify
step "Verifying CUDA actually works"
if ! "$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
if not torch.cuda.is_available():
    print("CUDA UNAVAILABLE - training would run on CPU, far too slow for this train graph")
    raise SystemExit(3)
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
free, total = torch.cuda.mem_get_info()
torch.nn.functional.conv2d(torch.rand(4, 3, 256, 256, device="cuda"),
                           torch.rand(12, 3, 5, 5, device="cuda"))
print(f"GPU {name}  sm_{cap[0]}{cap[1]}  VRAM {total/1e9:.1f} GB (free {free/1e9:.1f})  conv OK")
print("SUGGESTED_BATCH=" + ("4" if total / 1e9 >= 14 else "2"))
PY
then
  echo "This torch build does not run on the local driver.  Retry with e.g."
  echo "  TORCH_INDEX=cu126 bash setup.sh      (or cu121 / cu118 / cpu)"
  exit 1
fi

# ---------------------------------------------------------------- 4. data
if [[ "$SKIP_DOWNLOAD" == 1 ]]; then
  step "Skipping dataset download (SKIP_DOWNLOAD=1)"
elif [[ -d data/LOLv1/our485/low ]]; then
  step "data/LOLv1 already present - skipping download"
else
  step "Downloading LOLv1 via hf-mirror.com (~350 MB)"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  "$VENV/bin/python" scripts/download_lolv1.py
fi

# ---------------------------------------------------------------- 5. self-check
if [[ "$SKIP_CHECK" != 1 ]]; then
  step "Running algorithm self-checks (re-parameterisation, IWO, loader)"
  "$VENV/bin/python" scripts/selfcheck.py
fi

step "Ready"
cat <<EOS
  train stage 1 :  $VENV/bin/python scripts/train.py --config lolv1_stage1
  train stage 2 :  $VENV/bin/python scripts/train.py --config lolv1_stage2_iwo
  detached      :  nohup bash scripts/run_two_stage.sh > logs/runner.log 2>&1 &
  report        :  $VENV/bin/python scripts/report.py
On a card with >=14 GB free VRAM add  --set optim.batch_size=4  (the release's value).
EOS
