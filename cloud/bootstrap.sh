#!/usr/bin/env bash
# On-instance bootstrap.  Run from anywhere inside the unpacked bundle:
#
#   bash cloud/bootstrap.sh                      # image already ships a CUDA torch (Kaggle/AutoDL)
#   SKIP_TORCH=0 bash cloud/bootstrap.sh         # bare image: install torch matching the driver
#   PYBIN=python3.10 bash cloud/bootstrap.sh     # image python is not `python3`
#
# Steps: env setup + 15 self-checks -> timed 3-epoch ETA probe -> launch
# stage1 -> IWO stage2 detached, with auto-restart from state_last.pt on crash.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(dirname "$HERE")"          # bundle root holds repro/, MobileIE-main/, cloud/
REPRO="$BUNDLE/repro"
export SKIP_TORCH="${SKIP_TORCH:-1}"
export PYBIN="${PYBIN:-python3}"

cd "$REPRO"
echo "### bundle=$BUNDLE"
echo "### SKIP_TORCH=$SKIP_TORCH PYBIN=$PYBIN"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no nvidia-smi"

bash setup.sh
PY="$BUNDLE/.venv/bin/python"

echo
echo "### ETA probe - 3 real epochs at the release's batch size 4"
echo "###   (includes first-iteration cuDNN autotuning, so it reads slightly pessimistic)"
T0=$(date +%s)
TQDM_DISABLE=1 "$PY" scripts/train.py --config lolv1_cloud_stage1 \
  --set optim.epochs=3 optim.warmup_epochs=0 --run-name cloud_probe
T1=$(date +%s)
rm -rf runs/cloud_probe
PER=$(( (T1 - T0) / 3 ))
echo "### measured ${PER} s/epoch"
echo "###   stage1 1000 ep  ~ $(( PER * 1000 / 3600 )) h"
echo "###   both stages     ~ $(( PER * 2000 / 3600 )) h"
echo "###   + no-IWO control ~ $(( PER * 2000 / 3600 )) h on top (2000 ep)"

echo
echo "### launching stage1 -> stage2 detached"
nohup bash scripts/run_two_stage.sh lolv1_cloud_stage1 lolv1_cloud_stage2_iwo >/dev/null 2>&1 &
sleep 15
echo "### progress:"
echo "    tail -f logs/lolv1_cloud_stage1.log"
echo "    tail -n 2 runs/lolv1_cloud_s1/metrics.jsonl"
echo
echo "### optional third arm (the Fig. 10 control, run it after the first two finish):"
echo "    cd $REPRO && nohup .venv/bin/python scripts/train.py --config ../cloud/configs/lolv1_cloud_ctrl_noiwo.yaml \\"
echo "      >> logs/lolv1_cloud_ctrl.log 2>&1 &"
echo
echo "### DO NOT trust the instance disk - pull runs/ back to your laptop from another terminal:"
echo "    powershell -File cloud\\sync_back.ps1 -User root -Port <port> -Host <region.autodl.com> \\"
echo "      -RemoteDir /root/MobileIE -IntervalSec 300"
