#!/usr/bin/env bash
# Run both LOLv1 stages back to back, detached, so an ~18h chain survives the
# terminal AND survives being killed: a stage that dies restarts from
# runs/<exp_name>/ckpt/state_last.pt (up to MAXRETRY times).
#
#   nohup bash scripts/run_two_stage.sh > /dev/null 2>&1 &
#   bash scripts/run_two_stage.sh lolv1_stage1 lolv1_stage2_iwo
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
S1="${1:-lolv1_stage1}"
S2="${2:-lolv1_stage2_iwo}"
MAXRETRY="${MAXRETRY:-4}"
PY="${PYTHON:-../.venv/bin/python}"
mkdir -p logs
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_stage() {
  local cfg="$1" try=0
  while (( try < MAXRETRY )); do
    try=$(( try + 1 ))
    if (( try == 1 )); then
      echo "[runner] $cfg start (attempt 1) $(date)" >> logs/runner.log
      "$PY" scripts/train.py --config "$cfg" >> "logs/$cfg.log" 2>&1
    else
      echo "[runner] $cfg resuming from state_last.pt (attempt $try) $(date)" >> logs/runner.log
      "$PY" scripts/train.py --config "$cfg" --resume "$cfg" >> "logs/$cfg.log" 2>&1
    fi
    if [[ $? -eq 0 ]]; then
      echo "[runner] $cfg done $(date)" >> logs/runner.log
      return 0
    fi
    echo "[runner] $cfg exited non-zero $(date) -- retrying in 20s" >> logs/runner.log
    sleep 20
  done
  echo "[runner] $cfg gave up after $MAXRETRY retries $(date)" >> logs/runner.log
  return 1
}

run_stage "$S1" && run_stage "$S2" && echo "[runner] both stages done $(date)" >> logs/runner.log
