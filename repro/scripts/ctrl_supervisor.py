"""Unattended supervisor for the LOLv1 no-IWO control run (2000 epochs).

Replaces the cmd-based wrappers (``queue_control.cmd`` / ``run_two_stage.cmd``)
for this run, for three reasons:

1. ``queue_control.cmd`` gives up after ``MAXRETRY=4``.  This machine performs a
   shutdown transition roughly every 1-3 hours (see
   ``MobileIE_后续验证计划.md`` Phase 0), so retries must be unbounded.  That is
   safe because ``train.py`` always resumes from
   ``runs/<exp>/ckpt/state_last.pt``: a retry costs at most one epoch, never
   progress.
2. Those wrappers export ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``,
   which this platform does not support (PyTorch prints a UserWarning and the
   option is a no-op).  It is replaced by ``max_split_size_mb``.
3. ``cmd``-level backoff used ``timeout``, which needs a real console and returns
   immediately when the host has none.

``train.py`` exits 0 once the run reaches ``optim.epochs`` (2000), at which point
this supervisor exits 0 as well.

Run it from your own terminal (a console that stays open), NOT from a tool
session -- processes spawned inside a tool session are reaped when that call
returns:

    D:\\AIWorkSpace\\MobileIE\\.venv\\Scripts\\python.exe repro\\scripts\\ctrl_supervisor.py

Stop it with Ctrl+C; ``state_last.pt`` is kept, so the next start resumes.

Logs: ``logs/ctrl_supervisor.log`` (lifecycle), ``logs/lolv1_ctrl_noiwo.log``
(training stdout), ``runs/lolv1_ctrl_noiwo/metrics.jsonl`` (per-epoch metrics).
"""
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
PY = REPRO.parent / ".venv" / "Scripts" / "python.exe"
CFG = "lolv1_ctrl_noiwo"
LOG = REPRO / "logs" / "ctrl_supervisor.log"
TRAIN_LOG = REPRO / "logs" / f"{CFG}.log"
MUTEX_NAME = "MobileIE_ctrl_supervisor_singleton"


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def acquire_singleton():
    """Named mutex: released automatically by the OS when this process dies, so
    it cannot go stale the way a lock file does."""
    handle = ctypes.windll.kernel32.CreateMutexW(None, False, MUTEX_NAME)
    if handle and ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        return None
    return handle


def main() -> int:
    if not PY.exists():
        log(f"no interpreter at {PY}")
        return 1
    if acquire_singleton() is None:
        log("another supervisor is already running -- refusing to start a second one")
        return 1

    env = dict(os.environ)
    env["TQDM_DISABLE"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    attempt = 0
    log(f"supervisor start pid={os.getpid()} cfg={CFG}")
    while True:
        attempt += 1
        log(f"attempt {attempt} start")
        with open(TRAIN_LOG, "a", encoding="utf-8") as f:
            rc = subprocess.run([str(PY), "scripts/train.py", "--config", CFG],
                                cwd=str(REPRO), env=env,
                                stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            log("done rc=0")
            return 0
        log(f"exited rc={rc} -- retry in 20s")
        time.sleep(20)


if __name__ == "__main__":
    sys.exit(main())
