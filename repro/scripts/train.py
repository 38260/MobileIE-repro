"""Launch a training stage.

    python scripts/train.py --config lolv1_stage1
    python scripts/train.py --config lolv1_stage2_iwo
    python scripts/train.py --config lolv1_stage1 --set optim.batch_size=2 optim.epochs=5
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from mobileie.config import RUNS_ROOT, deep_update, latest_run, load_config, make_run, parse_override
from mobileie.engine import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config name under configs/ or a path")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="dotted overrides, e.g. optim.epochs=200")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", metavar="RUN_NAME", default=None,
                    help="continue a specific runs/<RUN_NAME> from its ckpt/state_last.pt")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing state and start a new timestamped run")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    deep_update(cfg, parse_override(args.set))
    resume_name = args.resume
    if not resume_name and not args.fresh and not args.run_name:
        found = latest_run(cfg["exp_name"], want="ckpt/state_last.pt")
        resume_name = found.name if found else None
    if resume_name:
        state = RUNS_ROOT / resume_name / "ckpt" / "state_last.pt"
        if not state.exists():
            raise SystemExit(f"no resumable state at {state}")
        cfg["optim"]["resume_from"] = str(state)
    run = make_run(cfg, args.run_name or resume_name, resume=bool(resume_name))
    print(f"run dir: {run.root}  [{'resume ' + resume_name if resume_name else 'fresh start'}]")
    train(cfg, run, args.device)


if __name__ == "__main__":
    main()
