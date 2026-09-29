import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPRO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPRO_ROOT.parent
OFFICIAL_ROOT = PROJECT_ROOT / "MobileIE-main" / "MobileIE-main"
DATA_ROOT = REPRO_ROOT / "data"
RUNS_ROOT = REPRO_ROOT / "runs"
RESULTS_ROOT = REPRO_ROOT / "results"

_SUFFIX = re.compile(r"^_\d{6}$")
_RUN_REF = re.compile(r"^runs[/\\](?P<exp>[^/\\]+)[/\\](?P<rest>.+)$")


def run_dirs(exp_name: str) -> list[Path]:
    """runs/<exp> plus the timestamped siblings runs/<exp>_HHMMSS, newest first.

    A fresh run never overwrites an existing dir, so a resumed stage has to find its
    own state by prefix instead of by the bare exp name.
    """
    out = [p for p in RUNS_ROOT.glob(f"{exp_name}*")
           if p.is_dir() and (p.name == exp_name or _SUFFIX.fullmatch(p.name[len(exp_name):]))]

    def stamp(p: Path) -> float:
        files = [p.stat().st_mtime] + [f.stat().st_mtime for f in (p / "ckpt").glob("*") if f.is_file()]
        return max(files)

    return sorted(out, key=stamp, reverse=True)


def latest_run(exp_name: str, want: str | None = None) -> Path | None:
    for path in run_dirs(exp_name):
        if want is None or (path / want).exists():
            return path
    return None


def resolve_run_ref(value: str) -> str:
    """Rewrite runs/<exp>/<rel> to that exp's newest run dir holding <rel>."""
    match = _RUN_REF.match(value)
    if not match:
        return value
    found = latest_run(match.group("exp"), want=match.group("rest"))
    return str(found / match.group("rest")) if found else value


_PATH_KEYS = {"data": ("_inp", "_gt"), "model": ("pretrained",), "optim": ("init_from",)}


def resolve_paths(cfg: dict) -> dict:
    """Dataset / checkpoint paths in the YAML are written relative to repro/."""
    for section, suffixes in _PATH_KEYS.items():
        for key, value in (cfg.get(section) or {}).items():
            if not isinstance(value, str) or not value:
                continue
            if key.endswith(suffixes):
                value = resolve_run_ref(value)
                if not Path(value).is_absolute():
                    value = str((REPRO_ROOT / value).resolve())
                cfg[section][key] = value
    return cfg


def load_config(name: str) -> dict:
    """Accepts a config name under repro/configs (e.g. 'lolv1_stage1') or a real path
    ('/abs/x.yaml', './cloud/configs/x.yaml') so cloud configs can live outside repro/."""
    candidate = Path(name)
    if candidate.suffix in (".yaml", ".yml") and candidate.exists():
        path = candidate.resolve()
    elif candidate.is_absolute():
        path = candidate
    else:
        path = REPRO_ROOT / "configs" / f"{name}.yaml"
        if not path.exists() and (ROOT := (REPRO_ROOT / name).resolve()).exists():
            path = ROOT
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return resolve_paths(cfg)


def deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_override(pairs):
    """`a.b.c=value` strings -> nested dict with YAML-typed values."""
    out = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        out_key = key.split(".")
        node = out
        for part in out_key[:-1]:
            node = node.setdefault(part, {})
        node[out_key[-1]] = yaml.safe_load(raw)
    return out


@dataclass
class RunPaths:
    root: Path
    config: dict = field(default_factory=dict)

    def __post_init__(self):
        (self.root / "ckpt").mkdir(parents=True, exist_ok=True)
        (self.root / "images").mkdir(parents=True, exist_ok=True)

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def log(self) -> Path:
        return self.root / "train.log"

    def append_metric(self, record: dict):
        record["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.metrics, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def write_config(self):
        with open(self.root / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True, sort_keys=False)

    def write_json(self, name: str, payload) -> Path:
        path = self.root / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path


def make_run(cfg: dict, run_name: str | None = None, resume: bool = False) -> RunPaths:
    name = run_name or cfg["exp_name"]
    if resume:
        root = latest_run(name, want="ckpt/state_last.pt") or RUNS_ROOT / name
    elif (RUNS_ROOT / name).exists():
        root = RUNS_ROOT / f"{name}_{time.strftime('%H%M%S')}"
    else:
        root = RUNS_ROOT / name
    return RunPaths(root=root, config=cfg)


def ensure_env():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
