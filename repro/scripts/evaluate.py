"""Evaluate a checkpoint on LOLv1's eval15 split with the paper's metric set.

    python scripts/evaluate.py --config lolv1_pretrained
    python scripts/evaluate.py --config lolv1_stage1 --run-name lolv1_stage1 --save-images
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

import numpy as np
import torch
import cv2

from mobileie import model as M
from mobileie.complexity import describe, measure_latency
from mobileie.config import RUNS_ROOT, deep_update, load_config, parse_override
from mobileie.engine import evaluate


def to_bgr(t: torch.Tensor) -> np.ndarray:
    arr = (t.detach().clamp(0, 1).mul(255).permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    return arr[..., ::-1]


@torch.no_grad()
def save_triptychs(net, cfg, out_dir: Path, device):
    from mobileie.data import PairedImageDataset
    ds = PairedImageDataset(cfg["data"]["test_inp"], cfg["data"]["test_gt"], cache=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(ds)):
        inp, gt, name = ds[i]
        out = net(inp.unsqueeze(0).to(device))[0].cpu()
        strip = np.concatenate([to_bgr(inp), to_bgr(out), to_bgr(gt)], axis=1)
        cv2.imwrite(str(out_dir / f"{name}.png"), strip)
    print(f"input|enhanced|GT strips written to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default=None, help="checkpoint .pkl (default: model.pretrained)")
    ap.add_argument("--graph", default=None, choices=["slim", "plain", "iwo"])
    ap.add_argument("--run-name", default=None, help="load ckpt from runs/<name>/ckpt/model_best*.pkl")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config)
    deep_update(cfg, parse_override(args.set))

    weights, graph = args.weights, args.graph
    if args.run_name:
        root = RUNS_ROOT / args.run_name / "ckpt"
        prefer = [p for p in (root / "model_best.pkl", root / "model_best_slim.pkl") if p.exists()]
        if not prefer:
            raise FileNotFoundError(f"no checkpoint under {root}")
        weights = str(prefer[0])

    state = M.read_state(weights or cfg["model"]["pretrained"])
    path = weights or cfg["model"]["pretrained"]
    graph = graph or M.graph_type_of(state)
    print(f"checkpoint: {path}\ndetected graph: {graph}")

    net, graph = M.build_from_checkpoint(cfg, path, graph)
    device = cfg["device"]

    if graph != "slim":
        net = net.to(device).eval()
        summary = evaluate(net, cfg, device, split=args.split)
        slim = net.slim().to(device).eval()
        slim_summary = evaluate(slim, cfg, device, split=args.split)
        slim_check = {"folded_matches_original": bool(
            abs(summary["psnr"] - slim_summary["psnr"]) < 0.02)}
        metrics_net = slim
    else:
        net = net.to(device).eval()
        summary = evaluate(net, cfg, device, split=args.split)
        slim_check = {}
        metrics_net = net

    shape = tuple(cfg["eval"].get("shape", [1, 3, 400, 600]))
    cx = describe(metrics_net, shape, device)
    lat = measure_latency(metrics_net, shape, device)

    out = {
        "config": cfg["_config_path"],
        "checkpoint": str(path),
        "graph": graph,
        "split": args.split,
        "metrics": {k: v for k, v in summary.items() if k != "per_image"},
        "complexity": cx,
        "latency": lat,
        "reparam_check": slim_check,
        "reference": cfg.get("reference", {}),
        "per_image": summary.get("per_image", []),
    }

    tag = args.tag or Path(cfg["_config_path"]).stem
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    target = results_dir / f"eval_{tag}.json"
    with open(target, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    if args.save_images:
        run_dir = RUNS_ROOT / (args.run_name or tag) / "images"
        save_triptychs(metrics_net, cfg, run_dir, torch.device(device))

    print(json.dumps({k: out[k] for k in ("metrics", "complexity", "latency", "reparam_check")},
                     indent=2))
    print("written:", target)


if __name__ == "__main__":
    main()
