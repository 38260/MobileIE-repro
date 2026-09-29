"""Verify the *deployable* artifact: pretrain/lolv1.tflite, on x86 CPU.

The mobile half of the paper's story is told in TFLite, so the file that gets
pushed to a phone must itself be checked against Table 1 - a tflite that only
matches the .pkl in shape but drifts in quality would invalidate every on-device
number measured later.  Inputs are NHWC as the released test_TFLite_RGB.py expects.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import time

import numpy as np
import torch

from mobileie.config import OFFICIAL_ROOT, RESULTS_ROOT
from mobileie.data import PairedImageDataset
from mobileie.metrics import mae, psnr_official, psnr_rgb, ssim_rgb

MODEL = OFFICIAL_ROOT / "pretrain" / "lolv1.tflite"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--inp", default="data/LOLv1/eval15/low")
    ap.add_argument("--gt", default="data/LOLv1/eval15/high")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=args.model)
    interp.allocate_tensors()
    det_in, det_out = interp.get_input_details()[0], interp.get_output_details()[0]
    print(f"model : {args.model}")
    print(f"input : {det_in['shape']} {det_in['dtype']} name={det_in['name']}")
    print(f"output: {det_out['shape']} {det_out['dtype']} name={det_out['name']}")
    _, h, w, c = det_in["shape"]
    fixed_hw = h > 0 and w > 0

    ds = PairedImageDataset(args.inp, args.gt, cache=True)
    if fixed_hw:
        from PIL import Image
        sizes = {(Image.open(p).size[1], Image.open(p).size[0]) for p in ds.inp_files}
        if sizes != {(h, w)}:
            raise SystemExit(f"model is fixed at HxW={h}x{w} but the dataset has {sizes}")

    rows = []
    for i in range(len(ds)):
        inp, gt, name = ds[i]
        x = inp.permute(1, 2, 0).numpy()[None].astype(np.float32)  # NCHW -> NHWC
        interp.set_tensor(det_in["index"], x)
        t0 = time.perf_counter()
        interp.invoke()
        dt = time.perf_counter() - t0
        out = interp.get_tensor(det_out["index"])[0]  # HWC
        pred = torch.from_numpy(np.ascontiguousarray(out.transpose(2, 0, 1))).float().clamp(0, 1)[None]
        gt4 = gt[None]
        rows.append({
            "name": name,
            "psnr": float(psnr_rgb(pred, gt4)),
            "psnr_per_channel": float(psnr_official(pred, gt4)),
            "ssim": ssim_rgb(out, gt.permute(1, 2, 0).numpy()),
            "mae": float(mae(pred, gt4)),
            "ms_x86_cpu": dt * 1000,
            "max_abs_out": float(out.max()), "min_abs_out": float(out.min()),
        })
        print(f"  {name:>6}  PSNR {rows[-1]['psnr']:6.3f}  SSIM {rows[-1]['ssim']:.4f}  "
              f"invoke {dt*1000:6.2f} ms")

    summary = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "psnr_per_channel", "ssim", "mae")}
    summary["ms_x86_cpu_mean"] = float(np.mean([r["ms_x86_cpu"] for r in rows]))
    summary["output_range"] = [min(r["min_abs_out"] for r in rows), max(r["max_abs_out"] for r in rows)]
    summary["n"] = len(rows)

    ref = {}
    pkl_path = RESULTS_ROOT / "eval_pretrained.json"
    if pkl_path.exists():
        ref = json.loads(pkl_path.read_text(encoding="utf-8"))["metrics"]
        print("\nvs the .pkl checkpoint measured by scripts/evaluate.py:")
        for key in ("psnr", "ssim"):
            print(f"  {key:6} tflite={summary[key]:.4f}  pkl={ref[key]:.4f}  "
                  f"delta={summary[key]-ref[key]:+.4f}")
    print("\nvs paper Table 1 (LOLv1):")
    for key, paper in (("psnr", 23.62), ("ssim", 0.812), ("psnr_per_channel", None)):
        if paper is not None:
            print(f"  {key:6} measured={summary[key]:.4f}  paper={paper}  delta={summary[key]-paper:+.4f}")

    out_path = RESULTS_ROOT / "tflite_lolv1.json"
    out_path.write_text(json.dumps({"model": args.model, "summary": summary,
                                    "reference_pkl": ref, "per_image": rows}, indent=2), encoding="utf-8")
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
