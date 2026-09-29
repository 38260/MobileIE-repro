"""Time a TFLite model on whatever machine runs it - the laptop or the tablet.

One script on purpose: the paper never says how its SoC latency was taken (runtime,
delegate, precision, warm-up), so the only defensible number is one measured with a
protocol we fix ourselves and then reuse verbatim on both devices.

    python bench_tflite.py --model lolv1.tflite                      # latency only
    python bench_tflite.py --model lolv1.tflite --quality \           # + PSNR/MAE gate
        --inp data/LOLv1/eval15/low --gt data/LOLv1/eval15/high

Needs: numpy, pillow, ai-edge-litert (pip wheels exist for x86_64 and aarch64 glibc).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

WARMUP = 20


def make_interpreter(model: str, threads: int):
    from ai_edge_litert.interpreter import Interpreter
    try:
        interp = Interpreter(model_path=str(model), num_threads=threads)
    except TypeError:                      # older runtime builds have no num_threads
        interp = Interpreter(model_path=str(model))
    interp.allocate_tensors()
    return interp


def input_shape(interp, height: int, width: int) -> tuple:
    shape = [int(s) for s in interp.get_input_details()[0]["shape"]]
    return tuple(width if i == 2 and s == -1 else height if i == 1 and s == -1 else s
                 for i, s in enumerate(shape))


def random_input(interp, height: int, width: int) -> np.ndarray:
    shape = input_shape(interp, height, width)
    rng = np.random.default_rng(0)
    return rng.random(shape, dtype=np.float32)


def bench(interp, x: np.ndarray, iters: int, warmup: int) -> dict:
    inp = interp.get_input_details()[0]
    interp.set_tensor(inp["index"], x)
    for _ in range(warmup):
        interp.invoke()
    ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        interp.invoke()
        ms.append((time.perf_counter() - t0) * 1000.0)
    ms.sort()
    return {"frames": len(ms), "min_ms": round(ms[0], 3), "median_ms": round(statistics.median(ms), 3),
            "p95_ms": round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 3),
            "mean_ms": round(sum(ms) / len(ms), 3), "fps_median": round(1000.0 / statistics.median(ms), 1)}


def quality(interp, inp_dir: str, gt_dir: str, height: int, width: int) -> dict:
    from PIL import Image
    src, dst = Path(inp_dir), Path(gt_dir)
    names = sorted(p.name for p in src.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    psnr, mae = [], []
    for name in names:
        low = np.asarray(Image.open(src / name).convert("RGB").resize((width, height), Image.BICUBIC), np.float32) / 255.0
        high = np.asarray(Image.open(dst / name).convert("RGB").resize((width, height), Image.BICUBIC), np.float32) / 255.0
        interp.set_tensor(inp["index"], low[None, ...])
        interp.invoke()
        pred = np.squeeze(interp.get_tensor(out["index"])).clip(0.0, 1.0)
        mse = float(np.mean((pred - high) ** 2))
        psnr.append(10.0 * np.log10(1.0 / max(mse, 1e-12)))
        mae.append(float(np.mean(np.abs(pred - high))))
    return {"images": len(names), "psnr_rgb": round(float(np.mean(psnr)), 4), "mae": round(float(np.mean(mae)), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads; say which one you used")
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--inp", default="data/LOLv1/eval15/low")
    ap.add_argument("--gt", default="data/LOLv1/eval15/high")
    ap.add_argument("--json", default="", help="write the result to this file")
    args = ap.parse_args()

    interp = make_interpreter(args.model, args.threads)
    x = random_input(interp, args.height, args.width)
    result = {"model": str(args.model), "input_shape": list(x.shape), "threads": args.threads,
              "warmup": args.warmup, "note": "invoke() only: no image decode, no resize, no upload"}
    result.update(bench(interp, x, args.iters, args.warmup))
    if args.quality:
        result.update(quality(interp, args.inp, args.gt, args.height, args.width))

    import platform
    result["host"] = f"{platform.machine()} {platform.system()}"
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
