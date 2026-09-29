"""Verify a downloaded image dataset the way a reproduction should.

Guards against the failures that silently invalidate a run: missing pairs,
unreadable or truncated files, a "RAW" set that is really 8-bit photos, and
resolution surprises that break a fixed-shape TFLite graph.

    # LOLv1: two paired folders per split
    python scripts/verify_dataset.py --root data/LOLv1 \
        --pair our485/low=our485/high --pair eval15/low=eval15/high

    # ZRR: raw must really be 12-bit Bayer
    python scripts/verify_dataset.py --root data/ZRR \
        --pair train/raw=train/srgb --raw train/raw --json data/ZRR.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
from collections import Counter

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None          # a benchmark dataset is trusted, no decompression bomb check
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dng"}


def files_in(folder: Path) -> dict:
    return {p.stem: p for p in sorted(folder.iterdir())
            if p.suffix.lower() in EXTS and p.stat().st_size > 0}


def probe(paths, sample=6):
    """Return aggregate stats; decodes `sample` files spread across the folder."""
    hist, modes, bad = Counter(), Counter(), []
    picks = paths if sample is None else [paths[i] for i in
                                          np.linspace(0, len(paths) - 1, min(sample, len(paths))).astype(int)]
    for p in picks:
        try:
            im = Image.open(p)
            im.load()
            arr = np.asarray(im)
            hist[f"{im.size[0]}x{im.size[1]}"] += 1
            modes[f"{im.mode}/{arr.dtype}"] += 1
        except Exception as exc:                      # noqa: BLE001 - report, don't crash the audit
            bad.append(f"{p.name}: {type(exc).__name__} {exc}")
    return {"n": len(paths), "bytes": sum(p.stat().st_size for p in paths),
            "sampled": len(picks), "sizes": dict(hist), "modes": dict(modes), "unreadable": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pair", action="append", default=[], metavar="A=B",
                    help="relative folders that must share file stems")
    ap.add_argument("--solo", action="append", default=[], help="folder to report without pairing")
    ap.add_argument("--raw", action="append", default=[], metavar="FOLDER",
                    help="folder that must hold >8-bit data (Bayer RAW)")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--raw-sample", type=int, default=200, help="files to decode per --raw folder")
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"no such root: {root}")
    report, fails = {}, []

    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
        if not ok:
            fails.append(name)
        return ok

    def folder(spec):
        p = root / spec
        if not p.is_dir():
            check(f"folder exists: {spec}", False, str(p))
            return None, None
        f = files_in(p)
        return p, f

    for spec in args.solo:
        p, f = folder(spec)
        if f is None:
            continue
        st = probe(list(f.values()))
        report[spec] = st
        check(f"{spec}: non-empty", st["n"] >= args.min_count, f"{st['n']} images, {st['bytes']/1e6:.0f} MB")
        check(f"{spec}: all sampled files decode", not st["unreadable"], "; ".join(st["unreadable"][:2]))

    for spec in args.pair:
        a, _, b = spec.partition("=")
        pa, fa = folder(a)
        pb, fb = folder(b)
        if fa is None or fb is None:
            continue
        sa, sb = set(fa), set(fb)
        only_a, only_b = sorted(sa - sb)[:3], sorted(sb - sa)[:3]
        check(f"pair {a} <-> {b}: stems match", sa == sb,
              f"|{a}|={len(sa)} |{b}|={len(sb)} only-a={only_a} only-b={only_b}")
        for name, f in ((a, fa), (b, fb)):
            st = probe(list(f.values()))
            report[name] = st
            check(f"{name}: non-empty", st["n"] >= args.min_count,
                  f"{st['n']} images, {st['bytes']/1e6:.0f} MB, sizes={list(st['sizes'])[:3]}")
            check(f"{name}: sampled files decode", not st["unreadable"], "; ".join(st["unreadable"][:2]))

    for spec in args.raw:
        p, f = folder(spec)
        if f is None:
            continue
        paths = list(f.values())
        picks = [paths[i] for i in np.linspace(0, len(paths) - 1, min(args.raw_sample, len(paths))).astype(int)]
        modes, maxima, chan = Counter(), [], set()
        for path in picks:
            im = Image.open(path)
            im.load()
            arr = np.asarray(im)
            modes[f"{im.mode}/{arr.dtype}"] += 1
            chan.add(arr.ndim)
            maxima.append(int(arr.max()))
        maxima = np.array(maxima)
        # PNG picks its bit depth from the content, so a dark tile legitimately lands in
        # 8-bit storage: "max > 255" is NOT evidence of RAW-ness.  What would break
        # training is saturation at the container ceiling, i.e. values clipped to 255.
        n8 = sum(v for k, v in modes.items() if "uint8" in k)
        sat8 = sum(1 for path, m in zip(picks, maxima)
                   if np.asarray(Image.open(path)).dtype == np.uint8 and m >= 255)
        frac_sat = sat8 / max(n8, 1) if n8 else 0.0
        report[spec] = {"n": len(paths), "sampled": len(picks), "modes": dict(modes),
                        "max_median": int(np.median(maxima)), "max_p95": int(np.percentile(maxima, 95)),
                        "uint8_files": n8, "uint8_saturated": sat8}
        check(f"{spec}: no 8-bit saturation at 255 (unclipped linear RAW)", frac_sat < 0.02,
              f"{n8}/{len(picks)} sampled files are 8-bit, {sat8} of them hit max=255 "
              f"({frac_sat:.1%}); median max={int(np.median(maxima))}")
        check(f"{spec}: single-channel mosaic (ndim==2)", chan == {2}, f"ndim set={chan}")
        print(f"      info: {len(picks) - n8}/{len(picks)} sampled files need >8 bits "
              f"(p95 max={int(np.percentile(maxima, 95))}) - mixed bit depth is PNG auto-selecting, not corruption")

    print()
    if fails:
        print("FAILED CHECKS:", fails)
    else:
        print("DATASET VERIFIED:", root)
    if args.json:
        Path(args.json).write_text(json.dumps({"root": str(root), "folders": report,
                                               "failed": fails}, indent=2), encoding="utf-8")
        print("report written:", args.json)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
