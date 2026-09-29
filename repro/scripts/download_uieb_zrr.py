"""Fetch the MobileIE UIEB (underwater) and ZRR (RAW->sRGB) benchmarks.

Network reality on this box (verified 2026-09-26): huggingface.co direct,
drive.google.com and openbayes.com are unreachable; hf-mirror.com, github.com
and download.ai-benchmark.com work.

UIEB
  source : https://hf-mirror.com/datasets/CK234/my_UIEB  -> dataset.zip (786,403,255 B)
  ships  : train/input_train_800 (800 png) + train/gt_train_800 (800 png)
           test/UIEB/input_testUIEB_(90) + test/UIEB/gt_testUIEB_(90)  (90 png each)
           (+ test/UIEB/UIEB_60 = 60 challenge inputs with no GT, test/SQUID)
  -> data/UIEB/{train,test}/{input,gt}
  NOTE the paper's "950 train / 90 test" is not what any reachable UIEB archive
  contains: the benchmark is 890 raw/GT pairs (+60 challenge), usually split 800/90.

ZRR = "Zurich RAW-to-DSLR" (PyNet/LiteISP), i.e. the huawei_raw + canon folders that
MobileIE's config/isp.yaml points at (isp/{train,test}/huawei_raw, isp/{train,test}/canon)
  source : https://download.ai-benchmark.com/s/GsfikXymMKHxCwr/download/
           Zurich-RAW-to-DSLR-Dataset.zip  (23,808,958,387 B, sha1 1494ef9c24972076cd3bae18c54b47850d4c7244)
  The archive is never downloaded whole: only the byte spans of the four wanted
  directories are pulled with parallel HTTP Range requests and the members are
  sliced out locally and CRC-32 checked against the central directory.
    test/huawei_raw + test/canon   : 1,204 pairs,  371 MB  -> data/ZRR/test/{raw,srgb}
    train/huawei_raw + train/canon : 46,839 pairs, 14.5 GB -> data/ZRR/train/{raw,srgb}
  Skipped (not needed by the ISP task): train|test/*_visualized (1.57 GB),
  full_resolution/* (6.1 GB, 168 oversized singles), test/huawei_full_resolution.
  The paper's "190 train / 50 test" are RAW/sRGB image *groups*; the shipped,
  loader-compatible split of those groups is 46,839 train + 1,204 test 448x448
  Bayer-patch pairs (which is also why the paper's ISP table quotes FPS at
  448x448; the "256x256 Bayer" sentence in Sec. 4.1 does not match the data).
  RAW member = single-channel PNG, 801/1204 test files carry 16-bit values
  (mode I;16, black level ~64) and 403/1204 were stored as mode L because their
  values all fit in 8 bits; ispdata.py's /4095 is applied either way.

Usage (run from repro/ with the venv python):
  python scripts/download_uieb_zrr.py uieb
  python scripts/download_uieb_zrr.py zrr --only test
  python scripts/download_uieb_zrr.py zrr
  python scripts/download_uieb_zrr.py verify
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import struct
import sys
import threading
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
DATA = REPRO / "data"
CACHE = DATA / "_downloads"
UIEB_ZIP = CACHE / "UIEB_dataset.zip"
PARTS = CACHE / "zrr_parts"

UIEB_URL = "https://hf-mirror.com/datasets/CK234/my_UIEB/resolve/main/dataset.zip"
UIEB_SIZE = 786_403_255
# final (post-redirect) WebDAV endpoint of the ZRR archive; supports HTTP Range
ZRR_URL = "https://download.ai-benchmark.com/public.php/dav/files/GsfikXymMKHxCwr"
ZRR_ORIG = ("https://download.ai-benchmark.com/s/GsfikXymMKHxCwr/download/"
            "Zurich-RAW-to-DSLR-Dataset.zip")
CHUNK = 32 * 1024 * 1024
UA = "Mozilla/5.0 (compatible; MobileIE-repro/1.0)"

ZRR_MAP = {  # zip directory -> destination sub-folder under data/ZRR/<split>/
    "train/huawei_raw": "raw",
    "train/canon": "srgb",
    "test/huawei_raw": "raw",
    "test/canon": "srgb",
}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def log(*a):
    print(*a, flush=True)


# -------------------------------------------------------------------------- io
def http_get(url: str, rng: tuple[int, int] | None = None,
             timeout: int = 300, tries: int = 6) -> bytes:
    headers = {"User-Agent": UA, "Accept-Encoding": "identity"}
    if rng:
        headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
    last: Exception | None = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=headers), timeout=timeout) as r:
                buf = bytearray()
                while True:
                    b = r.read(1 << 21)
                    if not b:
                        break
                    buf += b
                data = bytes(buf)
            if rng and len(data) != rng[1] - rng[0] + 1:
                raise RuntimeError(f"short read {len(data)} of {rng}")
            return data
        except Exception as e:  # noqa: BLE001 - transient CDN/Nextcloud failures
            last = e
            time.sleep(1 + 2 * attempt)
    raise RuntimeError(f"GET {url} {rng} failed after {tries} tries: {last}")


def remote_size(url: str) -> int:
    with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"}),
            timeout=120) as r:
        return int(r.headers["Content-Range"].split("/")[-1])


def download_whole(url: str, dest: Path, expect: int | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expect is None:
        expect = remote_size(url)
    if dest.exists() and dest.stat().st_size == expect:
        log(f"[cache] {dest.name}: already {expect:,} B")
        return dest
    have = dest.stat().st_size if dest.exists() else 0
    log(f"[get ] {url}\n       -> {dest} ({have:,} B on disk, {expect:,} B total)")
    step = 64 * 1024 * 1024
    t0 = time.time()
    with open(dest, "ab" if have else "wb") as f:
        off = have
        while off < expect:
            end = min(off + step - 1, expect - 1)
            f.write(http_get(url, (off, end)))
            f.flush()
            off = end + 1
            got = off - have
            log(f"       {off:,}/{expect:,}  {got / max(time.time() - t0, 1e-9) / 2**20:6.2f} MiB/s")
    if dest.stat().st_size != expect:
        raise RuntimeError(f"{dest.name}: {dest.stat().st_size:,} != {expect:,}")
    return dest


# ------------------------------------------------------------------------ uieb
UIEB_LAYOUT = [
    ("train/input_train_800/", "UIEB/train/input"),
    ("train/gt_train_800/", "UIEB/train/gt"),
    ("test/UIEB/input_testUIEB_(90)/", "UIEB/test/input"),
    ("test/UIEB/gt_testUIEB_(90)/", "UIEB/test/gt"),
]


def do_uieb() -> None:
    download_whole(UIEB_URL, UIEB_ZIP, UIEB_SIZE)
    with zipfile.ZipFile(UIEB_ZIP) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        for prefix, dsub in UIEB_LAYOUT:
            out = DATA / dsub
            out.mkdir(parents=True, exist_ok=True)
            n = skip = 0
            for m in infos:
                if not m.filename.startswith(prefix):
                    continue
                name = m.filename[len(prefix):]
                if not name or name.startswith("."):
                    continue
                dst = out / name
                if dst.exists() and dst.stat().st_size == m.file_size:
                    skip += 1
                    continue
                dst.write_bytes(zf.read(m))
                n += 1
            log(f"[uieb] {dsub}: {n} written, {skip} already present")


# ------------------------------------------------------------------------- zrr
def zrr_central_dir() -> list[dict]:
    """Read the (zip64) central directory of the remote archive, ~15.7 MB."""
    tail_len = 40 * 1024 * 1024
    total = remote_size(ZRR_URL)
    blob = http_get(ZRR_URL, (total - tail_len, total - 1), timeout=900)
    z64 = blob.rfind(b"PK\x06\x06")           # zip64 end of central directory record
    if z64 < 0:
        raise RuntimeError("no zip64 EOCD found in the archive tail")
    entries = struct.unpack("<Q", blob[z64 + 32:z64 + 40])[0]
    cd_size = struct.unpack("<Q", blob[z64 + 40:z64 + 48])[0]
    cd_off = struct.unpack("<Q", blob[z64 + 48:z64 + 56])[0]
    log(f"[zrr ] archive {total:,} B, central dir {entries:,} entries "
        f"at {cd_off:,} ({cd_size:,} B)")
    blob_start = total - len(blob)
    if cd_off < blob_start:
        blob = http_get(ZRR_URL, (cd_off, cd_off + cd_size - 1), timeout=900)
        blob_start = cd_off
    out: list[dict] = []
    ndirs = 0
    p = cd_off - blob_start
    while blob[p:p + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", blob[p + 10:p + 12])[0]
        crc = struct.unpack("<I", blob[p + 16:p + 20])[0]
        comp = struct.unpack("<I", blob[p + 20:p + 24])[0]      # compressed size
        uncomp = struct.unpack("<I", blob[p + 24:p + 28])[0]    # uncompressed size
        nlen = struct.unpack("<H", blob[p + 28:p + 30])[0]
        elen = struct.unpack("<H", blob[p + 30:p + 32])[0]
        clen = struct.unpack("<H", blob[p + 32:p + 34])[0]
        lho = struct.unpack("<I", blob[p + 42:p + 46])[0]
        name = blob[p + 46:p + 46 + nlen].decode("utf-8", "surrogateescape")
        q, qe = p + 46 + nlen, p + 46 + nlen + elen
        while q + 4 <= qe:
            hid = struct.unpack("<H", blob[q:q + 2])[0]
            dsz = struct.unpack("<H", blob[q + 2:q + 4])[0]
            if hid == 1:  # zip64 ext info: a field is present only when its u32 was 0xFFFFFFFF
                v = blob[q + 4:q + 4 + dsz]
                o = 0
                if uncomp == 0xFFFFFFFF:
                    uncomp, o = struct.unpack("<Q", v[o:o + 8])[0], o + 8
                if comp == 0xFFFFFFFF:
                    comp, o = struct.unpack("<Q", v[o:o + 8])[0], o + 8
                if lho == 0xFFFFFFFF and o + 8 <= len(v):
                    lho = struct.unpack("<Q", v[o:o + 8])[0]
                break
            q += 4 + dsz
        if name.endswith("/"):
            ndirs += 1
        else:
            out.append(dict(name=name, lho=lho, comp=comp, uncomp=uncomp,
                            crc=crc, method=method))
        nxt = p + 46 + nlen + elen + clen
        if nxt >= len(blob) or blob[nxt:nxt + 4] != b"PK\x01\x02":
            break
        p = nxt
    if len(out) + ndirs != entries:
        raise RuntimeError(f"central dir truncated: parsed {len(out):,} files + "
                           f"{ndirs} dirs != {entries:,} entries")
    log(f"[zrr ] central dir complete: {len(out):,} file members (+{ndirs} dir entries)")
    return out


_dl_bytes = [0]
_dlock = threading.Lock()


def _chunk_path(abs_off: int) -> Path:
    return PARTS / f"chunk_{abs_off // CHUNK:08d}.bin"


def _fetch_chunk(rng: tuple[int, int]) -> None:
    a, b = rng
    assert b // CHUNK == a // CHUNK, "chunk straddles a boundary"
    path = _chunk_path(a)
    want = b - a + 1
    if path.exists() and path.stat().st_size == want:
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(http_get(ZRR_URL, (a, b), timeout=900))
    if tmp.stat().st_size != want:
        raise RuntimeError(f"chunk {a}: bad size")
    os.replace(tmp, path)
    with _dlock:
        _dl_bytes[0] += want


def _fetch_many(ranges: list[tuple[int, int]], workers: int) -> None:
    cached = sum(1 for a, b in ranges
                 if _chunk_path(a).exists() and _chunk_path(a).stat().st_size == b - a + 1)
    if cached:
        log(f"       {cached}/{len(ranges)} chunks already cached")
    t0 = time.time()
    _dl_bytes[0] = 0
    done = [0]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(_fetch_chunk, ranges):
            done[0] += 1
            if done[0] % 20 == 0 or done[0] == len(ranges):
                el = max(time.time() - t0, 1e-9)
                log(f"       {done[0]}/{len(ranges)} chunks  "
                    f"{_dl_bytes[0] / el / 2**20:6.2f} MiB/s  elapsed {el:,.0f}s")


class SpanReader:
    """Reader over the cached CHUNK-sized blocks that cover [lo, hi)."""

    def __init__(self, lo: int, hi: int):
        self.lo, self.hi = lo, hi

    def read(self, off: int, n: int) -> bytes:
        assert self.lo <= off and off + n <= self.hi, (off, n)
        idx = off // CHUNK
        buf = bytearray()
        while len(buf) < n:
            base = idx * CHUNK
            with open(_chunk_path(base), "rb") as f:
                f.seek(off + len(buf) - base)
                buf += f.read(n - len(buf))
            idx += 1
        return bytes(buf)


def zrr_runs(members: list[dict], splits: tuple[str, ...]) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for m in members:
        parts = m["name"].split("/")
        if len(parts) != 3:
            continue
        run = "/".join(parts[:2])
        if run in ZRR_MAP and parts[0] in splits:
            runs.setdefault(run, []).append(m)
    for ms in runs.values():
        ms.sort(key=lambda m: m["lho"])
    return runs


def extract_run(runs: dict[str, list[dict]], workers: int) -> None:
    for run, ms in sorted(runs.items()):
        split, src = run.split("/")
        dest = DATA / "ZRR" / split / ZRR_MAP[run]
        dest.mkdir(parents=True, exist_ok=True)
        lo = ms[0]["lho"]
        hi = max(m["lho"] + 30 + len(m["name"].encode()) + m["comp"] for m in ms)
        gaps = hi - lo - sum(m["comp"] for m in ms)
        span_lo = lo // CHUNK * CHUNK
        span_hi = ((hi + CHUNK - 1) // CHUNK) * CHUNK - 1
        ranges = [(span_lo + i * CHUNK, min(span_lo + (i + 1) * CHUNK - 1, span_hi))
                  for i in range((span_hi - span_lo + 1) // CHUNK)]
        log(f"[zrr ] {run}: {len(ms):,} members, {sum(m['uncomp'] for m in ms):,} B "
            f"payload, span {lo:,}..{hi:,} ({(hi - lo) / 2**30:.2f} GiB incl. "
            f"{gaps / 2**20:.0f} MiB headers)")
        _fetch_many(ranges, workers)
        rdr = SpanReader(lo, hi + 64)
        written = skip = 0
        for m in ms:
            dst = dest / m["name"].rsplit("/", 1)[1]
            if dst.exists() and dst.stat().st_size == m["uncomp"]:
                skip += 1
                continue
            hdr = rdr.read(m["lho"], 30)
            if hdr[:4] != b"PK\x03\x04":
                raise RuntimeError(f"bad local header for {m['name']}")
            nlen = struct.unpack("<H", hdr[26:28])[0]
            elen = struct.unpack("<H", hdr[28:30])[0]
            data = rdr.read(m["lho"] + 30 + nlen + elen, m["comp"])
            payload = data if m["method"] == 0 else zlib.decompress(data, -15)
            if len(payload) != m["uncomp"] or (zlib.crc32(payload) & 0xFFFFFFFF) != m["crc"]:
                raise RuntimeError(f"CRC/size mismatch for {m['name']}")
            tmp = dst.with_name(dst.name + ".tmp")
            tmp.write_bytes(payload)
            os.replace(tmp, dst)
            written += 1
            if (written + skip) % 2000 == 0:
                log(f"       {written + skip:,}/{len(ms):,} extracted")
        log(f"       -> {dest.relative_to(DATA)}: {written:,} written, {skip:,} cached")
        for a, _b in ranges:  # release the byte cache for this run
            pth = _chunk_path(a)
            if pth.exists():
                pth.unlink()


def do_zrr(splits: tuple[str, ...], workers: int) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    members = zrr_central_dir()
    runs = zrr_runs(members, splits)
    log(f"[zrr ] to pull: {sum(len(v) for v in runs.values()):,} members, "
        f"{sum(m['comp'] for v in runs.values() for m in v):,} B compressed")
    extract_run(runs, workers)


# ---------------------------------------------------------------------- verify
def images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXT)


def _pick(paths: list[Path], k: int) -> list[Path]:
    if len(paths) <= k:
        return list(paths)
    return [paths[round(i * (len(paths) - 1) / (k - 1))] for i in range(k)]


def sample_files(paths: list[Path], k: int = 3) -> list[dict]:
    """Fully decode k spread-out files: shape / dtype / min / max."""
    import numpy as np
    from PIL import Image
    out = []
    for p in _pick(paths, k):
        try:
            with Image.open(p) as im:
                mode, size, arr = im.mode, list(im.size), np.asarray(im)
            out.append(dict(file=p.name, pil_mode=mode, pil_size_wh=size,
                            shape=list(arr.shape), dtype=str(arr.dtype),
                            channels=(1 if arr.ndim == 2 else int(arr.shape[2])),
                            min=int(arr.min()), max=int(arr.max()),
                            bytes=p.stat().st_size))
        except Exception as e:  # noqa: BLE001
            out.append(dict(file=p.name, error=f"{type(e).__name__}: {e}"))
    return out


def header_stats(paths: list[Path], cap: int = 3000) -> dict:
    """Header-only (fast) survey: PIL modes and resolutions."""
    import random
    from PIL import Image
    scan = paths if len(paths) <= cap else random.Random(0).sample(paths, cap)
    modes: collections.Counter = collections.Counter()
    res: collections.Counter = collections.Counter()
    for p in scan:
        try:
            with Image.open(p) as im:
                modes[im.mode] += 1
                res[tuple(im.size)] += 1
        except Exception:  # noqa: BLE001
            modes["unreadable"] += 1
    return dict(scanned=len(scan), of_total=len(paths), pil_modes=dict(modes),
                distinct_resolutions=len(res),
                smallest_wh=list(min(res)), largest_wh=list(max(res)))


def bayer_smoke(paths: list[Path], n: int = 3) -> list[dict]:
    """MobileIE-main/data/ispdata.py: single-channel Bayer PNG -> 4x(H/2)x(W/2)/4095."""
    import numpy as np
    from PIL import Image
    res = []
    for p in _pick(paths, n):
        with Image.open(p) as im:
            mode, raw = im.mode, np.asarray(im)
        if raw.ndim != 2:
            res.append(dict(file=p.name, error=f"not single-channel: {raw.shape}"))
            continue
        h, w = raw.shape
        if h % 2 or w % 2:
            res.append(dict(file=p.name, error=f"odd dims {h}x{w}"))
            continue
        rggb = raw.reshape(h // 2, 2, w // 2, 2).transpose(
            [1, 3, 0, 2]).reshape(-1, h // 2, w // 2)
        res.append(dict(file=p.name, pil_mode=mode, mosaic_hw=[h, w],
                        input_dtype=str(raw.dtype), input_min=int(raw.min()),
                        input_max=int(raw.max()), above_8bit=bool(raw.max() > 255),
                        rggb_shape=list(rggb.shape),
                        normalised_max=float(rggb.max()) / 4095))
    return res


def depth_proof(paths: list[Path], n: int = 5) -> dict:
    smoke = bayer_smoke(paths, n)
    stats = header_stats(paths)
    return dict(
        note=("each RAW file is opened exactly as MobileIE-main/data/ispdata.py does; "
              "mode I;16 == 16-bit container holding 12-bit Bayer values (<=4095), "
              "mode L == same values that happened to fit in 8 bits"),
        samples=smoke,
        all_single_channel=bool(smoke) and all("error" not in s for s in smoke),
        samples_above_255=sum(1 for s in smoke if s.get("above_8bit")),
        sample_count=len(smoke),
        max_value_seen_in_samples=max((s.get("input_max", 0) for s in smoke), default=0),
        any_value_over_4095=any(s.get("input_max", 0) > 4095 for s in smoke),
        folder_pil_modes=stats["pil_modes"], folder_scanned=stats["scanned"],
        mosaic_resolutions=stats["distinct_resolutions"],
        largest_wh=stats["largest_wh"], smallest_wh=stats["smallest_wh"],
    )


def do_verify() -> int:
    groups = [("UIEB/train/input", "UIEB/train/gt"),
              ("UIEB/test/input", "UIEB/test/gt"),
              ("ZRR/train/raw", "ZRR/train/srgb"),
              ("ZRR/test/raw", "ZRR/test/srgb")]
    man: dict = {
        "sources": {
            "UIEB": dict(repo="hf-mirror dataset CK234/my_UIEB", url=UIEB_URL,
                         archive=str(UIEB_ZIP),
                         archive_bytes=UIEB_ZIP.stat().st_size if UIEB_ZIP.exists() else 0),
            "ZRR": dict(repo="Zurich RAW-to-DSLR (PyNet/LiteISP, ai-benchmark.com)",
                        url=ZRR_ORIG, archive_bytes_total=23_808_958_387,
                        method="parallel HTTP Range spans + local CRC-32 check"),
        },
        "folders": {}, "pairs": {},
    }
    for left, right in groups:
        for d in (left, right):
            root = DATA / d
            files = images(root) if root.is_dir() else []
            exts = collections.Counter(p.suffix.lower() for p in files)
            hs = header_stats(files)
            man["folders"][d] = dict(
                path=str(root), file_count=len(files),
                total_bytes=sum(p.stat().st_size for p in files),
                distinct_extensions=sorted(exts), extension_counts=dict(exts),
                samples=sample_files(files, 3),
                pil_modes=hs["pil_modes"], pil_modes_scanned=hs["scanned"],
                distinct_resolutions=hs["distinct_resolutions"],
                smallest_wh=hs["smallest_wh"], largest_wh=hs["largest_wh"],
            )
        lf, rf = images(DATA / left), images(DATA / right)
        lstem = collections.Counter(p.stem for p in lf)
        rstem = collections.Counter(p.stem for p in rf)
        miss_r = sorted(set(lstem) - set(rstem))
        miss_l = sorted(set(rstem) - set(lstem))
        dup_l = {k: v for k, v in lstem.items() if v > 1}
        dup_r = {k: v for k, v in rstem.items() if v > 1}
        lfull = {p.name for p in lf}
        rfull = {p.name for p in rf}
        key = f"{left} <-> {right}"
        man["pairs"][key] = dict(
            left=left, right=right, left_count=len(lf), right_count=len(rf),
            all_input_stems_have_reference=not (miss_r or miss_l),
            missing_reference_stems=miss_r[:50], missing_input_stems=miss_l[:50],
            missing_reference_count=len(miss_r), missing_input_count=len(miss_l),
            duplicate_stems_on_left=dup_l, duplicate_stems_on_right=dup_r,
            # the released ispdata.py indexes BOTH folders with the same filename
            # string, so it needs identical names, not just identical stems
            exact_filenames_identical=sorted(lfull) == sorted(rfull),
            gt_samples_are_3ch_uint8=bool(sample_files(rf, 3)) and all(
                s.get("channels") == 3 and s.get("dtype") == "uint8"
                for s in sample_files(rf, 3))),
        man["pairs"][key]["reference_side_samples"] = sample_files(rf, 3)
        flag = " [ok] " if not (miss_r or miss_l) else " [BAD]"
        log(f"{flag} {key}: {len(lf)} vs {len(rf)} files, stems unpaired="
            f"{len(miss_r)}/{len(miss_l)}, identical filenames="
            f"{man['pairs'][key]['exact_filenames_identical']}")
    for split in ("train", "test"):
        raw = images(DATA / "ZRR" / split / "raw")
        if raw:
            man.setdefault("zrr_raw_depth_proof", {})[split] = depth_proof(raw, 5)
    man["deviations"] = {
        "UIEB_pair_count": ("expected 950 train + 90 test; no reachable UIEB archive "
                            "holds that. The benchmark is 890 raw/GT pairs + 60 "
                            "challenge inputs with no GT, and the community split of "
                            "the 890 is 800 train / 90 test - that is what is installed, "
                            "fully paired. Extensions are all .png (no jpg)."),
        "ZRR_granularity": ("expected ~190 train + 50 test pairs; ZRR ships the 190/50 "
                            "image GROUPS as 46,839 train + 1,204 test 448x448 patch "
                            "pairs (the archive's full_resolution/ holds only 168 "
                            "singles and no train/test split). Installed the patch "
                            "split, which is what config/isp.yaml loads."),
        "ZRR_raw_depth": ("RAW PNGs are single-channel Bayer mosaics but are split "
                          "801/1204 mode I;16 and 403/1204 mode L: patches whose "
                          "values all fit in 8 bits were stored 8-bit by the archive "
                          "producer. Values are unclipped (black level ~64, max ~1023 "
                          "in test), so ispdata.py's /4095 stays correct."),
        "ZRR_filename_coupling": ("raw is N.png and srgb is N.jpg: stems match (so "
                                  "repro/src/mobileie/data.py works) but the released "
                                  "ispdata.py uses one filename for both folders, so it "
                                  "needs a 1-line stem-lookup patch or a jpg->png rename."),
    }
    out = DATA / "manifest.json"
    out.write_text(json.dumps(man, indent=2), encoding="utf-8")
    log(f"[manifest] {out}")
    total = 0
    for k, v in man["folders"].items():
        total += v["total_bytes"]
        log(f"     {k:22s} n={v['file_count']:6,} bytes={v['total_bytes']:14,} "
            f"ext={v['distinct_extensions']} res={v['largest_wh']}")
    log(f"     {'TOTAL on disk':22s} bytes={total:,}")
    ok = all(v["all_input_stems_have_reference"] for v in man["pairs"].values())
    return 0 if ok else 1


def main() -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["uieb", "zrr", "verify", "all"])
    ap.add_argument("--only", choices=["train", "test"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if a.what in ("uieb", "all"):
        do_uieb()
    if a.what in ("zrr", "all"):
        do_zrr((a.only,) if a.only else ("test", "train"), a.workers)
    return do_verify()


if __name__ == "__main__":
    sys.exit(main())
