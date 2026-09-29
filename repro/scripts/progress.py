"""Progress + ETA for the local two-stage training, from the files on disk only.

Nothing here talks to the trainer: it reads runs/<exp>/metrics.jsonl, the run's
config.yaml and logs/runner.log, so it works even if the training was launched by
hand, from another shell, or from a machine you just copied runs/ back from.

    python scripts/progress.py            # one shot
    python scripts/progress.py --watch    # refresh every 30 s
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import html
import json
import statistics
import subprocess
import time
from datetime import datetime, timedelta

from mobileie.config import REPRO_ROOT, RUNS_ROOT

DEFAULT_CHAIN = ["lolv1_stage1", "lolv1_stage2_iwo"]


def read_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def chain_from_logs() -> list[str]:
    """Whatever configs the runner actually announced, in order, plus the defaults."""
    log = REPRO_ROOT / "logs" / "runner.log"
    seen = []
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            for tok in line.split():
                if tok in DEFAULT_CHAIN or tok.startswith(("lolv1_", "uieb_", "zrr_")):
                    if tok not in seen:
                        seen.append(tok)
    for c in DEFAULT_CHAIN:
        if c not in seen:
            seen.append(c)
    return seen


def cfg_of(cfg_name: str):
    """Resolve a config to the run dir holding its newest progress.

    `make_run` suffixes a directory when the name is taken (lolv1_stage1_190523), so a
    relaunched chain writes somewhere other than the config's own exp_name.  Status has
    to follow the freshest metrics file, or the board shows a dead run while a live one
    is training.
    """
    plain = REPRO_ROOT / "configs" / f"{cfg_name}.yaml"
    cfg = read_yaml(plain) if plain.exists() else None
    exp = (cfg or {}).get("exp_name", cfg_name)
    cands = [p for p in RUNS_ROOT.iterdir()
             if p.is_dir() and (p.name == exp or p.name.startswith(exp + "_"))]
    if not cands:
        return cfg, RUNS_ROOT / exp
    newest = max(cands, key=lambda p: (p / "metrics.jsonl").stat().st_mtime
                 if (p / "metrics.jsonl").exists() else 0)
    if cfg is None and (newest / "config.yaml").exists():
        cfg = read_yaml(newest / "config.yaml")
    return cfg, newest


def rows_of(run_dir: Path) -> list[dict]:
    f = run_dir / "metrics.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def gpu_line() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        util, mem, pw = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
        return f"GPU 利用率 {util}%   显存 {mem} MiB   功耗 {pw} W"
    except Exception:                                   # noqa: BLE001 - status must never crash
        return "GPU 状态取不到（nvidia-smi 不可用）"


def rate(rows, window=25):
    """Median s/epoch over the last `window` logged epochs (robust to the first
    cuDNN-autotuned epoch and to one-off stalls)."""
    ts = [datetime.strptime(r["ts"], "%Y-%m-%d %H:%M:%S") for r in rows if "ts" in r]
    if len(ts) < 3:
        return None
    diffs = [(b - a).total_seconds() for a, b in zip(ts[-window - 1:], ts[-window:])]
    diffs = [d for d in diffs if 0 < d < 600]
    return statistics.median(diffs) if diffs else None


def render(show_watch_hint=True):
    print(f"\n{'='*72}\n MobileIE 本地训练进度   {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*72}")
    print(f" {gpu_line()}")

    rate_hint = None
    total_left = 0.0
    lines = []
    for cfg_name in chain_from_logs():
        cfg, run_dir = cfg_of(cfg_name)
        if cfg is None:
            lines.append(("skip", f" {cfg_name:<22} 配置读不到，跳过"))
            continue
        o = cfg["optim"]
        stage = o.get("stage", 1)
        warm = int(o.get("warmup_epochs", 0)) if stage == 1 else 0
        planned = int(o["epochs"])
        expected = planned + warm
        rows = rows_of(run_dir)
        done = len(rows)
        r = rate(rows) or rate_hint
        if r:
            rate_hint = r
        if not rows:
            est = f"，约 {planned * r / 3600:.1f} h" if r else ""
            lines.append(("wait", f" {cfg_name:<22} stage{stage} 未开始（{planned} epoch{est}）"))
            total_left += planned * r if r else 0
            continue
        age = (datetime.now() - datetime.strptime(rows[-1]["ts"], "%Y-%m-%d %H:%M:%S")).total_seconds()
        alive = r is None or age < max(3 * r, 120)
        pct = 100.0 * done / expected
        bar = "#" * int(pct // 5) + "." * (20 - int(pct // 5))
        best = max((x for x in rows if "val_psnr" in x), key=lambda x: x["val_psnr"], default=None)
        head = (f" {cfg_name:<22} stage{stage} [{bar}] {pct:5.1f}%  "
                f"{done}/{expected} ep  {'运行中' if alive else '已停'}")
        detail = []
        if r:
            left = (expected - done) * r
            total_left += left
            detail.append(f"   {r:.1f} s/epoch → 剩 {left/3600:.2f} h，约 "
                          f"{(datetime.now()+timedelta(seconds=left)):%m-%d %H:%M} 完成")
        last = rows[-1]
        detail.append(f"   最新 ep{last['epoch']} loss {last.get('train_loss', float('nan')):.4f}"
                      + (f" val_psnr {last['val_psnr']:.3f}" if "val_psnr" in last else ""))
        if best:
            detail.append(f"   best val_psnr {best['val_psnr']:.3f} @ ep{best['epoch']}"
                          f"   (论文 Table 1 LOLv1 = 23.62 dB)")
        if not alive:
            detail.append(f"   最后一条记录 {age/60:.0f} 分钟前 → 进程疑似已停，"
                          f"续跑：run.cmd start 或 python scripts/train.py --config {cfg_name} --resume {cfg['exp_name']}")
        lines.append(("row", head))
        lines += [("sub", d) for d in detail]

    for kind, text in lines:
        print(text + ("" if kind != "wait" else ""))
    if total_left:
        print(f"\n 链路合计剩余 ≈ {total_left/3600:.2f} h，约 "
              f"{(datetime.now()+timedelta(seconds=total_left)):%m-%d %H:%M} 全部跑完")
    else:
        print("\n 没有进行中的训练（runs/ 下无 metrics.jsonl）→ run.cmd start 启动")
    if show_watch_hint:
        print(" 持续刷新：run.cmd watch     停止：run.cmd stop\n")
    return total_left


def as_json():
    payload = {}
    for cfg_name in chain_from_logs():
        cfg, run_dir = cfg_of(cfg_name)
        if cfg is None:
            continue
        rows = rows_of(run_dir)
        payload[cfg_name] = {"exp_name": cfg["exp_name"], "stage": cfg["optim"].get("stage", 1),
                             "epochs": int(cfg["optim"]["epochs"]),
                             "warmup": int(cfg["optim"].get("warmup_epochs", 0)),
                             "done": len(rows), "s_per_epoch": rate(rows),
                             "last": rows[-1] if rows else None}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


# --------------------------------------------------------------- 展示层共享定义
# dashboard.py 与 board.py 共用这些，免得各写一份又互相漂移。

# lolv1_stage1 的 1-1000 轮已完整包含在 lolv1_ctrl_noiwo 里（同一条 run 从
# ckpt/state_last.pt 续跑），所以不再单独出卡。只有合并目标也在链路里时才跳过，
# 否则会把唯一的卡也藏掉。
MERGED_INTO = {"lolv1_stage1": "lolv1_ctrl_noiwo"}

# 卡片线色：蓝 = 无 IWO 的连续 2000 轮，红 = 第 1000 轮后接入 IWO。warmup 是自监督
# 预热，给中性灰，免得被误认成主曲线。
CARD_COLORS = {
    "lolv1_ctrl_noiwo": {"warmup": "#9aa5b4", "train_s1": "#0000ff"},
    "lolv1_stage2_iwo": {"warmup": "#9aa5b4", "train_s2": "#ff0000"},
}

# 论文 Fig. 10 的两臂对照：蓝 = 无 IWO 的连续 2000 轮，红 = 第 1000 轮后接入 IWO。
# 阶段二的 epoch 计数器从 1 开始，按论文口径平移到全局轴 1001-2000 才能同图比较。
FIG10_ARMS = (("lolv1_ctrl_noiwo", 0, "#0000ff", "Without IWO"),
              ("lolv1_stage2_iwo", 1000, "#ff0000", "Apply IWO (+1000)"))


def visible_chain() -> list[str]:
    """chain_from_logs() minus whatever has been merged into another card."""
    chain = chain_from_logs()
    return [c for c in chain if MERGED_INTO.get(c) not in chain]


def arm_rows(cfg_name: str) -> list[dict]:
    """One arm's metric rows on the shared global axis (warm-up excluded)."""
    cfg, run_dir = cfg_of(cfg_name)
    if cfg is None:
        return []
    return [r for r in rows_of(run_dir) if r.get("phase") != "warmup"]


def fig10_card() -> str:
    """The two-arm Fig. 10 comparison as HTML; empty until both arms have data."""
    from mobileie.charts import EPOCH_TICK_STEP, chart_div

    arms = [dict(cfg_name=c, color=col, label=lab, off=o, rows=arm_rows(c))
            for c, o, col, lab in FIG10_ARMS]
    if not [a for a in arms if len(a["rows"]) >= 2]:
        return ""

    def series_for(key):
        out = []
        for a in arms:
            pts = [[r["epoch"] + a["off"], r[key], {"lr": r.get("lr"), "ts": r.get("ts")}]
                   for r in a["rows"] if r.get(key) is not None]
            out.append({"label": a["label"], "color": a["color"], "pts": sorted(pts)})
        return out

    def chart(key, title, unit, log):
        return chart_div({"title": title, "unit": unit, "log": log, "h": 240,
                          "xstep": EPOCH_TICK_STEP, "series": series_for(key)})

    def stat(a):
        lo, hi = a["rows"][0]["epoch"] + a["off"], a["rows"][-1]["epoch"] + a["off"]
        best = max(a["rows"], key=lambda r: r.get("val_psnr") or -1e9)
        return (f'<div><span style="color:{a["color"]}">■</span> <b>{html.escape(a["label"])}</b> '
                f'<span class="k">{html.escape(a["cfg_name"])} · 全局 ep{lo}–{hi} · '
                f'best val {best.get("val_psnr", float("nan")):.4f} @ep{best["epoch"] + a["off"]} · '
                f'末轮 loss {a["rows"][-1].get("train_loss", float("nan")):.4f}</span></div>')

    note = ('<div class="k">两臂起点不同，读形状不读绝对差：蓝线 ep1–1000 就是阶段一（plain），'
            '红线从阶段一 <b>best</b>（ep579 权重）热启动、自带 1–1000 的调度，这里按论文的 '
            '"先 1000 轮再上 IWO" 平移到 1001–2000。已排除 10 轮 warmup。'
            'val_psnr 是看板口径（逐通道、不 clip），15 张图均值，单轮噪声中位 0.85 dB。'
            '▲/▼ = 每条曲线的最高/最低点（显示「数值@epoch」）；横轴每格 '
            f'{EPOCH_TICK_STEP} 轮。</div>')
    return (f'<div class="card"><b>两臂对照 · 论文 Fig. 10 口径</b>'
            f'{"".join(stat(a) for a in arms)}{note}'
            f'{chart("train_loss", "train_loss（log10 · 越低越好）", "", True)}'
            f'{chart("val_psnr", "val_psnr（dB · 越高越好）", "dB", False)}</div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="refresh every --interval seconds")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.json:
        as_json()
        return
    if not args.watch:
        render(show_watch_hint=True)
        return
    while True:
        render(show_watch_hint=False)
        print(f" (每 {args.interval}s 刷新，Ctrl+C 退出)")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
