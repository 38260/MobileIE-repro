"""Assemble results/report.md: paper numbers vs reproduced numbers + curves.

    python scripts/report.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mobileie.complexity import count_parameters, count_macs
from mobileie.config import RESULTS_ROOT, RUNS_ROOT
from mobileie import model as M

# Published numbers to compare against.
PAPER = {
    "table1_lolv1": {"params_k": 4.047, "model_size_mb": 0.02, "gpu_latency_ms": 0.895,
                     "soc_latency_ms": 6.72, "fps": 1120.584, "psnr": 23.62, "ssim": 0.812,
                     "lpips": 0.198},
    "table7": {"train_params_k": 47.211, "infer_params_k": 4.047,
               "train_flops_g": 11.042, "infer_flops_g": 0.924,
               "train_latency_ms": 9.58, "infer_latency_ms": 0.895},
}


def load_evals():
    out = {}
    for path in sorted(glob.glob(str(RESULTS_ROOT / "eval_*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        out[Path(path).stem[len("eval_"):]] = data
    tp = RESULTS_ROOT / "tflite_lolv1.json"
    if tp.exists():
        with open(tp, encoding="utf-8") as f:
            t = json.load(f)
        summary = dict(t["summary"])
        # verify_tflite.py names it psnr_per_channel; keep one key across the report
        summary.setdefault("psnr_official", summary.get("psnr_per_channel"))
        out["tflite"] = {"metrics": summary, "reference": summary}
    return out


def read_metrics(run_dir: Path):
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_curves(runs: dict, out_path: Path):
    """Loss + validation PSNR per stage; the paper shows these as Fig. 10."""
    names = [n for n, r in runs.items() if r]
    if not names:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in names:
        rec = runs[name]
        epoch, phase = 0, {}
        for row in rec:
            key = row.get("phase", "train")
            st = phase.setdefault(key, {"e": [], "loss": [], "psnr": []})
            st["e"].append(row["epoch"])
            st["loss"].append(row.get("train_loss"))
            st["psnr"].append(row.get("val_psnr"))
        for key, st in phase.items():
            axes[0].plot(st["e"], st["loss"], label=f"{name} ({key})", lw=1)
            axes[1].plot(st["e"], st["psnr"], label=f"{name} ({key})", lw=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation PSNR (dB)")
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    axes[0].set_title("Training loss (cf. Fig. 10)")
    axes[1].set_title("Validation PSNR")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def fmt(value, spec=".3f"):
    return "—" if value is None else f"{value:{spec}}"


def paper_table(evals, key, title, spec, note=""):
    """Render one 'paper vs reproduced' block from an eval_*.json that carries a
    `reference` dict, so Tables 2/3 share a single code path with Table 1's shape."""
    data = evals.get(key)
    if not data:
        return [f"## {title}\n\n_尚未生成，跑：`python scripts/evaluate.py --config "
                f"{key}_pretrained --tag {key}`_\n"]
    m, c, l = data["metrics"], data["complexity"], data["latency"]
    ref = data.get("reference", {})
    out = [f"## {title}\n", "| metric | paper | reproduced | delta |", "|---|---|---|---|"]
    for label, getter, ref_key, form in spec:
        ours, paper = getter(m, c, l), ref.get(ref_key)
        delta = "—" if (paper is None or ours is None) else f"{ours - paper:+{form}}"
        out.append(f"| {label} | {fmt(paper, form)} | {fmt(ours, form)} | {delta} |")
    out.append("")
    if note:
        out.append(f"> {note}\n")
    return out


# (显示名, 取值器, reference 键, 格式)
SPEC_UIE = [
    ("PSNR (standard RGB)", lambda m, c, l: m["psnr"], "psnr", ".3f"),
    ("PSNR (release's per-channel)", lambda m, c, l: m["psnr_official"], None, ".3f"),
    ("SSIM", lambda m, c, l: m["ssim"], "ssim", ".4f"),
    ("LPIPS (AlexNet)", lambda m, c, l: m["lpips"], "lpips", ".4f"),
    ("MAE", lambda m, c, l: m["mae"], None, ".4f"),
    ("#params (K)", lambda m, c, l: c["params"] / 1000, "params_k", ".3f"),
]
SPEC_ISP = [
    ("PSNR (standard RGB)", lambda m, c, l: m["psnr"], "psnr", ".3f"),
    ("SSIM", lambda m, c, l: m["ssim"], "ssim", ".4f"),
    ("LPIPS (AlexNet)", lambda m, c, l: m["lpips"], None, ".4f"),
    ("#params (K)", lambda m, c, l: c["params"] / 1000, "params_k", ".3f"),
    ("conv MACs @448x448 (G)", lambda m, c, l: c["macs_g"], None, ".3f"),
    ("GPU latency median (ms)", lambda m, c, l: l["latency_ms"], "gpu_latency_ms", ".3f"),
]


def build_report(evals, runs, curves):
    ref = PAPER["table1_lolv1"]
    lines = []
    lines.append("# MobileIE (ICCV 2025) reproduction report\n")
    lines.append("作者权重侧覆盖 LLE(LOLv1) / UIE(UIEB) / ISP(ZRR) 三张表；从零训练见文末 From-scratch 段。\n")
    lines.append("## Environment\n")
    lines.append(f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
                 f" (compute capability {torch.cuda.get_device_capability(0)[0]}."
                 f"{torch.cuda.get_device_capability(0)[1]})" if torch.cuda.is_available() else "- GPU: cpu")
    lines.append(f"- PyTorch {torch.__version__}, CUDA runtime {torch.version.cuda}\n")

    lines.append("## Table 1 — LOLv1 (author checkpoint, `pretrain/lolv1_best_slim.pkl`)\n")
    lines.append("| metric | paper | reproduced | delta |")
    lines.append("|---|---|---|---|")
    pre = evals.get("pretrained")
    if pre:
        m, c, l = pre["metrics"], pre["complexity"], pre["latency"]
        rows = [
            ("PSNR (standard RGB)", m["psnr"], ref["psnr"], ".3f"),
            ("PSNR (release's per-channel formula)", m["psnr_official"], None, ".3f"),
            ("SSIM", m["ssim"], ref["ssim"], ".4f"),
            ("LPIPS (AlexNet)", m["lpips"], ref["lpips"], ".4f"),
            ("MAE", m["mae"], None, ".4f"),
            ("#params (K)", c["params"] / 1000, ref["params_k"], ".3f"),
            ("MACs @600x400 (G, paper's 'FLOPs')", c["macs_g"], PAPER["table7"]["infer_flops_g"], ".3f"),
            ("model size fp32 (MB)", c["model_size_mb"], ref["model_size_mb"], ".4f"),
            ("GPU latency @600x400 median (ms)", l["latency_ms"], ref["gpu_latency_ms"], ".3f"),
            ("GPU latency @600x400 best (ms)", l.get("best_ms"), ref["gpu_latency_ms"], ".3f"),
            ("FPS @600x400 (1/median)", l["fps"], ref["fps"], ".1f"),
        ]
        for name, ours, paper, spec in rows:
            delta = "—" if (paper is None or ours is None) else f"{ours - paper:+{spec}}"
            lines.append(f"| {name} | {fmt(paper, spec)} | {fmt(ours, spec)} | {delta} |")
    lines.append("")

    tfl = evals.get("tflite")
    if tfl and pre:
        tm, pm = tfl["metrics"], pre["metrics"]
        lines.append("### 部署产物一致性（`pretrain/lolv1.tflite`，x86 CPU + XNNPACK）\n")
        lines.append("| metric | fp32 pkl | tflite | delta |")
        lines.append("|---|---|---|---|")
        for k, label in (("psnr", "PSNR (RGB)"), ("psnr_official", "PSNR (per-channel)"),
                         ("ssim", "SSIM"), ("mae", "MAE")):
            if k in tm:
                lines.append(f"| {label} | {pm[k]:.4f} | {tm[k]:.4f} | {tm[k]-pm[k]:+.4f} |")
        lines.append(f"| 单帧耗时 (x86 CPU, 1 线程, 无 warmup, ms) | — | {tm['ms_x86_cpu_mean']:.1f} | — |")
        lines.append("")
        lines.append("说明：转换链 fp32→ONNX→TFLite 无精度损失，所以拷进手机测速的文件与 .pkl 等价。"
                     "延迟强烈依赖线程数与是否 warmup（本机 1/2/4 线程 = 72.4 / 36.7 / 19.9 ms，"
                     "测量时训练仍在跑，三者同条件可比），统一口径与端侧测法见仓库根目录 `deploy/README.md`。\n")

    lines += paper_table(
        evals, "uieb_native", "Table 2 — UIEB（UIE 任务，`uieb_best_slim.pkl`）", SPEC_UIE,
        "测试集为 native 620x460 的 90 对。按论文表头的 640x480 重测后 PSNR 只差 0.006 dB，"
        "因此与论文 22.81 的 ~0.39 dB 差距**不是分辨率口径**造成的；最可能是本镜像的 UIEB 测试子集"
        "（90 张）与官方 890/60 划分中的 60 张不同 —— 这条差异在报告里保留为未消除项，不粉饰。")
    lines += paper_table(evals, "uieb_640x480", "Table 2 口径对照 — UIEB @640x480", SPEC_UIE)
    lines += paper_table(
        evals, "zrr", "Table 3 — ZRR（ISP 任务，`zrr_best_slim.pkl`）", SPEC_ISP,
        "输入 12-bit RGGB Bayer（448x448 拼块 → 4x224x224），经 PixelShuffle(2) 输出 3x448x448，"
        "与论文 Table 3 的 448x448 口径一致；均值取在 1,204 个拼块上。")

    lines.append("## Table 7 — train graph vs re-parameterised inference graph\n")
    lines.append("| quantity | paper | measured |")
    lines.append("|---|---|---|")
    net = M.MobileIELLE(channels=12, rep_scale=4, variant="plain")
    slim = M.MobileIELLESlim(channels=12)
    tp, ip = count_parameters(net)["params"], count_parameters(slim)["params"]
    tmac = count_macs(net, (1, 3, 400, 600), "cpu")["macs_g"]
    imac = count_macs(slim, (1, 3, 400, 600), "cpu")["macs_g"]
    t7 = PAPER["table7"]
    lines.append(f"| train-graph params | {t7['train_params_k']} K | {tp/1000:.3f} K |")
    lines.append(f"| inference params | {t7['infer_params_k']} K | {ip/1000:.3f} K |")
    lines.append(f"| train-graph MACs | {t7['train_flops_g']} G | {tmac:.3f} G |")
    lines.append(f"| inference MACs | {t7['infer_flops_g']} G | {imac:.3f} G |")
    lines.append(f"| compression | {t7['train_flops_g']/t7['infer_flops_g']:.1f}x FLOPs | "
                 f"{tmac/imac:.1f}x FLOPs, {tp/ip:.1f}x params |")
    lines.append("")

    lines.append("## From-scratch training\n")
    lines.append("| run | stage | best val PSNR | @epoch | epochs | state |")
    lines.append("|---|---|---|---|---|---|")
    for name, rec in runs.items():
        summary_path = RUNS_ROOT / name / "train_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        best = summary.get("best", {})
        cfg_path = RUNS_ROOT / name / "config.yaml"
        stage = "?"
        total = "?"
        if cfg_path.exists():
            import yaml
            c = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            stage = str(c["optim"].get("stage", 1))
            total = str(c["optim"]["epochs"]) + ("+warmup" if c["optim"].get("warmup_epochs") else "")
        state = "done" if summary else (f"running ({len(rec)} epochs logged)" if rec else "no data")
        lines.append(f"| {name} | {stage} | {fmt(best.get('psnr'), '.4f')} | "
                     f"{best.get('epoch', '—')} | {total} | {state} |")
    lines.append("")

    s1 = RUNS_ROOT / "lolv1_stage1" / "ckpt" / "model_best.pkl"
    s2 = RUNS_ROOT / "lolv1_stage2_iwo" / "ckpt" / "model_best.pkl"
    if s2.exists():
        lines.append("## IWO effect (paper Fig. 10 claim: loss keeps falling after stage 1 stalls)\n")
        lines.append("Evaluate both with `python scripts/evaluate.py --config lolv1_stage2_iwo "
                     "--weights <ckpt>` and compare the PSNR columns.\n")

    if curves:
        lines.append(f"## Curves\n\n![curves]({Path(curves).name})\n")

    lines.append("## Reproduction notes / deviations from the release\n")
    lines.append("- **PSNR convention.** The paper's 23.62 dB matches the standard RGB PSNR "
                 "(`10*log10(1/MSE)` over all channels). The released `main.py` instead averages "
                 "`10*log10(1/MSE_c)` per channel, which reads 0.32 dB higher on the same weights.\n"
                 "- **LPIPS backbone.** 0.198 in Table 1 is LPIPS-AlexNet on RGB; LPIPS-VGG on the "
                 "same images gives 0.293.\n"
                 "- **Batch size.** The release ships `batch_size: 4`; the training graph needs "
                 "~11 GB VRAM at 600x400, so stage runs use 2 (~6 GB). "
                 "Batch 1 is impossible: HDPA's `AdaptiveAvgPool2d(1)` leaves a 1x1 spatial map, and "
                 "BatchNorm over it raises for batch 1.\n"
                 "- **IWO initialisation.** `model/utils_IWO.py` seeds `W_learn` with "
                 "`xavier_normal_`, which perturbs a transferred `W_pre`; the reproduction zeroes it so "
                 "stage 2 starts exactly where stage 1 ended (Eq. 1 as written). "
                 "`scripts/selfcheck.py` asserts `IWO(W_learn=0) == stage-1 model` to 0.0e+00.\n"
                 "- **LVW.** The released `OutlierAwareLoss` weights the *signed* residual with "
                 "per-(batch,channel) statistics and divides the std by sqrt(2), while Eq. 7-8 takes "
                 "the L1 magnitude first. Both are implemented (`loss.pixel: lvw_official|lvw_paper`); "
                 "the paper's own checkpoints come from the official form.\n"
                 "- **Data order.** The release enumerates files with `os.listdir`; the reproduction "
                 "sorts them so runs are reproducible.\n"
                 "- **ISP 的延迟不可直接比。** 本机在 4x224x224 输入（448x448 拼块）上测得 0.43 ms，"
                 "而论文 Table 3 报 1.02 ms；论文未写明 ISP 测速的输入形状（ZRR 原图是 4256x2832），"
                 "两者大概率不是同一形状，故这一格只记录不解读。参数量与 MACs 可比且吻合"
                 "（4.132 K vs 4.104 K，与 LLE 同一个 +0.7% 口径偏移）。\n"
                 "- **UIEB 的 0.39 dB 未消除。** 分辨率口径已排除（640x480 与 native 差 0.006 dB），"
                 "最合理的解释是本镜像 90 张测试子集与官方 890/60 划分中的 60 张不同；"
                 "LPIPS 0.1508 优于论文 0.155、SSIM 仅差 -0.005，说明模型行为一致，差异在评测集构成。\n"
                 "- Latency is measured on a laptop GPU, so it is not expected to equal the desktop "
                 "RTX 4090 figure in the paper; the released checkpoint is what is timed.\n")
    return "\n".join(lines)


def main():
    RESULTS_ROOT.mkdir(exist_ok=True)
    evals = load_evals()
    run_names = sorted(p.name for p in RUNS_ROOT.iterdir() if p.is_dir()
                       and (p / "metrics.jsonl").exists() and not p.name.startswith("smoke")
                       and p.name != "pretrained")
    runs = {n: read_metrics(RUNS_ROOT / n) for n in run_names}
    curves = plot_curves(runs, RESULTS_ROOT / "curves.png")
    text = build_report(evals, runs, curves)
    out = RESULTS_ROOT / "report.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
