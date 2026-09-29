# MobileIE (ICCV 2025) — reproduction

Paper: *MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image
Enhancement on Mobile Devices* (Yan et al., ICCV 2025), `../Yan_MobileIE_..._paper.pdf`.
Official code: `../MobileIE-main/MobileIE-main` (kept untouched, used as the reference
implementation of the re-parameterisation maths).

Scope reproduced here: the **LLE task on LOLv1** (paper Table 1, Table 7, Eq. 1-10,
Fig. 10) — author-checkpoint verification plus full two-stage training from scratch.
UIE/UIEB and ISP/ZRR: the author-weight side is verified (Tables 2 and 3, both datasets
downloaded via hf-mirror); training them from scratch is not yet scheduled — ZRR is
missing `train/raw`, so ISP from scratch needs another data source.

**逐条对照"论文怎么说 → 我们怎么做 → 结果对不对得上"见 [`论文复现对照.md`](论文复现对照.md)**
（三表核对、复杂度账本、从零两阶段进度、端侧部署、论文没写的 12 项假设、未做清单、每个数字的复现命令）。
**核心算法拆解与复现流程见 [`算法复现说明.md`](算法复现说明.md)**（公式↔代码对照、
折叠数学、7 步流程与判据、踩坑清单、我做的假设）。
**在这台笔记本上怎么排期、每件事花多久见 [`本地训练计划.md`](本地训练计划.md)**（数字全部本机实测）。
**云端跑法见 [`../cloud/README.md`](../cloud/README.md)**，一键入口是仓库根的 `run.cmd`。

## Layout

```
repro/
  configs/          lolv1_stage1 / lolv1_stage2_iwo / lolv1_pretrained / lolv1_smoke*
                    uieb_pretrained (Table 2)   zrr_pretrained (Table 3)
  data/LOLv1/       our485/{low,high} (485 pairs), eval15/{low,high} (15), 600x400 PNG
  data/UIEB/        train/{input,gt} (800 pairs), test/{input,gt} (90)  - an 800/90 re-split
  data/ZRR/         test/{raw,srgb} (1204 tiles, 448x448, raw = 16-bit PNG Bayer)
                    train 46839 + test 1204 tiles of 448x448; mixed PNG bit depth is benign
  src/mobileie/     the reproduction layer
    config.py       YAML config, dotted CLI overrides, run directories, path resolution
    model.py        train/inference graphs; variant plain|iwo, task lle (3->3) | isp (RGGB->RGB)
    data.py         RGB pairs (+ optional crop / resize) and 12-bit Bayer pairs
    losses.py       LVW (both the released and the Eq. 7-10 reading) + warm-up objective
    metrics.py      PSNR (two conventions) / SSIM / LPIPS / MAE
    complexity.py   params, conv MACs, per-iteration latency (shape from eval.shape)
    engine.py       warm-up, training, IWO transfer, folding, state_last resume, evaluation
  scripts/
    download_lolv1.py / download_uieb_zrr.py   datasets through hf-mirror.com
    selfcheck.py          15 assertions: folding, IWO, loader parity, crop alignment
    train.py              one stage;  --set dotted.key=value;  --resume <run> to continue
    evaluate.py           metrics + complexity + latency for any checkpoint
    progress.py           progress / s-per-epoch / ETA, computed from files on disk only
    verify_dataset.py     dataset gate: pairing, decodability, sizes, true >8-bit RAW
    verify_tflite.py      the deployable .tflite vs the .pkl it came from
    report.py             writes results/report.md (Tables 1/2/3/7) and curves.png
    run_two_stage.cmd/.sh stage chain with crash-auto-resume (MAXRETRY)
  runs/             per-run config.yaml, ckpt/, metrics.jsonl, train.log, images/
  results/          report.md, curves.png, eval_*.json
```

## Setup

One command picks and installs the CUDA wheel line that matches the local NVIDIA
driver, creates the venv, fetches LOLv1 through `hf-mirror.com` and runs the self-checks:

```powershell
powershell -ExecutionPolicy Bypass -File repro\setup.ps1            # Windows
```
```bash
bash repro/setup.sh                                                 # Linux / WSL2
```

Useful switches: `-TorchIndex cu121|cu126|cu128|cpu`, `-PypiMirror cn`, `-SkipDownload`,
`-SkipCheck` (and the `TORCH_INDEX= / PYPI_MIRROR= / SKIP_DOWNLOAD=1` env equivalents on bash).
`requirements.txt` deliberately does **not** pin torch - the right build depends on the driver.

Driver → wheel line used by the script: `RTX 50-series → cu128` (sm_120 needs it),
driver ≥ 570.62 → cu128, ≥ 528 → cu121, ≥ 450 → cu118, otherwise CPU.

## Moving to another machine

1. Copy the **whole `MobileIE` folder**, not just `repro/`: `src/mobileie/model.py` loads the
   re-parameterisation maths from `../MobileIE-main/MobileIE-main/model/utils*.py` by path and
   refuses to run otherwise (the error says exactly this).
2. Do **not** copy `.venv` — it is machine-specific. Re-run `setup.ps1`/`setup.sh`.
3. Safe to copy as-is: `data/LOLv1` (350 MB, skip the download with `-SkipDownload`),
   `runs/*` (checkpoints + metrics), `results/*`, `configs/`.
4. Skip `third_party/AI_Benchmark_V7.0.0.apk` (1.97 GB) — it is the phone-benchmark installer,
   not part of training.
5. Batch size by VRAM: the release ships `batch_size: 4`, which needs ~11 GB at 600x400.
   `setup` prints a suggested value; override per run with
   `--set optim.batch_size=4`. Measured on this 5070 Ti Laptop: bs=2 is 0.061 s/sample while
   bs=4 is 0.0955 s/sample, so the smaller batch is *faster* here (the multi-branch concat is
   bandwidth-bound) - on a 24 GB card with bs=4 expect fewer, longer steps, not a free speedup.
6. Nothing in `configs/` uses an absolute path, so the project works unchanged at a different
   drive letter or mount point.

## Reproduce

**Windows 一键入口在项目根目录：`run.cmd`**
`start` 启动两阶段（5 秒倒计时可取消，已有链路存活则拒绝）· `status`/`watch` 终端进度与 ETA ·
`stop` 保留断点停止 · `eval`/`verify`/`report` ·
**`board` 起一个实时控制台** `http://127.0.0.1:8765`（只绑本机，因为它能启停进程）：
按钮手动触发训练/冒烟/评估/停止，页面每 3 秒拉真实状态，动态画 loss 与 val PSNR 曲线、
进度条、剩余时间与预计完成时刻、GPU 利用率/显存/功耗/温度、日志尾部。
数据源与 `run.cmd status`、`report.py` 完全相同（都只读 `runs/*/metrics.jsonl`），不会互相对不上。
细节与时间预算见 [`本地训练计划.md`](本地训练计划.md)，云端跑法见 [`../cloud/README.md`](../cloud/README.md)。

```bash
# 1. author checkpoint vs Table 1  (~1 min)
python scripts/evaluate.py --config lolv1_pretrained --tag pretrained --save-images

# 2. from scratch: stage 1 (1000 epochs) then IWO stage 2 (1000 epochs), unattended
cmd //c 'start "" /min scripts\run_two_stage.cmd'      # ~19 h on a 5070 Ti Laptop
tail -a logs/stage1.log                                 # watch progress

# or one stage at a time
python scripts/train.py --config lolv1_stage1      --run-name lolv1_stage1
python scripts/train.py --config lolv1_stage2_iwo  --run-name lolv1_stage2_iwo

# 3. score the produced checkpoints and rebuild the report
python scripts/evaluate.py --config lolv1_stage1      --run-name lolv1_stage1      --tag stage1
python scripts/evaluate.py --config lolv1_stage2_iwo  --run-name lolv1_stage2_iwo  --tag stage2
python scripts/report.py
```

## Resuming an interrupted run

Each epoch rewrites `runs/<name>/ckpt/state_last.pt` (~73 KB): weights, Adam moments, the
cosine-restart cycle position, the best-model tracker and the CPU/CUDA RNG streams.

```bash
python scripts/train.py --config lolv1_stage1 --resume lolv1_stage1     # manual
cmd //c 'start "" /min scripts\run_two_stage.cmd'   # automatic: a stage that dies is
bash  scripts/run_two_stage.sh                      # relaunched from its state file, x MAXRETRY
```

Verified on the smoke config: 2 epochs + resume-2 epochs continues the loss curve without
bouncing (1.904 -> 1.730) and carries the earlier `best` forward, so a kill costs at most the
in-flight epoch.

Two honesty notes:

- Strict reproducibility is **off by default**: `cudnn.benchmark` autotunes conv algorithms, so
  two identically-seeded runs already differ by ~1e-4 in the epoch-1 loss (the released code has
  no determinism controls either). Set `optim.cudnn_benchmark: false` for bit-identical reruns.
- A resumed run therefore continues the same *optimisation state*, not the same bit stream.

## What already matches (author checkpoint, `pretrain/lolv1_best_slim.pkl`, eval15)

| | paper Table 1 | measured |
|---|---|---|
| PSNR | 23.62 | **23.634** |
| SSIM | 0.812 | **0.8125** |
| LPIPS | 0.198 | **0.1976** (AlexNet, RGB) |
| #params | 4.047 K | 4.075 K |
| FLOPs (conv MACs) | 0.924 G | 0.919 G |
| GPU latency 600x400 | 0.895 ms (RTX 4090) | 1.204 ms per-frame-sync / 0.994 ms pipelined (RTX 5070 Ti Laptop, idle) |
| FPS 600x400 | 1120.6 | 830 / 1006 (same two timing conventions) |
| re-parameterisation | folds exactly | `max|dy| = 2.1e-7` (selfcheck) |

## Findings worth knowing

- **PSNR convention**: the paper's 23.62 dB is standard RGB PSNR. The released `main.py`
  averages per-channel PSNR, which reads 0.32 dB higher for the same weights.
- **LPIPS backbone**: 0.198 is LPIPS-AlexNet; LPIPS-VGG on the same output gives 0.293.
- **Batch size >= 2 is mandatory** for the train graph: HDPA's `AdaptiveAvgPool2d(1)` reduces
  the map to 1x1, so BatchNorm over it raises `Expected more than 1 value per channel` at bs=1.
  This also means the release cannot run validation while `net.train()` is active.
- **IWO start value**: `model/utils_IWO.py` initialises the additive `W_learn` with
  `xavier_normal_`, so loading a stage-1 checkpoint perturbs it. Eq. 1 says
  `W_final = Frozen(W_pre) + W_learn`, so the reproduction zeroes `W_learn`
  (`model.iwo_learn_init: zero`) and asserts that stage 2 then starts bit-identical to
  stage 1 (`selfcheck.py`).
- **LVW**: the released `OutlierAwareLoss` weights the signed residual with per-(batch,channel)
  statistics divided by sqrt(2), while Eq. 7-8 take the L1 magnitude first. Both are available
  as `loss.pixel`.
- **`model/__init__.py` in the release never imports `utils_IWO.py`** — `model/lle.py` and
  `model/isp.py` both `from .utils import ...`, so the shipped training entry point cannot
  actually run the IWO stage. The reproduction wires it up in `mobileie/model.py`.
- The official `main.py` hard-codes `os.environ['CUDA_VISIBLE_DEVICES'] = '5'`, which hides the
  GPU on a single-GPU machine; this harness never sets it.

## Measured cost

Full-resolution 600x400, no crops (as released): 8.2 it/s at batch 2 (~6.0 GB VRAM) → ~33 s per
epoch over our485 → stage 1 ≈ 9.3 h, IWO stage 2 ≈ 9.3 h on an RTX 5070 Ti Laptop. The release's
batch 4 needs ~11.0 GB.
