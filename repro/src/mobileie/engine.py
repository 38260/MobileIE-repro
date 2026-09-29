"""Training / evaluation driver.

Reproduces the recipe of Sec. 4.1: 10-epoch low-lr warm-up, Adam +
cosine-annealing-with-warm-restarts, then the Incremental Weight Optimisation
stage that freezes ``W_pre`` and trains the additive ``W_learn`` (Eq. 1-2).
Checkpoint selection follows the released ``main.py`` (best validation PSNR).
"""
import contextlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from . import model as M
from .data import make_loaders
from .losses import build_loss
from .metrics import Evaluator


def amp_ctx(device: torch.device, mode: str):
    """bf16 halves the activation traffic of the multi-branch train graph; autocast
    keeps BatchNorm in fp32, which matters because BN running stats feed the fold."""
    if device.type != "cuda" or mode in (None, "none", "fp32"):
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[mode]
    return torch.autocast("cuda", dtype=dtype)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] '{name}' requested but no CUDA device visible, falling back to cpu")
        return torch.device("cpu")
    return torch.device(name)


def to_device(batch, device):
    return tuple(t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)


@torch.no_grad()
def validate(net, loader, device, clip: bool = False) -> dict:
    """Validation PSNR, computed exactly like the released ``main.py`` (raw output)."""
    net.eval()
    values = []
    for inp, gt, _ in loader:
        inp, gt = inp.to(device), gt.to(device)
        out = net(inp)
        values.append(float((1 / ((out - gt) ** 2).mean((2, 3))).log10().mean() * 10))
    net.train()
    return {"psnr": float(np.mean(values)), "n": len(values)}


def build_optimizer(net, cfg, lr=None):
    o = cfg["optim"]
    return torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad],
        lr=lr if lr is not None else float(o["lr"]),
        weight_decay=float(o.get("weight_decay", 0)),
    )


def build_scheduler(optim, cfg):
    s = cfg["optim"]["scheduler"]
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, T_0=int(s["t0"]), T_mult=int(s["t_mult"]), eta_min=float(s["eta_min"])
    )


def save_state(path: Path, net, optim, sched, epoch: int, phase: str, best: dict, scaler=None):
    """Everything needed to continue: weights, Adam moments, cosine-restart cycle
    position, the best-model tracker and the RNG streams that drive crop order."""
    payload = {
        "model": net.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "epoch": epoch,
        "phase": phase,
        "best": best,
        "rng_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["rng_cuda"] = torch.cuda.get_rng_state_all()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_state(path: Path, net, optim, sched, scaler=None):
    st = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(st["model"])
    optim.load_state_dict(st["optim"])
    sched.load_state_dict(st["sched"])
    if scaler is not None and "scaler" in st:
        scaler.load_state_dict(st["scaler"])
    torch.set_rng_state(st["rng_cpu"])
    if torch.cuda.is_available() and "rng_cuda" in st:
        torch.cuda.set_rng_state_all(st["rng_cuda"])
    return int(st["epoch"]), str(st["phase"]), dict(st.get("best") or {"psnr": -1.0, "epoch": -1})


def run_warmup(net, loaders, cfg, run, device, logger):
    epochs = int(cfg["optim"].get("warmup_epochs", 0))
    if epochs <= 0:
        return
    train_loader, valid_loader = loaders
    _, warmup_loss = build_loss(cfg)
    optim = build_optimizer(net, cfg, lr=float(cfg["optim"]["lr_warmup"]))
    amp = cfg["optim"].get("amp", "none")
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16" and device.type == "cuda"))
    logger(f"warm-up start: {epochs} epochs @ lr={cfg['optim']['lr_warmup']}")
    for epo in range(epochs):
        losses = []
        for inp, gt, _ in tqdm(train_loader, ncols=80, desc=f"warmup {epo+1}/{epochs}", leave=False):
            inp, gt = to_device((inp, gt), device)
            optim.zero_grad(set_to_none=True)
            with amp_ctx(device, amp):
                out1, out2 = net.forward_warm(inp)
                loss = warmup_loss(inp, gt, out1, out2)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            losses.append(loss.detach().item())
        val = validate(net, valid_loader, device)
        logger(f"warmup epoch: {epo+1}, loss: {np.mean(losses):.6f}, val_psnr: {val['psnr']:.4f}")
        run.append_metric({"phase": "warmup", "epoch": epo + 1,
                           "train_loss": float(np.mean(losses)), "val_psnr": val["psnr"]})
    torch.save(net.state_dict(), run.root / "ckpt" / "model_warmup.pkl")
    logger("warm-up done")


def transfer_stage1_to_iwo(net_iwo: torch.nn.Module, ckpt_path: str, logger=print):
    """Load W_pre into the frozen conv_out weights; W_learn starts at zero."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = net_iwo.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint has keys the IWO model does not know: {unexpected[:5]}")
    iwo_keys = sorted(k for k in missing if k.endswith("weight1"))
    other = sorted(set(missing) - set(iwo_keys))
    if other:
        raise RuntimeError(f"stage-1 checkpoint is missing parameters: {other[:5]}")
    zeroed = 0
    for name, module in net_iwo.named_modules():
        if hasattr(module, "weight1") and torch.is_tensor(getattr(module, "weight1")) \
                and getattr(module, "weight1", None) is not None and module.weight1.dim() == 4:
            with torch.no_grad():
                module.weight1.zero_()
            zeroed += 1
    frozen = sum(1 for _, p in net_iwo.named_parameters() if not p.requires_grad)
    logger(f"[IWO] transferred {len(iwo_keys)} W_learn slots (zeroed {zeroed}), "
           f"frozen prior weights kept at stage-1 optimum")
    return {"iwo_slots": len(iwo_keys), "zeroed": zeroed, "frozen_conv_out": frozen}


def train(cfg: dict, run, device_name: str | None = None):
    device = resolve_device(device_name or cfg["device"])
    set_seed(int(cfg.get("seed", 0)))
    # Autotuned conv kernels pick different algorithms per run, so the loss stream is
    # not bit-reproducible even with a fixed seed (measured: 1e-4 by epoch 1).  Set
    # optim.cudnn_benchmark: false for strict determinism at a small speed cost.
    benchmark = bool(cfg["optim"].get("cudnn_benchmark", True))
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = not benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    log_lines = []

    def logger(message: str):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        log_lines.append(line)
        with open(run.log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    run.write_config()
    stage = cfg["optim"].get("stage", 1)
    net = M.build(cfg, variant="iwo" if stage >= 2 else "plain").to(device)

    if stage >= 2:
        init = cfg["optim"].get("init_from")
        if not init:
            raise ValueError("stage 2 needs optim.init_from pointing at the stage-1 checkpoint")
        info = transfer_stage1_to_iwo(net, init, logger)
        run.write_json("iwo_transfer.json", {"stage1_ckpt": init, **info})

    train_loader, valid_loader = make_loaders(cfg)
    logger(f"train pairs: {len(train_loader.dataset)}, valid pairs: {len(valid_loader.dataset)}")

    loss_fn, _ = build_loss(cfg)
    logger(f"network: {type(net).__name__} variant={cfg['model']['type'] if stage < 2 else 'iwo'} "
           f"params={M.count_parameters(net)/1000:.3f}K")

    net.train()
    epochs = int(cfg["optim"]["epochs"])
    optim = build_optimizer(net, cfg)
    sched = build_scheduler(optim, cfg)
    grad_clip = cfg["optim"].get("grad_clip")
    save_every = int(cfg["optim"].get("save_every", 50))
    amp = cfg["optim"].get("amp", "none")
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16" and device.type == "cuda"))
    logger(f"mixed precision: {amp}")

    state_path = run.root / "ckpt" / "state_last.pt"
    resume_from = cfg["optim"].get("resume_from")
    start_epoch, best = 0, {"psnr": -1.0, "epoch": -1}
    if resume_from:
        src = Path(resume_from)
        if not src.exists():
            raise FileNotFoundError(f"optim.resume_from does not exist: {src}")
        done, phase, best = load_state(src, net, optim, sched, scaler)
        start_epoch = done + 1
        logger(f"resumed {src}: {done+1} epochs done, best val psnr {best['psnr']:.4f} "
               f"@ epoch {best['epoch']}, continuing at epoch {start_epoch+1}")
    elif stage == 1:
        run_warmup(net, (train_loader, valid_loader), cfg, run, device, logger)
        # written now so a crash in the first training epochs is still recoverable
        save_state(state_path, net, optim, sched, -1, "warmup", best, scaler)

    logger(f"training start: {epochs} epochs, lr={cfg['optim']['lr']}, batch={cfg['optim']['batch_size']}")
    for epo in range(start_epoch, epochs):
        losses, parts_sum, parts_n = [], {}, 0
        for inp, gt, _ in tqdm(train_loader, ncols=80, desc=f"epoch {epo+1}/{epochs}", leave=False):
            inp, gt = to_device((inp, gt), device)
            optim.zero_grad(set_to_none=True)
            with amp_ctx(device, amp):
                out = net(inp)
                loss = loss_fn(out, gt)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optim)
                clip_grad_norm_(net.parameters(), float(grad_clip))
            scaler.step(optim)
            scaler.update()
            losses.append(loss.detach().item())
            if hasattr(loss_fn, "forward_parts"):
                for key, val in loss_fn.forward_parts(out, gt).items():
                    parts_sum[key] = parts_sum.get(key, 0.0) + val
                parts_n += 1
        sched.step()

        val = validate(net, valid_loader, device)
        mean_loss = float(np.mean(losses))
        parts = {k: v / parts_n for k, v in parts_sum.items()} if parts_n else {}
        logger(f"epoch: {epo+1}, loss: {mean_loss:.6f}, val_psnr: {val['psnr']:.4f}, "
               f"lr: {optim.param_groups[0]['lr']:.3e}"
               + (f", pixel {parts['loss_pixel']:.4f}, cos {parts['loss_cos']:.4f}, "
                  f"psnr_term {parts['loss_psnr_term']:.4f}" if parts else ""))
        run.append_metric({"phase": f"train_s{stage}", "epoch": epo + 1, "train_loss": mean_loss,
                           "val_psnr": val["psnr"], "lr": optim.param_groups[0]["lr"], **parts})

        if (epo + 1) % save_every == 0:
            torch.save(net.state_dict(), run.root / "ckpt" / f"model_{epo+1}.pkl")
        if val["psnr"] > best["psnr"]:
            best = {"psnr": val["psnr"], "epoch": epo + 1}
            torch.save(net.state_dict(), run.root / "ckpt" / "model_best.pkl")
            if cfg["optim"].get("save_slim", True):
                slim = net.slim().to(device)
                torch.save(slim.state_dict(), run.root / "ckpt" / "model_best_slim.pkl")
            logger(f"  -> new best (epoch {epo+1}), checkpoint + re-parameterised copy saved")
        save_state(state_path, net, optim, sched, epo, "train", best, scaler)

    final = {"best": best, "stage": stage, "params": M.count_parameters(net),
             "epochs_planned": epochs, "resumed_from": resume_from}
    torch.save(net.state_dict(), run.root / "ckpt" / "model_last.pkl")
    run.write_json("train_summary.json", final)
    logger(f"training done, best val psnr {best['psnr']:.4f} @ epoch {best['epoch']}")
    return net, final


@torch.no_grad()
def evaluate(net, cfg, device_name, run=None, split="test", save_images=False):
    device = resolve_device(device_name or cfg["device"])
    net = net.to(device).eval()
    loader = make_loaders({**cfg, "split": split})
    ev = Evaluator(device=str(device), lpips_net=cfg["eval"].get("lpips_net", "vgg"),
                   use_lpips=cfg["eval"].get("use_lpips", True),
                   clip=cfg["eval"].get("clip", True))
    for inp, gt, name in tqdm(loader, ncols=80, desc="evaluate"):
        inp = inp.to(device)
        gt = gt.to(device)
        ev.update(net(inp), gt, name[0])
    summary = ev.summarize()
    if run is not None:
        run.write_json(f"eval_{split}.json", summary)
    return summary
