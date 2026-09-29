"""Correctness self-checks for the reproduction layer.

Runs in seconds and proves three things the whole reproduction rests on:
  1. the released MBRConv modules load and fold into a plain conv exactly
     (train-graph output == re-parameterised output, which is the paper's
     central efficiency claim);
  2. IWO's W_final = Frozen(W_pre) + W_learn really does start at W_pre;
  3. the re-implemented LOLv1 loader returns the same tensors as the released
     ``data/lledata.py``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MobileIE-main" / "MobileIE-main"))

import torch  # noqa: E402

from mobileie import model as M  # noqa: E402
from mobileie.config import OFFICIAL_ROOT, load_config  # noqa: E402
from mobileie.data import PairedImageDataset  # noqa: E402
from mobileie.losses import build_loss  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


def main():
    cfg = load_config("lolv1_smoke")
    torch.manual_seed(0)
    x = torch.rand(1, 3, 96, 128)

    print(f"official repo: {OFFICIAL_ROOT} (exists: {OFFICIAL_ROOT.exists()})")

    # 1. re-parameterisation equivalence, plain modules
    net = M.MobileIELLE(channels=12, rep_scale=4, variant="plain").eval()
    slim = net.slim().eval()
    with torch.no_grad():
        a, b = net(x), slim(x)
    diff = (a - b).abs().max().item()
    check("MBRConv folds to single conv (plain)", diff < 2e-5, f"max|dy|={diff:.2e}")
    check("slim parameter count ~4K", abs(M.count_parameters(slim) - 4047) < 100,
          f"{M.count_parameters(slim)} params")
    check("train graph parameter count ~47K", abs(M.count_parameters(net) - 47211) < 4000,
          f"{M.count_parameters(net)} params")

    # 2. IWO: zero W_learn must reproduce the stage-1 network bit for bit
    stage1 = M.MobileIELLE(channels=12, rep_scale=4, variant="plain").eval()
    sd = stage1.state_dict()
    iwo = M.MobileIELLE(channels=12, rep_scale=4, variant="iwo", iwo_learn_init="zero")
    missing, unexpected = iwo.load_state_dict(sd, strict=False)
    check("stage-1 ckpt loads into IWO graph", not unexpected,
          f"missing={len(missing)} unexpected={len(unexpected)}")
    iwo.eval()
    with torch.no_grad():
        y1, y2 = stage1(x), iwo(x)
    check("IWO(W_learn=0) == stage-1 model", (y1 - y2).abs().max().item() < 1e-6,
          f"max|dy|={(y1-y2).abs().max().item():.2e}")
    frozen = [n for n, p in iwo.named_parameters() if not p.requires_grad and n.endswith("conv_out.weight")]
    # FST also owns a scalar `weight1`, so filter on the 4-D conv_out-shaped delta.
    trainable = [n for n, p in iwo.named_parameters()
                 if p.requires_grad and n.endswith("weight1") and p.dim() == 4]
    check("prior weights frozen / W_learn trainable", len(frozen) == 7 and len(trainable) == 7,
          f"frozen={len(frozen)} learnable={len(trainable)}")
    with torch.no_grad():
        for _, m in iwo.named_modules():
            if getattr(m, "weight1", None) is not None and torch.is_tensor(m.weight1) and m.weight1.dim() == 4:
                m.weight1.add_(0.01)
                break
        y3 = iwo(x)
    check("W_learn actually drives the output", (y3 - y2).abs().max().item() > 1e-6)

    # 3. loader parity with the released LLEData
    inp = str(Path(cfg["data"]["train_inp"]).resolve())
    gt = str(Path(cfg["data"]["train_gt"]).resolve())
    mine = PairedImageDataset(inp, gt, cache=True)

    from types import SimpleNamespace
    from data.lledata import LLEData  # official
    theirs = LLEData(SimpleNamespace(device="cpu"), inp, gt)

    same_order = sorted(theirs.img_li) == [n + ".png" for n in mine.names]
    check("loader enumerates the same files", same_order,
          f"mine={len(mine)} theirs={len(theirs.img_li)}")
    idx = 0
    ti, tg, tname = theirs[idx]
    mi, mg, mname = mine[idx]
    check("loader returns identical pixels",
          torch.allclose(ti, mi, atol=1e-6) and torch.allclose(tg, mg, atol=1e-6) and mname == tname,
          f"{mname} max|d|={(ti-mi).abs().max().item():.1e}")

    # 4. a training crop must slice input and ground truth at the SAME offset
    real_randint = torch.randint
    torch.randint = lambda *a, **k: torch.tensor([7])
    try:
        crop_ds = PairedImageDataset(inp, gt, cache=True, crop=64)
        ci, cg, _ = crop_ds[0]
    finally:
        torch.randint = real_randint
    fi, fg, _ = PairedImageDataset(inp, gt, cache=True)[0]
    aligned = (torch.equal(ci, fi[:, 7:71, 7:71]) and torch.equal(cg, fg[:, 7:71, 7:71]))
    check("train crop aligned across inp/gt", aligned, f"crop shape={tuple(ci.shape)}")

    # 5. losses are finite and shaped like the release's
    loss_fn, warm_fn = build_loss(cfg)
    out = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)
    for pixel in ("lvw_official", "lvw_paper", "l1", "l2"):
        c = dict(cfg)
        c["loss"] = {"pixel": pixel}
        lf, _ = build_loss(c)
        l = lf(out, target)
        check(f"loss {pixel} finite", bool(torch.isfinite(l)), f"value={float(l):.5f}")
    wl = warm_fn(torch.rand(2, 3, 32, 32), target, out, torch.rand(2, 3, 32, 32))
    check("warm-up loss finite", bool(torch.isfinite(wl)), f"value={float(wl):.5f}")

    print()
    print("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
