"""Parameter / MAC / latency accounting for the Table 1 and Table 7 numbers."""
import time

import torch
import torch.nn as nn


def count_parameters(net: nn.Module) -> dict:
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in net.buffers())
    return {"params": total, "trainable_params": trainable, "buffers": buffers,
            "model_size_mb": total * 4 / 1024 / 1024}


def _conv_mac(module, inp, out):
    kh, kw = module.kernel_size
    per_output = kh * kw * (module.in_channels // module.groups)
    module.__mac__ = float(out.numel() * per_output)


def count_macs(net: nn.Module, input_shape=(1, 3, 400, 600), device="cpu") -> dict:
    """Convolution-only MAC count, the convention restoration papers use."""
    net = net.to(device).eval()
    handles = []
    for module in net.modules():
        module.__mac__ = 0.0
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(_conv_mac))
    with torch.no_grad():
        net(torch.zeros(*input_shape, device=device))
    for h in handles:
        h.remove()
    macs = sum(m.__mac__ for m in net.modules() if isinstance(m, nn.Conv2d))
    for module in net.modules():
        if hasattr(module, "__mac__"):
            del module.__mac__
    return {"macs": macs, "flops_2mac": 2 * macs,
            "macs_g": macs / 1e9, "flops_g": 2 * macs / 1e9}


@torch.no_grad()
def measure_latency(net: nn.Module, input_shape=(1, 3, 400, 600), device="cuda",
                    warmup: int = 20, iters: int = 100) -> dict:
    """Per-iteration latency.  Report min/median as well as the mean: a laptop GPU
    that is also driving the display throttles, and then only the mean moves."""
    dev = torch.device(device)
    net = net.to(dev).eval()
    x = torch.zeros(*input_shape, device=dev)
    for _ in range(warmup):
        net(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    stamps = []
    for _ in range(iters):
        t0 = time.perf_counter()
        net(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        stamps.append(time.perf_counter() - t0)
    stamps.sort()
    median = stamps[len(stamps) // 2]
    return {"latency_ms": median * 1000, "best_ms": stamps[0] * 1000,
            "mean_ms": sum(stamps) / len(stamps) * 1000,
            "fps": 1 / median, "device": str(dev), "shape": list(input_shape),
            "warmup": warmup, "iters": iters}


def describe(net: nn.Module, input_shape=(1, 3, 400, 600), device="cuda") -> dict:
    info = count_parameters(net)
    info.update(count_macs(net, input_shape, device="cpu"))
    info["resolution"] = list(input_shape)
    return info
