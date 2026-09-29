"""MobileIE-LLE network.

The re-parameterisation maths (``MBRConv{1,3,5}.slim``) is delicate, so it is not
re-implemented here: the official module files are loaded straight from the
released repository and used as-is.  Only the *topology* is rebuilt so that the
plain modules (paper Fig. 2 "Training Stage") and the IWO modules
(``utils_IWO.py``, paper Sec. 3.2 / Eq. 1-2) become interchangeable, which the
official ``model/lle.py`` cannot do because it hard-codes ``from .utils import``.
"""
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

from .config import OFFICIAL_ROOT

VARIANTS = ("plain", "iwo")


def _load_module(alias: str, filename: str):
    path = OFFICIAL_ROOT / "model" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"official module not found: {path}\n"
            "the reproduction depends on the released repo being unpacked next to repro/"
        )
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_MODULES = {
    "plain": lambda: _load_module("mobileie_official_utils", "utils.py"),
    "iwo": lambda: _load_module("mobileie_official_utils_iwo", "utils_IWO.py"),
}
_CACHE: dict = {}


def modules(variant: str = "plain"):
    if variant not in VARIANTS:
        raise ValueError(f"unknown module variant {variant!r}, choose from {VARIANTS}")
    if variant not in _CACHE:
        _CACHE[variant] = _MODULES[variant]()
    return _CACHE[variant]


class MobileIELLE(nn.Module):
    """Train-time MobileIE.

    `task` selects the I/O contract only, exactly as the release splits
    ``model/lle.py`` (3-ch RGB in, 3-ch RGB out) and ``model/isp.py``
    (4-ch RGGB half-res in, PixelShuffle -> 3-ch full-res out); every MBRConv /
    FST / HDPA module is shared.
    """

    def __init__(self, channels: int = 12, rep_scale: int = 4, variant: str = "plain",
                 iwo_learn_init: str = "zero", task: str = "lle"):
        super().__init__()
        m = modules(variant)
        self._m = m
        self.channels = channels
        self.rep_scale = rep_scale
        self.variant = variant
        self.task = task
        in_ch, out_ch = (3, 3) if task == "lle" else (4, 3)
        if task not in ("lle", "isp"):
            raise ValueError(f"unknown task {task!r}, choose from [lle, isp]")

        self.head = m.FST(
            nn.Sequential(
                m.MBRConv5(in_ch, channels, rep_scale=rep_scale),
                nn.PReLU(channels),
                m.MBRConv3(channels, channels, rep_scale=rep_scale),
            ),
            channels,
        )
        self.body = m.FST(m.MBRConv3(channels, channels, rep_scale=rep_scale), channels)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            m.MBRConv1(channels, channels, rep_scale=rep_scale),
            nn.Sigmoid(),
        )
        self.att1 = nn.Sequential(
            m.MBRConv1(1, channels, rep_scale=rep_scale),
            nn.Sigmoid(),
        )
        if task == "isp":
            # 12 ch at half resolution -> PixelShuffle -> 3 ch at full resolution -> refine
            self.tail = nn.Sequential(nn.PixelShuffle(2), m.MBRConv3(3, out_ch, rep_scale=rep_scale))
            self.tail_warm = m.MBRConv3(channels, in_ch, rep_scale=rep_scale)
        else:
            self.tail = m.MBRConv3(channels, out_ch, rep_scale=rep_scale)
            self.tail_warm = m.MBRConv3(channels, out_ch, rep_scale=rep_scale)
        self.drop = m.DropBlock(3)

        if variant == "iwo" and iwo_learn_init == "zero":
            # Eq. 1 reads W_final = Frozen(W_pre) + W_learn.  The released
            # utils_IWO.py seeds W_learn with xavier_normal_, which would inject
            # noise on top of a transferred W_pre; zero keeps stage-2's starting
            # point numerically identical to stage-1's endpoint.
            for mod in self.modules():
                if hasattr(mod, "weight1") and getattr(mod, "weight1", None) is not None:
                    if isinstance(mod.weight1, nn.Parameter) and mod.weight1.dim() == 4:
                        nn.init.zeros_(mod.weight1)

    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)
        x3 = self.att1(max_out)
        x4 = torch.mul(x2, x3) * x1
        return self.tail(x4)

    def forward_warm(self, x):
        x = self.drop(x)
        x = self.head(x)
        x = self.body(x)
        return self.tail(x), self.tail_warm(x)

    def iwo_state(self) -> bool:
        return self.variant == "iwo"

    def slim(self):
        """Fold every MBRConv branch set into one conv -> inference network."""
        m = self._m
        net_slim = MobileIELLESlim(self.channels, task=self.task)
        target = net_slim.state_dict()
        mbr_types = (m.MBRConv1, m.MBRConv3, m.MBRConv5)
        merged = []
        for name, mod in self.named_modules():
            if isinstance(mod, mbr_types):
                if f"{name}.weight" not in target:
                    continue  # tail_warm only exists in the train-time graph
                weight, bias = mod.slim()
                target[f"{name}.weight"] = weight
                target[f"{name}.bias"] = bias
                merged.append(name)
            elif isinstance(mod, m.FST):
                target[f"{name}.bias"] = mod.bias.detach()
                target[f"{name}.weight1"] = mod.weight1.detach()
                target[f"{name}.weight2"] = mod.weight2.detach()
            elif isinstance(mod, nn.PReLU):
                target[f"{name}.weight"] = mod.weight.detach()

        tail_name = "tail" if self.task == "lle" else "tail.1"
        expected = {"head.block1.0", "head.block1.2", "body.block1", "att.1", "att1.0", tail_name}
        if set(merged) != expected:
            raise RuntimeError(f"re-parameterisation covered {sorted(merged)}, expected {sorted(expected)}")
        net_slim.load_state_dict(target)
        return net_slim


class MobileIELLESlim(nn.Module):
    """Inference-time MobileIE: plain convs only, nothing re-parameterisable left."""

    def __init__(self, channels: int = 12, task: str = "lle"):
        super().__init__()
        m = modules("plain")
        self.channels = channels
        self.task = task
        in_ch = 3 if task == "lle" else 4
        self.head = m.FSTS(
            nn.Sequential(
                nn.Conv2d(in_ch, channels, 5, 1, 2),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1),
            ),
            channels,
        )
        self.body = m.FSTS(nn.Conv2d(channels, channels, 3, 1, 1), channels)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.att1 = nn.Sequential(nn.Conv2d(1, channels, 1, 1), nn.Sigmoid())
        if task == "isp":
            self.tail = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(3, 3, 3, 1, 1))
        else:
            self.tail = nn.Conv2d(channels, 3, 3, 1, 1)

    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)
        x3 = self.att1(max_out)
        x4 = torch.mul(x3, x2) * x1
        return self.tail(x4)


def build(cfg: dict, variant: str | None = None):
    model_cfg = cfg["model"]
    task = model_cfg.get("task", "lle")
    variant = variant or model_cfg["type"]
    if variant == "slim":
        return MobileIELLESlim(channels=model_cfg["channels"], task=task)
    return MobileIELLE(
        channels=model_cfg["channels"],
        rep_scale=model_cfg["rep_scale"],
        variant=variant,
        iwo_learn_init=model_cfg.get("iwo_learn_init", "zero"),
        task=task,
    )


def load_official_slim_checkpoint(net: nn.Module, path: str):
    """Weights are plain tensors, so weights_only=True keeps loading inert."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v
             for k, v in state.items()}
    net.load_state_dict(state)
    return net


def read_state(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def graph_type_of(state: dict) -> str:
    """Which graph a checkpoint belongs to, read off its key names."""
    if any(k.endswith("conv_out.weight") for k in state):
        deltas = [k for k in state if k.endswith("weight1") and state[k].dim() == 4]
        return "iwo" if deltas else "plain"
    return "slim"


def build_from_checkpoint(cfg: dict, weights: str | None = None, graph: str | None = None):
    state = read_state(weights or cfg["model"]["pretrained"])
    detected = graph_type_of(state)
    graph = graph or detected
    if graph != detected:
        raise ValueError(f"checkpoint looks like a '{detected}' graph but '{graph}' was requested")
    net = build(cfg, variant=graph)
    net.load_state_dict(state)
    return net.eval(), detected


def count_parameters(net: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad or not trainable_only)
