"""Training objectives, mirroring the released ``loss.py``.

The paper's LVW loss (Eq. 7-10) is what ``loss.py`` calls ``OutlierAwareLoss``.
The released implementation weights the *signed* residual with statistics taken
over the spatial dims, whereas Eq. 7-8 first takes the L1 magnitude and then its
local mean/variance.  Both are provided: ``lvw_official`` is the default because
the shipped checkpoints were trained with it, and ``lvw_paper`` lets the
formulation actually written in the paper be checked.
"""
import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return ((nn.functional.mse_loss(pred, target, reduction="none") + self.eps2) ** 0.5).mean()


class LVWOfficial(nn.Module):
    """``OutlierAwareLoss`` in the released code."""

    def forward(self, out, gt):
        delta = out - gt
        std = delta.std((2, 3), keepdim=True) / (2 ** 0.5)
        mean = delta.mean((2, 3), keepdim=True)
        weight = torch.tanh((delta - mean).abs() / (std + 1e-6)).detach()
        return (delta.abs() * weight).mean()


class LVWPaper(nn.Module):
    """Eq. 7-10 verbatim: weight the L1 magnitude by its own local statistics."""

    def forward(self, out, gt):
        delta = (out - gt).abs()
        mean = delta.mean((2, 3), keepdim=True)
        var = ((delta - mean) ** 2).mean((2, 3), keepdim=True)
        weight = torch.tanh((delta - mean).abs() / (var.sqrt() + 1e-6)).detach()
        return (weight * delta).mean()


class PSNRLoss(nn.Module):
    """Differentiable PSNR objective, as in the released ``loss.py``."""

    def forward(self, pred, target):
        rmse = (((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).sqrt()
        loss = 20 * torch.log10(1 / rmse).mean()
        return (50.0 - loss) / 100.0


class WarmupLoss(nn.Module):
    """Self-supervised pre-fitting stage: reconstruct the input, then the target."""

    def __init__(self, pixel: nn.Module | None = None):
        super().__init__()
        self.cb = CharbonnierLoss(1e-8)
        self.cs = nn.CosineSimilarity()
        self.pixel = pixel

    def forward(self, inp, gt, warm_out1, warm_out2):
        return self.cb(warm_out2, inp) + (self.cb(warm_out1, gt)
                                          + (1 - self.cs(warm_out1.clamp(0, 1), gt)).mean())


class MobileIELoss(nn.Module):
    """Pixel term + colour-consistency term + differentiable-PSNR term."""

    def __init__(self, pixel: str = "lvw_official"):
        super().__init__()
        pixel_terms = {"lvw_official": LVWOfficial, "lvw_paper": LVWPaper,
                       "l1": lambda: nn.L1Loss(), "l2": lambda: nn.MSELoss()}
        if pixel not in pixel_terms:
            raise ValueError(f"unknown pixel loss {pixel!r}, choose from {sorted(pixel_terms)}")
        self.pixel = pixel_terms[pixel]()
        self.cs = nn.CosineSimilarity()
        self.psnr = PSNRLoss()

    def forward(self, out, gt):
        return (self.pixel(out, gt) + (1 - self.cs(out.clamp(0, 1), gt)).mean()) + 2 * self.psnr(out, gt)

    def forward_parts(self, out, gt) -> dict:
        """The three terms separately.

        The composite is ~99% the differentiable-PSNR term, so its absolute value is not
        comparable with the "Loss" axis of paper Figure 10; the pixel term is.
        """
        with torch.no_grad():
            return {"loss_pixel": float(self.pixel(out, gt)),
                    "loss_cos": float((1 - self.cs(out.clamp(0, 1), gt)).mean()),
                    "loss_psnr_term": float(2 * self.psnr(out, gt))}


def build_loss(cfg: dict):
    return MobileIELoss(cfg["loss"].get("pixel", "lvw_official")), WarmupLoss()
