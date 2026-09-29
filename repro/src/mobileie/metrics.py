"""Evaluation metrics for image enhancement.

The released ``main.py`` only computes a per-channel PSNR, while the paper's
tables report PSNR / SSIM / LPIPS, so the missing metrics are implemented here.
Two PSNR conventions are kept side by side because they differ by a few
hundredths of a dB and only the paper's own convention can be compared with
Table 1.
"""
import numpy as np
import torch


def psnr_official(out: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """The released evaluation formula: mean over channels of 10*log10(1/MSE_c)."""
    mse = ((out - gt) ** 2).mean((2, 3))
    return (1 / mse).log10().mean() * 10


def psnr_rgb(out: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Standard RGB PSNR: 10*log10(1/MSE) over all pixels and channels."""
    mse = ((out - gt) ** 2).mean()
    return torch.log10(1 / mse) * 10


def mae(out: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (out - gt).abs().mean()


def ssim_rgb(out: np.ndarray, gt: np.ndarray) -> float:
    """SSIM with the 11-tap Gaussian window used by the MATLAB reference."""
    from skimage.metrics import structural_similarity
    return float(structural_similarity(
        out, gt, channel_axis=2, data_range=1.0,
        gaussian_weights=True, use_sample_covariance=False,
    ))


class LPIPSDistance:
    """Lazy LPIPS (VGG) holder; the paper's IE tables use the VGG backbone."""

    def __init__(self, device: str = "cuda", net: str = "vgg"):
        self.device = torch.device(device)
        self.net = net
        self._model = None

    def __call__(self, out: torch.Tensor, gt: torch.Tensor) -> float:
        if self._model is None:
            import lpips
            self._model = lpips.LPIPS(net=self.net).to(self.device).eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
        with torch.no_grad():
            # lpips expects [-1, 1]
            return float(self._model(out * 2 - 1, gt * 2 - 1).mean())


class Evaluator:
    """Accumulates per-image metrics and reports their mean.

    ``clip`` decides whether the raw network output is used (what the released
    ``main.py`` does) or the [0, 1]-clipped result (what the reported numbers
    normally mean).
    """

    def __init__(self, device: str = "cuda", lpips_net: str = "vgg", use_lpips: bool = True,
                 clip: bool = True):
        self.device = torch.device(device)
        self.use_lpips = use_lpips
        self.clip = clip
        self.lpips = LPIPSDistance(device, lpips_net) if use_lpips else None
        self.records = []

    @torch.no_grad()
    def update(self, out: torch.Tensor, gt: torch.Tensor, name: str = ""):
        out = out.to(self.device).float()
        gt = gt.to(self.device).float()
        pred = out.clamp(0, 1) if self.clip else out
        record = {
            "name": name,
            "psnr_official": float(psnr_official(pred, gt)),
            "psnr": float(psnr_rgb(pred, gt)),
            "psnr_unclipped": float(psnr_official(out, gt)),
            "mae": float(mae(pred, gt)),
            "ssim": ssim_rgb(pred[0].permute(1, 2, 0).cpu().numpy(), gt[0].permute(1, 2, 0).cpu().numpy()),
        }
        if self.lpips is not None:
            record["lpips"] = self.lpips(pred.clamp(0, 1), gt)
        self.records.append(record)
        return record

    def summarize(self) -> dict:
        if not self.records:
            return {}
        keys = [k for k in ("psnr_official", "psnr", "psnr_unclipped", "ssim", "lpips", "mae")
                if k in self.records[0]]
        summary = {k: float(np.mean([r[k] for r in self.records])) for k in keys}
        summary["n_images"] = len(self.records)
        summary["per_image"] = self.records
        return summary
