"""LOLv1 dataloading.

Faithful to the released ``data/lledata.py``: full-resolution pairs, no crop, no
augmentation, pixels normalised to [0, 1].  Two departures, both neutral to the
numbers: file order is sorted rather than ``os.listdir`` order (the original is
arbitrary and not reproducible), and tensors stay on CPU so the DataLoader can
use workers instead of moving tensors inside ``__getitem__``.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder: Path) -> dict[str, Path]:
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"no images found in {folder}")
    return {p.stem: p for p in files}


class PairedImageDataset(Dataset):
    def __init__(self, inp_dir: str, gt_dir: str | None = None, cache: bool = True,
                 crop: int | None = None, resize_wh=None):
        self.inp_dir = Path(inp_dir)
        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.crop = crop
        self.resize_wh = tuple(resize_wh) if resize_wh else None
        inp_files = list_images(self.inp_dir)
        self.names = list(inp_files)

        if self.gt_dir:
            gt_files = list_images(self.gt_dir)
            missing = [n for n in self.names if n not in gt_files]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} inputs have no ground truth in {self.gt_dir}, e.g. {missing[:3]}"
                )
        else:
            gt_files = {}

        self.inp_files = [inp_files[n] for n in self.names]
        self.gt_files = [gt_files[n] for n in self.names] if gt_files else None
        self._inp_cache = [self._read(p) for p in self.inp_files] if cache else None
        self._gt_cache = [self._read(p) for p in self.gt_files] if (cache and gt_files) else None

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"))

    @classmethod
    def _to_tensor(cls, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0)

    def _array(self, cached, path: Path) -> np.ndarray:
        return cached if cached is not None else self._read(path)

    def _fit(self, arr: np.ndarray) -> np.ndarray:
        if self.resize_wh and (arr.shape[1], arr.shape[0]) != self.resize_wh:
            arr = np.asarray(Image.fromarray(arr).resize(self.resize_wh, Image.BICUBIC))
        return arr

    def __getitem__(self, index):
        inp_arr = self._fit(self._array(
            None if self._inp_cache is None else self._inp_cache[index], self.inp_files[index]))
        name = self.names[index]
        if self.gt_files is None:
            return self._to_tensor(inp_arr), name
        gt_arr = self._fit(self._array(
            None if self._gt_cache is None else self._gt_cache[index], self.gt_files[index]))
        if self.crop:
            h, w = inp_arr.shape[:2]
            if h > self.crop and w > self.crop:
                y = int(torch.randint(0, h - self.crop + 1, (1,)))
                x = int(torch.randint(0, w - self.crop + 1, (1,)))
                inp_arr = inp_arr[y:y + self.crop, x:x + self.crop]
                gt_arr = gt_arr[y:y + self.crop, x:x + self.crop]
        return self._to_tensor(inp_arr), self._to_tensor(gt_arr), name

    def __len__(self):
        return len(self.names)


class BayerPairedDataset(Dataset):
    """ISP contract, identical to the released ``data/ispdata.py``.

    The raw half is a single-channel 12-bit Bayer mosaic (here a 16-bit PNG despite
    some folders calling it .jpg), normalised by 4095 and reshaped RGGB into
    ``4 x H/2 x W/2``; the sRGB half is an ordinary 3 x H x W / 255 target.
    """

    def __init__(self, raw_dir: str, rgb_dir: str | None = None, cache: bool = False, resize_wh=None):
        self.raw_dir = Path(raw_dir)
        self.rgb_dir = Path(rgb_dir) if rgb_dir else None
        self.resize_wh = tuple(resize_wh) if resize_wh else None
        raw_files = list_images(self.raw_dir)
        self.names = list(raw_files)
        if self.rgb_dir:
            rgb_files = list_images(self.rgb_dir)
            missing = [n for n in self.names if n not in rgb_files]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} raw frames have no sRGB in {self.rgb_dir}, e.g. {missing[:3]}")
        else:
            rgb_files = {}
        self.raw_files = [raw_files[n] for n in self.names]
        self.rgb_files = [rgb_files[n] for n in self.names] if rgb_files else None
        self._raw_cache = [np.asarray(Image.open(p)) for p in self.raw_files] if cache else None
        self._rgb_cache = ([np.asarray(Image.open(p).convert("RGB")) for p in self.rgb_files]
                           if (cache and rgb_files) else None)

    @staticmethod
    def bayer2rggb(mosaic: np.ndarray) -> np.ndarray:
        h, w = mosaic.shape
        return (mosaic.reshape(h // 2, 2, w // 2, 2)
                .transpose([1, 3, 0, 2]).reshape([-1, h // 2, w // 2]))

    def _raw_tensor(self, cached, path: Path) -> torch.Tensor:
        arr = cached if cached is not None else np.asarray(Image.open(path))
        if arr.ndim == 3:                        # some mirrors store the mosaic as 3-channel
            arr = arr[..., 0]
        if self.resize_wh and (arr.shape[1], arr.shape[0]) != self.resize_wh:
            arr = np.asarray(Image.fromarray(arr).resize(self.resize_wh, Image.NEAREST))
        planes = self.bayer2rggb(arr)
        return torch.from_numpy(np.ascontiguousarray(planes)).float().div_(4095.0)

    def _rgb_tensor(self, cached, path: Path) -> torch.Tensor:
        arr = cached if cached is not None else np.asarray(Image.open(path).convert("RGB"))
        if self.resize_wh and (arr.shape[1], arr.shape[0]) != self.resize_wh:
            arr = np.asarray(Image.fromarray(arr).resize(self.resize_wh, Image.BICUBIC))
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0)

    def __getitem__(self, index):
        raw = self._raw_tensor(None if self._raw_cache is None else self._raw_cache[index],
                               self.raw_files[index])
        name = self.names[index]
        if self.rgb_files is None:
            return raw, name
        rgb = self._rgb_tensor(None if self._rgb_cache is None else self._rgb_cache[index],
                               self.rgb_files[index])
        return raw, rgb, name

    def __len__(self):
        return len(self.names)


def build_dataset(kind: str, inp: str, gt: str | None, cache: bool, crop=None, resize_wh=None):
    if kind == "raw":
        return BayerPairedDataset(inp, gt, cache=cache, resize_wh=resize_wh)
    return PairedImageDataset(inp, gt, cache=cache, crop=crop, resize_wh=resize_wh)


def make_loaders(cfg: dict, train_batch_size: int | None = None):
    """Returns (train, valid) for split 'train', the test loader for 'test', demo loader otherwise."""
    data_cfg = cfg["data"]
    cache = data_cfg.get("cache_in_memory", True)
    workers = data_cfg.get("num_workers", 0)
    split = cfg.get("split", "train")
    kind = data_cfg.get("kind", "rgb")
    resize_wh = data_cfg.get("resize_wh")

    def ds(inp, gt, crop=None):
        cls = BayerPairedDataset if kind == "raw" else PairedImageDataset
        if kind == "raw":
            return cls(inp, gt, cache, resize_wh=resize_wh)
        return cls(inp, gt, cache, crop=crop, resize_wh=resize_wh)

    def loader(dataset, batch_size, shuffle, drop_last):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                          drop_last=drop_last, pin_memory=False)

    if split == "train":
        # crops apply to training only; validation and testing stay at full resolution
        crop = data_cfg.get("train_crop")
        train = ds(data_cfg["train_inp"], data_cfg["train_gt"], crop=crop or None)
        valid = ds(data_cfg["valid_inp"], data_cfg["valid_gt"])
        return loader(train, train_batch_size or cfg["optim"]["batch_size"], True, True), \
            loader(valid, 1, False, False)
    if split == "test":
        return loader(ds(data_cfg["test_inp"], data_cfg["test_gt"]), 1, False, False)
    if split == "demo":
        return loader(ds(data_cfg["demo_inp"], None), 1, False, False)
    raise ValueError(f"unknown split {split!r}")
