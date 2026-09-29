"""Fetch LOLv1 (our485 + eval15) from a Hugging Face mirror into data/LOLv1.

huggingface.co is unreachable from this machine, so HF_ENDPOINT is forced to the
hf-mirror.com reverse proxy.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

REPO_ID = "Benxelua/LOL_v1"
TARGET = Path(__file__).resolve().parents[1] / "data" / "LOLv1"
PATTERNS = ["our485/low/*.png", "our485/high/*.png", "eval15/low/*.png", "eval15/high/*.png"]


def main():
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(TARGET),
        allow_patterns=PATTERNS,
        max_workers=8,
        local_dir_use_symlinks=False,
    )
    print("downloaded to", root)

    for sub in ("our485/low", "our485/high", "eval15/low", "eval15/high"):
        d = TARGET / sub
        n = len([p for p in d.glob("*.png")])
        print(f"{sub}: {n} png")

    expect = {"our485": 485, "eval15": 15}
    ok = True
    for split, count in expect.items():
        for side in ("low", "high"):
            n = len(list((TARGET / split / side).glob("*.png")))
            if n != count:
                print(f"MISMATCH {split}/{side}: got {n}, expected {count}")
                ok = False
    print("COMPLETE" if ok else "INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
