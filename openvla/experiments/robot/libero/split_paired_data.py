"""
split_paired_data.py

Split an existing paired dataset folder into train / val folders by index.

Input structure:
  paired_data/
    view_A/frame_00001.png
    view_C/frame_00001.png

Output:
  train_data/
    view_A/...
    view_C/...
  val_data/
    view_A/...
    view_C/...

Usage:
  python experiments/robot/libero/split_paired_data.py \
    --src ./paired_data \
    --dst_train ./train_data \
    --dst_val ./val_data \
    --val_ratio 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, required=True)
    p.add_argument("--dst_train", type=str, required=True)
    p.add_argument("--dst_val", type=str, required=True)
    p.add_argument("--val_ratio", type=float, default=0.1)
    return p.parse_args()


def ensure_dirs(base: Path):
    (base / "view_A").mkdir(parents=True, exist_ok=True)
    (base / "view_C").mkdir(parents=True, exist_ok=True)


def main():
    a = parse_args()
    src = Path(a.src)
    dst_train = Path(a.dst_train)
    dst_val = Path(a.dst_val)

    ensure_dirs(dst_train)
    ensure_dirs(dst_val)

    files_a = sorted((src / "view_A").glob("*.png"))
    files_c = sorted((src / "view_C").glob("*.png"))
    assert [f.name for f in files_a] == [f.name for f in files_c], "A/C pairs must match"

    n = len(files_a)
    n_val = max(1, int(round(n * a.val_ratio)))
    n_train = n - n_val

    # First N_train for train, rest for val
    train_names = [f.name for f in files_a[:n_train]]
    val_names = [f.name for f in files_a[n_train:]]

    def copy_pairs(names, dst_base: Path):
        for name in names:
            shutil.copy2(src / "view_A" / name, dst_base / "view_A" / name)
            shutil.copy2(src / "view_C" / name, dst_base / "view_C" / name)

    copy_pairs(train_names, dst_train)
    copy_pairs(val_names, dst_val)

    print({
        "total": n,
        "train": len(train_names),
        "val": len(val_names),
        "dst_train": str(dst_train),
        "dst_val": str(dst_val),
    })


if __name__ == "__main__":
    main()


