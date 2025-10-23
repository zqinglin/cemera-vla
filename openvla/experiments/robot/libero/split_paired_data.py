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


def ensure_flat_dirs(base: Path) -> None:
    (base / "view_A").mkdir(parents=True, exist_ok=True)
    (base / "view_C").mkdir(parents=True, exist_ok=True)


def list_pairs(src: Path):
    """Return list of (task_rel, filename) pairs and a resolver to src paths.

    Supports two layouts:
      flat:  src/view_A/*.png, src/view_C/*.png
      nested: src/task_xx/view_A/*.png, src/task_xx/view_C/*.png
    """
    flat_a = src / "view_A"
    flat_c = src / "view_C"
    items: list[tuple[str, str]] = []

    if flat_a.is_dir() and flat_c.is_dir():
        files_a = sorted(flat_a.glob("*.png"))
        files_c = sorted(flat_c.glob("*.png"))
        assert [f.name for f in files_a] == [f.name for f in files_c], "A/C pairs must match"
        items = [("", f.name) for f in files_a]

        def resolver(task_rel: str, name: str):
            return flat_a / name, flat_c / name

        return items, resolver

    # nested
    items = []
    task_dirs = [d for d in src.iterdir() if d.is_dir()]
    def resolver(task_rel: str, name: str):
        return src / task_rel / "view_A" / name, src / task_rel / "view_C" / name

    for td in task_dirs:
        a_dir = td / "view_A"
        c_dir = td / "view_C"
        if not (a_dir.is_dir() and c_dir.is_dir()):
            continue
        files_a = sorted(a_dir.glob("*.png"))
        files_c = sorted(c_dir.glob("*.png"))
        assert [f.name for f in files_a] == [f.name for f in files_c], f"A/C mismatch in {td.name}"
        items.extend([(td.name, f.name) for f in files_a])

    return items, resolver


def main():
    a = parse_args()
    src = Path(a.src)
    dst_train = Path(a.dst_train)
    dst_val = Path(a.dst_val)

    # Always flatten outputs
    ensure_flat_dirs(dst_train)
    ensure_flat_dirs(dst_val)

    items, resolve = list_pairs(src)
    n = len(items)
    if n == 0:
        print({"total": 0, "train": 0, "val": 0, "dst_train": str(dst_train), "dst_val": str(dst_val)})
        return

    n_val = max(1, int(round(n * a.val_ratio)))
    n_train = n - n_val
    train_items = items[:n_train]
    val_items = items[n_train:]

    def copy_items(pairs, dst_base: Path):
        for task_rel, name in pairs:
            # Flatten naming: prefix task_rel to avoid collisions
            prefix = (task_rel + "__") if task_rel else ""
            out_name = prefix + name
            src_a, src_c = resolve(task_rel, name)
            dst_a = dst_base / "view_A" / out_name
            dst_c = dst_base / "view_C" / out_name
            shutil.copy2(src_a, dst_a)
            shutil.copy2(src_c, dst_c)

    copy_items(train_items, dst_train)
    copy_items(val_items, dst_val)

    print({
        "total": n,
        "train": len(train_items),
        "val": len(val_items),
        "dst_train": str(dst_train),
        "dst_val": str(dst_val),
    })


if __name__ == "__main__":
    main()


