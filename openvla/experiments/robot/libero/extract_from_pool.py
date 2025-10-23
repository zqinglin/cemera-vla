"""
extract_from_pool.py

From a trajectory pool saved by run_libero_eval.py (with --save_pool), uniformly
sample paired frames from two cameras (A and C) and export a flat dataset:

Input pool layout (example):
  pool_dir/
    task_00/episode_000/cam_agentview/frame_00010.png
    task_00/episode_000/cam_frontview/frame_00010.png
    ...

Output flat dataset:
  out_dir/
    view_A/task_00__ep_000__frame_00010.png
    view_C/task_00__ep_000__frame_00010.png

Usage:
  python experiments/robot/libero/extract_from_pool.py \
    --pool_dir ./trajectory_pool_A_front \
    --out_dir ./paired_final_A_front \
    --camera_a_name agentview \
    --camera_c_name frontview \
    --frames_per_episode 120 \
    --max_episodes_per_task 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import imageio


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--camera_a_name", type=str, default="agentview")
    p.add_argument("--camera_c_name", type=str, default="frontview")
    p.add_argument("--frames_per_episode", type=int, default=120)
    p.add_argument("--max_episodes_per_task", type=int, default=999999)
    return p.parse_args()


def ensure_flat_dirs(base: Path) -> None:
    (base / "view_A").mkdir(parents=True, exist_ok=True)
    (base / "view_C").mkdir(parents=True, exist_ok=True)


def load_png(path: Path):
    # imageio.v3 returns numpy; keep simple
    return imageio.imread(path)


def main():
    a = parse_args()
    pool = Path(a.pool_dir)
    out_dir = Path(a.out_dir)
    ensure_flat_dirs(out_dir)

    tasks = sorted([d for d in pool.iterdir() if d.is_dir() and d.name.startswith("task_")])
    total_saved = 0

    for task_dir in tasks:
        episodes = sorted([d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
        if not episodes:
            continue
        episodes = episodes[: a.max_episodes_per_task]

        for ep_dir in episodes:
            camA_dir = ep_dir / f"cam_{a.camera_a_name}"
            camC_dir = ep_dir / f"cam_{a.camera_c_name}"
            if not (camA_dir.is_dir() and camC_dir.is_dir()):
                continue

            filesA = sorted([p.name for p in camA_dir.glob("frame_*.png")])
            filesC = sorted([p.name for p in camC_dir.glob("frame_*.png")])
            if not filesA or not filesC:
                continue

            # Align by intersection of frame names
            inter = sorted(list(set(filesA).intersection(filesC)))
            if len(inter) == 0:
                continue

            k = min(a.frames_per_episode, len(inter))
            idxs = np.linspace(0, len(inter) - 1, num=k, dtype=int)
            for ii in idxs:
                name = inter[ii]
                imgA = load_png(camA_dir / name)
                imgC = load_png(camC_dir / name)
                prefix = f"{task_dir.name}__{ep_dir.name}__"
                out_name = prefix + name
                imageio.imwrite(out_dir / "view_A" / out_name, imgA)
                imageio.imwrite(out_dir / "view_C" / out_name, imgC)
                total_saved += 1

        print(f"[{task_dir.name}] episodes processed={len(episodes)} saved_so_far={total_saved}")

    print({"saved_pairs": total_saved, "out_dir": str(out_dir)})


if __name__ == "__main__":
    main()


