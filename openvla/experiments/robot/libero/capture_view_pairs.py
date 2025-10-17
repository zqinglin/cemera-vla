"""
capture_view_pairs.py

Collects synchronized (view_A, view_C) image pairs from LIBERO simulation by
stepping random actions. Images are saved with matching frame indices to ensure
strict time alignment.

Usage example:
    python experiments/robot/libero/capture_view_pairs.py \
      --camera_a_name agentview \
      --camera_c_name robot0_eye_in_hand \
      --output_dir ./paired_data \
      --num_pairs 5000
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import imageio
import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
)


@dataclass
class CaptureConfig:
    # Cameras and output
    camera_a_name: str = "agentview"
    camera_c_name: str = "robot0_eye_in_hand"
    output_dir: str = "./paired_data"

    # How many synchronized pairs to capture
    num_pairs: int = 5000

    # LIBERO suite/task to sample from
    task_suite_name: str = "libero_spatial"
    resolution: int = 256
    num_steps_wait: int = 10

    # Random rollouts
    max_steps_per_episode: int = 300
    seed: int = 7


def _ensure_dirs(base_dir: Path) -> None:
    (base_dir / "view_A").mkdir(parents=True, exist_ok=True)
    (base_dir / "view_C").mkdir(parents=True, exist_ok=True)


def _format_idx(idx: int) -> str:
    return f"{idx:05d}"


def _sample_random_action(scale_xyz: float = 0.08, scale_rot: float = 0.08) -> np.ndarray:
    """Generate a small random 7-DoF action: [dx,dy,dz, rx,ry,rz, gripper].

    The environment expects values roughly in [-1, 1]. We use small deltas to
    keep rollouts stable; gripper randomly toggles open/close.
    """
    dpos = np.clip(np.random.uniform(-scale_xyz, scale_xyz, size=3), -1.0, 1.0)
    drot = np.clip(np.random.uniform(-scale_rot, scale_rot, size=3), -1.0, 1.0)
    grip = np.random.choice([-1.0, 1.0])
    return np.concatenate([dpos, drot, [grip]], axis=0)


@draccus.wrap()
def main(cfg: CaptureConfig) -> None:
    # Seeding
    np.random.seed(cfg.seed)

    # Prepare output
    out_dir = Path(cfg.output_dir)
    _ensure_dirs(out_dir)

    # Initialize LIBERO task and env
    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task = task_suite.get_task(0)
    initial_states = task_suite.get_task_init_states(0)
    env, _ = get_libero_env(task, model_family="openvla", resolution=cfg.resolution)

    saved = 0
    episode_idx = 0
    init_idx = 0

    while saved < cfg.num_pairs:
        # Reset and set a (cycled) initial state for coverage
        env.reset()
        obs = env.set_init_state(initial_states[init_idx % len(initial_states)])
        init_idx += 1

        t = 0
        while t < cfg.max_steps_per_episode and saved < cfg.num_pairs:
            # First wait steps: do nothing for sim to settle
            if t < cfg.num_steps_wait:
                obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
                t += 1
                continue

            # Capture synchronized images at this timestep
            img_a = get_libero_image(obs, cfg.resolution, camera_name=cfg.camera_a_name)
            img_c = get_libero_image(obs, cfg.resolution, camera_name=cfg.camera_c_name)

            idx_str = _format_idx(saved + 1)
            imageio.imwrite(out_dir / "view_A" / f"frame_{idx_str}.png", img_a)
            imageio.imwrite(out_dir / "view_C" / f"frame_{idx_str}.png", img_c)
            saved += 1

            # Step a random action
            action = _sample_random_action()
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break
            t += 1

        episode_idx += 1

    print(f"Saved {saved} synchronized pairs to: {out_dir}")


if __name__ == "__main__":
    main()


