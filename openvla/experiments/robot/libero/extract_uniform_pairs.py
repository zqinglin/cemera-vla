import os
import sys
from dataclasses import dataclass
from pathlib import Path
import argparse
import numpy as np
import imageio

# Reuse repo tools
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_env, get_libero_image, get_libero_dummy_action, quat2axisangle
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_model, get_image_resize_size, get_action,
    normalize_gripper_action, invert_gripper_action, set_seed_everywhere
)
from libero.libero import benchmark


@dataclass
class Config:
    pretrained_checkpoint: str
    task_suite_name: str = "libero_spatial"
    num_examples: int = 5            # number of tasks / episodes to sample
    frames_per_episode: int = 120    # uniform frames per episode
    out_dir: str = "./paired_uniform"
    center_crop: bool = True
    seed: int = 7


def save_pair(out_root: Path, idx: int, imgA: np.ndarray, imgC: np.ndarray) -> None:
    outA = out_root / "view_A"; outC = out_root / "view_C"
    outA.mkdir(parents=True, exist_ok=True); outC.mkdir(parents=True, exist_ok=True)
    name = f"frame_{idx:05d}.png"
    imageio.imwrite(outA / name, imgA)
    imageio.imwrite(outC / name, imgC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_checkpoint", type=str, required=True)
    ap.add_argument("--task_suite_name", type=str, default="libero_spatial")
    ap.add_argument("--num_examples", type=int, default=5)
    ap.add_argument("--frames_per_episode", type=int, default=120)
    ap.add_argument("--out_dir", type=str, default="./paired_uniform")
    ap.add_argument("--center_crop", type=str, default="True")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = Config(
        pretrained_checkpoint=args.pretrained_checkpoint,
        task_suite_name=args.task_suite_name,
        num_examples=args.num_examples,
        frames_per_episode=args.frames_per_episode,
        out_dir=args.out_dir,
        center_crop=(args.center_crop.lower() == "true"),
        seed=args.seed,
    )

    set_seed_everywhere(cfg.seed)

    # Load model + processor
    model = get_model(type("C", (), {
        "model_family":"openvla",
        "pretrained_checkpoint":cfg.pretrained_checkpoint,
        "load_in_8bit":False, "load_in_4bit":False,
        "center_crop":cfg.center_crop
    }))
    processor = get_processor(type("C", (), {
        "model_family":"openvla",
        "pretrained_checkpoint":cfg.pretrained_checkpoint,
        "center_crop":cfg.center_crop
    }))

    out_root = Path(cfg.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    resize_size = get_image_resize_size(type("C", (), {"task_suite_name": cfg.task_suite_name}))

    saved = 0
    for task_id in range(min(cfg.num_examples, task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        env, task_desc = get_libero_env(task, "openvla", resolution=256)

        env.reset()
        obs = env.set_init_state(init_states[0])

        imgsA, imgsC = [], []
        t = 0
        # match eval max steps
        if cfg.task_suite_name == "libero_spatial":
            max_steps = 220
        elif cfg.task_suite_name == "libero_object":
            max_steps = 280
        elif cfg.task_suite_name == "libero_goal":
            max_steps = 300
        elif cfg.task_suite_name == "libero_10":
            max_steps = 520
        else:
            max_steps = 300

        num_steps_wait = 10
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
                t += 1
                continue

            imgA = get_libero_image(obs, resize_size, camera_name="agentview")
            imgC = get_libero_image(obs, resize_size, camera_name="frontview")
            # imgC = get_libero_image(obs, resize_size, camera_name="robot0_eye_in_hand")
            imgsA.append(imgA); imgsC.append(imgC)

            o = {
                "full_image": imgA,
                "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
            }
            # choose unnorm_key present in model.norm_stats if available
            unnorm_key = cfg.task_suite_name
            if hasattr(model, "norm_stats") and isinstance(model.norm_stats, dict) and model.norm_stats:
                if unnorm_key not in model.norm_stats:
                    unnorm_key = next(iter(model.norm_stats.keys()))

            action = get_action(
                type("C", (), {"model_family":"openvla", "unnorm_key": unnorm_key, "center_crop":cfg.center_crop}),
                model, o, task_desc, processor=processor
            )
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break
            t += 1

        total = len(imgsA)
        if total == 0:
            continue
        k = min(cfg.frames_per_episode, total)
        idxs = np.linspace(0, total-1, num=k, dtype=int)
        ep_root = out_root / f"task_{task_id:02d}"
        for j, ii in enumerate(idxs, start=1):
            save_pair(ep_root, j, imgsA[ii], imgsC[ii])
            saved += 1
        print(f"[task {task_id}] total={total}, sampled={k}, saved_pairs={saved}")

    print(f"Done. Total saved pairs: {saved}, out_dir={out_root}")


if __name__ == "__main__":
    main()


