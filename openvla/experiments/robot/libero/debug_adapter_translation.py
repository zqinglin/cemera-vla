"""
debug_adapter_translation.py

Compute three diagnostics over a held-out sample of paired images (A: agentview, C: other view):
  - loss_before = MSE(F_source,  F_target)
  - loss_after  = MSE(F_adapted, F_target)
  - loss_change = MSE(F_adapted, F_source)

Where features are extracted from the frozen VLA vision backbone, and F_adapted is the
output of the trained ShallowWideTransformerAdapter applied to F_source.

Usage:
  python experiments/robot/libero/debug_adapter_translation.py \
    --pretrained_checkpoint hf_models/openvla-openvla-7b-finetuned-libero-spatial \
    --data_root ./paired_data \
    --adapter_path adapter_ckpts/adapter_final.pth \
    --num_samples 128
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForVision2Seq

from experiments.robot.libero.adapter import ShallowWideTransformerAdapter, AdapterConfig


@dataclass
class DebugConfig:
    pretrained_checkpoint: str
    data_root: str
    adapter_path: str
    num_samples: int = 128
    img_size: int = 224
    device: str = "auto"


class PairedImageDataset(data.Dataset):
    def __init__(self, root: str, transform: transforms.Compose) -> None:
        super().__init__()
        self.root = Path(root)
        self.dir_a = self.root / "view_A"
        self.dir_c = self.root / "view_C"
        self.fnames = sorted([p.name for p in self.dir_a.glob("*.png")])
        fnames_c = sorted([p.name for p in self.dir_c.glob("*.png")])
        assert self.fnames == fnames_c, "view_A and view_C file names must match"
        self.transform = transform

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        name = self.fnames[idx]
        img_a = Image.open(self.dir_a / name).convert("RGB")
        img_c = Image.open(self.dir_c / name).convert("RGB")
        return self.transform(img_a), self.transform(img_c)


def parse_args() -> DebugConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--adapter_path", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--device", type=str, default="auto")
    a = p.parse_args()
    return DebugConfig(
        pretrained_checkpoint=a.pretrained_checkpoint,
        data_root=a.data_root,
        adapter_path=a.adapter_path,
        num_samples=a.num_samples,
        img_size=a.img_size,
        device=a.device,
    )


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def resolve_image_processor_src(ckpt: str) -> str:
    if os.path.isdir(ckpt):
        base = os.path.basename(os.path.abspath(ckpt)).lower()
        if "openvla-7b-finetuned-libero-spatial" in base:
            return "openvla/openvla-7b-finetuned-libero-spatial"
        if "openvla-7b-finetuned-libero-object" in base:
            return "openvla/openvla-7b-finetuned-libero-object"
        if "openvla-7b-finetuned-libero-goal" in base:
            return "openvla/openvla-7b-finetuned-libero-goal"
        if "openvla-7b-finetuned-libero-10" in base:
            return "openvla/openvla-7b-finetuned-libero-10"
        if "openvla-7b" in base:
            return "openvla/openvla-7b"
    return ckpt


def main(cfg: DebugConfig) -> None:
    device = resolve_device(cfg.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Load VLA (frozen) and image processor
    img_proc_src = resolve_image_processor_src(cfg.pretrained_checkpoint)
    image_processor = AutoImageProcessor.from_pretrained(img_proc_src, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()
    for p in vla.parameters():
        p.requires_grad_(False)

    # Load adapter
    adapter_cfg = AdapterConfig(
        num_patches=256, token_dim=2176, adapter_width=2688, nhead=21, num_layers=2, dropout=0.0
    )
    adapter = ShallowWideTransformerAdapter(adapter_cfg).to(device=device, dtype=dtype).eval()
    state = torch.load(cfg.adapter_path, map_location=device)
    adapter.load_state_dict(state)

    # Dataset / Loader (take a subset of size num_samples)
    tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.bfloat16 if dtype == torch.bfloat16 else torch.float32),
    ])
    ds_all = PairedImageDataset(cfg.data_root, tfm)
    n = min(cfg.num_samples, len(ds_all))
    idxs = list(range(n))
    subset = data.Subset(ds_all, idxs)
    loader = data.DataLoader(subset, batch_size=8, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))

    mse = nn.MSELoss(reduction="mean")
    total_before, total_after, total_change, count = 0.0, 0.0, 0.0, 0

    def extract_feats(img_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            imgs = [transforms.ToPILImage()(img.cpu().to(torch.float32)) for img in img_batch]
            proc = image_processor(images=imgs, return_tensors="pt")
            pixel_values = proc["pixel_values"].to(device, dtype=dtype)
            feat = vla.vision_backbone(pixel_values)
        return feat

    with torch.no_grad():
        for img_a, img_c in loader:
            img_a = img_a.to(device)
            img_c = img_c.to(device)
            f_target = extract_feats(img_a)
            f_source = extract_feats(img_c)
            f_adapted = adapter(f_source)

            lb = mse(f_source,  f_target).item()
            la = mse(f_adapted, f_target).item()
            lc = mse(f_adapted, f_source).item()

            total_before += lb
            total_after  += la
            total_change += lc
            count += 1

    print({
        "loss_before": total_before / count,
        "loss_after":  total_after  / count,
        "loss_change": total_change / count,
        "num_batches": count,
        "num_samples": n,
    })


if __name__ == "__main__":
    main(parse_args())


