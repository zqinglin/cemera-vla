"""
investigate_features.py

Purpose: Probe the OpenVLA vision encoder output tensor shape without changing model weights.

What it does:
- Loads a specified OpenVLA checkpoint
- Builds a dummy image tensor of the expected size
- Runs one forward pass through the vision backbone only
- Prints (batch_size, num_patches, token_dim)

Usage:
  python experiments/robot/libero/investigate_features.py \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial

Optional args:
  --height 224 --width 224    # override derived image size
  --device cuda:0             # default auto-detect
"""

from dataclasses import dataclass
from typing import Optional

import draccus
import torch
from transformers import AutoModelForVision2Seq


@dataclass
class ProbeConfig:
    pretrained_checkpoint: str = "openvla/openvla-7b-finetuned-libero-spatial"
    height: Optional[int] = None
    width: Optional[int] = None
    device: Optional[str] = None  # e.g., "cuda:0" or "cpu"


@draccus.wrap()
def main(cfg: ProbeConfig) -> None:
    # Resolve device & dtype
    device = (
        torch.device(cfg.device)
        if cfg.device is not None
        else (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Load VLA
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()

    # Derive image resolution from config if not provided
    # Note: config.image_sizes is a list; use the first as primary vision encoder input size.
    H = cfg.height if cfg.height is not None else int(getattr(vla.config, "image_sizes", [224])[0])
    W = cfg.width if cfg.width is not None else int(getattr(vla.config, "image_sizes", [224])[0])

    # Determine channels: 3 for single vision backbone; 6 if fused (stacked channels)
    use_fused = bool(getattr(vla.config, "use_fused_vision_backbone", False))
    C = 6 if use_fused else 3

    # Build dummy image tensor
    pixel_values = torch.randn(1, C, H, W, device=device, dtype=dtype)

    # Forward through vision backbone only
    with torch.no_grad():
        feat = vla.vision_backbone(pixel_values)

    # Report shape: (batch_size, num_patches, token_dim)
    bsz, num_patches, token_dim = feat.shape
    print("Vision encoder output shape:", (bsz, num_patches, token_dim))
    print("num_patches (N) =", num_patches)
    print("token_dim   (D) =", token_dim)


if __name__ == "__main__":
    main()


