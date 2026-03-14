#!/usr/bin/env python3
"""Generate monocular depth maps and surface normals using Depth Anything V2.

Processes all images in a directory and saves:
  - {stem}.npy — relative depth map (float32, H×W)
  - {stem}_normal.npy — surface normals from depth gradients (float32, H×W×3)

Depth maps are used as supervision signal during Gaussian Splatting training
(per-frame aligned L1 loss). Normal maps provide optional normal regularization.

DAv2 outputs relative (affine-invariant) depth — not metric. The training loss
performs per-frame least-squares alignment (scale + shift), so metric depth is
not needed.

Usage:
    python generate_depths.py --images /data/images --output /data/depths
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image


def load_model(encoder="vits"):
    """Load Depth Anything V2 model.

    Uses the small encoder (24M params) by default — sufficient for depth
    priors since we only need relative ordering, not metric accuracy.
    """
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    cfg = model_configs[encoder]
    model = DepthAnythingV2(**cfg)

    # Load pretrained weights from HuggingFace hub
    ckpt_name = f"depth_anything_v2_{encoder}.pth"
    ckpt_path = os.path.join("/workspace/checkpoints", ckpt_name)

    if not os.path.exists(ckpt_path):
        from huggingface_hub import hf_hub_download
        repo_ids = {
            "vits": "depth-anything/Depth-Anything-V2-Small",
            "vitb": "depth-anything/Depth-Anything-V2-Base",
            "vitl": "depth-anything/Depth-Anything-V2-Large",
        }
        ckpt_path = hf_hub_download(
            repo_id=repo_ids[encoder],
            filename=ckpt_name,
            cache_dir="/workspace/checkpoints",
        )

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def depth_to_normals(depth):
    """Compute surface normals from depth map via cross-product of gradients.

    Uses pixel-space finite differences (no focal length needed since normals
    are normalized). Convention: [-dz/dx, -dz/dy, 1] → normalized.

    Args:
        depth: (H, W) depth map (higher = farther from camera)

    Returns:
        (H, W, 3) unit normal map (float32)
    """
    dz_dx = np.gradient(depth, axis=1)
    dz_dy = np.gradient(depth, axis=0)
    normals = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    return (normals / norm).astype(np.float32)


def generate_depth_maps(images_dir, output_dir, encoder="vits", max_dim=518):
    """Generate depth maps for all images in directory.

    Args:
        images_dir: Directory containing input images
        output_dir: Directory to save .npy depth maps
        encoder: DAv2 encoder size (vits/vitb/vitl)
        max_dim: Max image dimension for inference (518 = DAv2 default)
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not image_files:
        print("[DEPTH] No images found")
        return 0

    print(f"[DEPTH] Loading Depth Anything V2 ({encoder})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(encoder).to(device)

    print(f"[DEPTH] Processing {len(image_files)} images...")

    for i, fname in enumerate(image_files):
        img_path = os.path.join(images_dir, fname)
        raw_image = Image.open(img_path).convert("RGB")
        w, h = raw_image.size

        # DAv2's infer_image handles resizing internally
        raw_np = np.array(raw_image)
        with torch.no_grad():
            depth = model.infer_image(raw_np, input_size=max_dim)

        # DAv2 outputs disparity-like values (higher = closer to camera).
        # Flip to depth convention (higher = farther) to match nerfstudio's
        # rendered depth (distance from camera). Without this flip, depth
        # gradients push Gaussians AWAY from surfaces → holes + low opacity.
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth_norm = 1.0 - (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth)

        # Save depth as .npy (float32, same HxW as original image)
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{stem}.npy")
        np.save(out_path, depth_norm.astype(np.float32))

        # Save surface normals from depth gradients
        normals = depth_to_normals(depth_norm)
        normal_path = os.path.join(output_dir, f"{stem}_normal.npy")
        np.save(normal_path, normals)

        if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
            print(f"[DEPTH] {i+1}/{len(image_files)} done")

    print(f"[DEPTH] Saved {len(image_files)} depth + normal maps to {output_dir}")
    return len(image_files)


def main():
    parser = argparse.ArgumentParser(description="Generate depth maps with DAv2")
    parser.add_argument("--images", required=True, help="Input images directory")
    parser.add_argument("--output", required=True, help="Output depths directory")
    parser.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl"],
                        help="DAv2 encoder size (default: vitl)")
    parser.add_argument("--max-dim", type=int, default=518,
                        help="Max inference resolution (default: 518)")
    args = parser.parse_args()

    generate_depth_maps(args.images, args.output, args.encoder, args.max_dim)


if __name__ == "__main__":
    main()
