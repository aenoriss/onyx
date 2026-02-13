#!/usr/bin/env python3
"""
Gaussian Splatting training wrapper for Onyx Pipeline.
Uses nerfstudio's splatfacto (built on gsplat) with proper data loading.

Creates an ephemeral nerfstudio-compatible workspace in /tmp/ns_workspace/
with symlinks to /data/images and /data/sparse. This avoids mutating the
shared volume layout.

Output: Standard PLY format compatible with Unreal, web viewers, etc.

Usage:
    python train.py --scene /path/to/scene [--iterations 30000] [--resolution 1]
"""

import argparse
import glob
import os
import sys
import subprocess

from pipeline_progress import progress, run_with_progress

NS_WORKSPACE = "/tmp/ns_workspace"


def setup_nerfstudio_workspace(scene_path):
    """Create ephemeral nerfstudio-compatible layout with symlinks.

    nerfstudio expects: {root}/colmap/sparse/0/ and {root}/colmap/images/
    Our shared volume has: {scene}/sparse/0/ and {scene}/images/

    Creates symlinks in /tmp/ns_workspace/ to bridge the gap.
    """
    colmap_dir = os.path.join(NS_WORKSPACE, "colmap")
    sparse_dir = os.path.join(colmap_dir, "sparse", "0")
    images_dir = os.path.join(colmap_dir, "images")

    os.makedirs(sparse_dir, exist_ok=True)

    # Check if scene already has colmap/ layout
    src_colmap = os.path.join(scene_path, "colmap")
    if os.path.exists(src_colmap):
        # Scene already has colmap/ structure — symlink directly
        src_sparse = os.path.join(src_colmap, "sparse", "0")
        src_images = os.path.join(src_colmap, "images")
    else:
        # Standard pipeline layout: sparse/0/ and images/ at root
        src_sparse = os.path.join(scene_path, "sparse", "0")
        src_images = os.path.join(scene_path, "images")

    # Symlink sparse files
    if os.path.exists(src_sparse):
        for f in os.listdir(src_sparse):
            src = os.path.join(src_sparse, f)
            dst = os.path.join(sparse_dir, f)
            if not os.path.exists(dst):
                os.symlink(src, dst)

    # Symlink images directory
    if not os.path.exists(images_dir):
        os.symlink(src_images, images_dir)

    # Also symlink masks if they exist
    masks_src = os.path.join(scene_path, "masks")
    masks_dst = os.path.join(colmap_dir, "masks")
    if os.path.exists(masks_src) and not os.path.exists(masks_dst):
        os.symlink(masks_src, masks_dst)

    return NS_WORKSPACE


def main():
    parser = argparse.ArgumentParser(description="Gaussian Splatting training wrapper")
    parser.add_argument("--scene", "-s", required=True,
                        help="Path to scene directory (containing images/ + sparse/)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: scene/output/splatfacto)")
    parser.add_argument("--iterations", "-i", type=int, default=30000,
                        help="Number of training iterations (default: 30000)")
    parser.add_argument("--resolution", "-r", type=int, default=1,
                        help="Resolution scale factor (1=full, 2=half, 4=quarter)")
    parser.add_argument("--eval", action="store_true",
                        help="Enable eval mode (hold out test images)")
    args = parser.parse_args()

    scene_path = os.path.abspath(args.scene)
    total_steps = 2

    # Set output path
    output_path = args.output or os.path.join(scene_path, "output", "splatfacto")
    os.makedirs(output_path, exist_ok=True)

    # Setup ephemeral nerfstudio workspace
    ws = setup_nerfstudio_workspace(scene_path)

    # Count images
    colmap_images = os.path.join(ws, "colmap", "images")
    num_images = 0
    if os.path.exists(colmap_images):
        num_images = len([f for f in os.listdir(colmap_images)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Check for masks
    masks_path = os.path.join(ws, "colmap", "masks")
    has_masks = os.path.exists(masks_path) and len(os.listdir(masks_path)) > 0
    num_masks = len(os.listdir(masks_path)) if has_masks else 0

    print("=" * 50)
    print("Gaussian Splatting Training (splatfacto)")
    print("=" * 50)
    print(f"Scene: {scene_path}")
    print(f"Workspace: {ws}")
    print(f"Output: {output_path}")
    print(f"Images: {num_images}")
    print(f"Masks: {num_masks if has_masks else 'None'}")
    print(f"Iterations: {args.iterations}")
    print(f"Resolution: 1/{args.resolution}")
    print("=" * 50)

    # ── Stage 1: Training ─────────────────────────────────────
    cmd = [
        "ns-train", "splatfacto",
        "--output-dir", output_path,
        "--max-num-iterations", str(args.iterations),
        "--vis", "tensorboard",
        "--steps-per-eval-batch", "0",
        "--steps-per-eval-image", "0",
        "--steps-per-eval-all-images", "0",
        "--steps-per-save", str(args.iterations),
        "colmap",
        "--data", ws,
        "--images-path", "colmap/images",
        "--downscale-factor", str(args.resolution),
    ]

    if has_masks:
        cmd.extend(["--masks-path", "colmap/masks"])

    # nerfstudio outputs step progress like various patterns
    train_patterns = {
        r"Step.*?(\d+)/(\d+)": lambda m: (
            round(int(m.group(1)) / int(m.group(2)) * 100),
            f"Step {m.group(1)}/{m.group(2)}",
        ),
        r"(\d+)%\|": lambda m: (
            int(m.group(1)),
            f"Training {m.group(1)}%",
        ),
    }

    run_with_progress(cmd, "training",
                      step=1, total_steps=total_steps,
                      patterns=train_patterns)

    # ── Stage 2: PLY Export ───────────────────────────────────
    progress("ply_export", "running", step=2, total_steps=total_steps)

    config_paths = glob.glob(os.path.join(output_path, "*", "splatfacto", "*", "config.yml"))
    if not config_paths:
        config_paths = glob.glob(os.path.join(output_path, "splatfacto", "*", "config.yml"))

    if not config_paths:
        print("Warning: No config found for PLY export")
        progress("ply_export", "failed", error="No config.yml found",
                 step=2, total_steps=total_steps)
        sys.exit(1)

    config_path = sorted(config_paths)[-1]
    print(f"Found config: {config_path}")

    ply_output = os.path.join(output_path, "ply")

    run_with_progress(
        ["ns-export", "gaussian-splat",
         "--load-config", config_path,
         "--output-dir", ply_output],
        stage="ply_export",
        step=2, total_steps=total_steps,
    )

    ply_path = os.path.join(ply_output, "splat.ply")
    if os.path.exists(ply_path):
        size_mb = os.path.getsize(ply_path) / (1024 * 1024)
        print(f"PLY exported: {ply_path} ({size_mb:.1f} MB)")

    progress("done", "completed", 100, step=total_steps, total_steps=total_steps)


if __name__ == "__main__":
    main()
