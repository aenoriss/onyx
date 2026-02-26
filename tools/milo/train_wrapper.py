#!/usr/bin/env python3
"""
MILo training wrapper for Onyx Pipeline.
Produces Gaussian splats (and optionally meshes) from COLMAP datasets.

Stages:
    1. Train        — MILo 3DGS training (default)
    2. Mesh Extract — SDF mesh extraction (if --extract-mesh)

Filtering is handled by onyx-segformer as a separate pipeline step.

Usage:
    # Train only (Gaussian splat output)
    python train_wrapper.py --scene /path/to/scene

    # Train + mesh extraction
    python train_wrapper.py --scene /path/to/scene --extract-mesh

    # Mesh extraction only (from existing checkpoint)
    python train_wrapper.py --scene /path/to/scene --skip-training --extract-mesh
"""

import argparse
import os
import sys
import time
from pathlib import Path

from pipeline_progress import progress, run_with_progress

MILO_DIR = "/workspace/MILo/milo"
TOTAL_ITERATIONS = 18000


def main():
    parser = argparse.ArgumentParser(description="MILo training wrapper")
    parser.add_argument("--scene", "-s", required=True,
                        help="Path to scene directory (containing colmap/)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: scene/output/milo)")
    parser.add_argument("--metric", "-m", default="indoor",
                        choices=["indoor", "outdoor"],
                        help="Scene type: indoor or outdoor (default: indoor)")
    parser.add_argument("--rasterizer", "-r", default="radegs",
                        choices=["radegs", "gof"],
                        help="Rasterization technique (default: radegs)")
    parser.add_argument("--mesh_config", default="default",
                        choices=["default", "highres", "veryhighres", "lowres", "verylowres"],
                        help="Mesh resolution config (default: default)")
    parser.add_argument("--dense", action="store_true",
                        help="Use dense Gaussians (recommended for highres/veryhighres)")
    # Iterations fixed at 18k by MiLo's configs/fast (not configurable via CLI)
    parser.add_argument("--eval", action="store_true",
                        help="Enable train/test split for evaluation")
    parser.add_argument("--extract-mesh", action="store_true",
                        help="Extract mesh after training (off by default)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training (use existing checkpoint)")
    args = parser.parse_args()

    start_time = time.time()
    scene_path = Path(args.scene).resolve()
    total_steps = 2 if args.extract_mesh else 1

    # Find COLMAP data - check both scene/colmap and scene directly
    colmap_path = scene_path / "colmap"
    if not colmap_path.exists():
        colmap_path = scene_path

    sparse_path = colmap_path / "sparse" / "0"
    if not sparse_path.exists():
        print(f"Error: COLMAP sparse reconstruction not found at {sparse_path}")
        sys.exit(1)

    # Set output path
    output_path = Path(args.output) if args.output else scene_path / "output" / "milo"
    output_path.mkdir(parents=True, exist_ok=True)

    # Count images
    images_path = colmap_path / "images"
    if not images_path.exists():
        images_path = scene_path / "images"

    num_images = 0
    if images_path.exists():
        num_images = len([f for f in images_path.iterdir()
                         if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])

    print("=" * 60)
    print("MILo: Mesh-In-the-Loop Gaussian Splatting")
    print("=" * 60)
    print(f"Scene:       {scene_path}")
    print(f"COLMAP:      {colmap_path}")
    print(f"Output:      {output_path}")
    print(f"Images:      {num_images}")
    print(f"Scene type:  {args.metric}")
    print(f"Rasterizer:  {args.rasterizer}")
    print(f"Mesh config: {args.mesh_config}")
    print(f"Dense:       {args.dense}")
    print(f"Iterations:  18000 (fixed by configs/fast)")
    print(f"Mesh:        {'Yes' if args.extract_mesh else 'No'}")
    print("=" * 60)

    # ── Stage 1: Training ─────────────────────────────────────
    if not args.skip_training:
        cmd = [
            "python", "-u", "train.py",
            "-s", str(colmap_path),
            "-m", str(output_path),
            "--imp_metric", args.metric,
            "--rasterizer", args.rasterizer,
            "--mesh_config", args.mesh_config,
            "--data_device", "cpu",
            # NOTE: --iterations not passed; MiLo's configs/fast hardcodes 18k
            # and its read_config() always overrides CLI args.
        ]

        if args.dense:
            cmd.append("--dense_gaussians")
        if args.eval:
            cmd.append("--eval")

        # MiLo outputs iteration progress like: "Step: 9900/18000"
        train_patterns = {
            r"Step:\s*(\d+)/(\d+)": lambda m: (
                round(int(m.group(1)) / int(m.group(2)) * 100),
                f"Step {m.group(1)}/{m.group(2)}",
            ),
        }

        run_with_progress(cmd, "training",
                          step=1, total_steps=total_steps,
                          patterns=train_patterns, cwd=MILO_DIR)

    # ── Stage 2: Mesh Extraction (if --extract-mesh) ──────────
    if args.extract_mesh:
        extract_cmd = [
            "python", "-u", "mesh_extract_sdf.py",
            "-s", str(colmap_path),
            "-m", str(output_path),
            "--rasterizer", args.rasterizer,
            "--config", args.mesh_config,
            "--data_device", "cpu",
        ]

        if args.eval:
            extract_cmd.append("--eval")

        # mesh_extract_sdf.py may output tqdm bars or iteration counts
        mesh_extract_patterns = {
            r"(\d+)%\|": lambda m: (
                int(m.group(1)),
                f"Mesh extraction {m.group(1)}%",
            ),
            r"(?i)iter(?:ation)?\s+(\d+)/(\d+)": lambda m: (
                round(int(m.group(1)) / int(m.group(2)) * 100),
                f"Iter {m.group(1)}/{m.group(2)}",
            ),
        }
        run_with_progress(extract_cmd, "mesh_extraction",
                          step=2, total_steps=total_steps,
                          patterns=mesh_extract_patterns, cwd=MILO_DIR)

        mesh_path = output_path / "mesh_learnable_sdf.ply"
        if mesh_path.exists():
            size_mb = mesh_path.stat().st_size / (1024 * 1024)
            print(f"  Mesh: {mesh_path} ({size_mb:.1f} MB)")

    # ── Summary ───────────────────────────────────────────────
    progress("done", "completed", 100, step=total_steps, total_steps=total_steps)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"[COMPLETE] Pipeline finished in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    print(f"\nOutputs in {output_path}:")
    for f in output_path.rglob("*.ply"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.relative_to(output_path)}: {size_mb:.1f} MB")
    for f in output_path.glob("*.pth"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
