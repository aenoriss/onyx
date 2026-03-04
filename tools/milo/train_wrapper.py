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


def _process_dense_init(init_src, sparse_path, target_pts=500_000):
    """Subsample a dense PLY cloud and inject it as sparse/0/points3D.bin.

    MILo reads points3D.bin and discards tracks entirely, so we write empty
    tracks. Camera poses (cameras.bin / images.bin) are untouched.
    """
    import struct
    import numpy as np
    import open3d as o3d
    import shutil

    print(f"[dense_init] Loading {init_src} ...")
    pcd = o3d.io.read_point_cloud(str(init_src))
    print(f"[dense_init] Loaded {len(pcd.points):,} points")

    # Stage 1: Statistical outlier removal (removes noise/floaters)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"[dense_init] After SOR: {len(pcd.points):,} points")

    # Stage 2: Uniform random subsample to target
    # Voxel-based downsampling is unreliable for surface point clouds (MVS output
    # lies on 2D surfaces, not in 3D volumes), so the volumetric voxel formula
    # produces wildly wrong voxel sizes. Random subsampling guarantees exactly
    # target_pts regardless of scene type or point distribution.
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)  # float [0,1]; may be empty if PLY has no color

    if len(colors) == 0 or colors.shape[0] != len(pts):
        print("[dense_init] Warning: no color data in PLY — defaulting to white")
        colors = np.ones((len(pts), 3), dtype=np.float32)

    if len(pts) > target_pts:
        idx = np.random.choice(len(pts), target_pts, replace=False)
        pts, colors = pts[idx], colors[idx]

    rgb = (colors * 255).clip(0, 255).astype(np.uint8)
    print(f"[dense_init] Final count: {len(pts):,} points")

    # Backup original sparse cloud (only once — skip if backup already exists)
    bin_path = sparse_path / "points3D.bin"
    backup_path = sparse_path / "points3D.bin.sparse_backup"
    if bin_path.exists() and not backup_path.exists():
        shutil.copy2(str(bin_path), str(backup_path))
        print(f"[dense_init] Backed up sparse cloud → {backup_path.name}")

    # Write COLMAP binary points3D.bin with empty tracks (error=0)
    # Format: num_pts(Q) | [point3D_id(Q) xyz(ddd) rgb(BBB) error(d) track_len(Q)] * N
    # Use '<' (little-endian, standard sizes) to match MILo's read_points3D_binary
    with open(str(bin_path), 'wb') as f:
        f.write(struct.pack('<Q', len(pts)))
        for i, (pt, color) in enumerate(zip(pts, rgb)):
            f.write(struct.pack('<QdddBBBd',
                i + 1,
                float(pt[0]), float(pt[1]), float(pt[2]),
                int(color[0]), int(color[1]), int(color[2]),
                0.0,  # reprojection error (unused by MILo)
            ))
            f.write(struct.pack('<Q', 0))  # track_length = 0 (no 2D observations)
    print(f"[dense_init] Written {len(pts):,} dense points → {bin_path}")


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
    parser.add_argument("--init_pcd", default=None,
                        help="Path to external point cloud PLY for Gaussian initialization "
                             "(injected as sparse/0/points3D.bin before MILo reads it)")
    parser.add_argument("--dense_init_pts", type=int, default=500_000,
                        help="Target point count when subsampling the dense init cloud (default: 500000)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Downscale images by this factor (e.g. 2 = half resolution)")
    parser.add_argument("--difix3d", action="store_true",
                        help="Enable Difix3D+ fix cycles within MILo training "
                             "(3 cycles at iters 9000, 13000, 17000 by default)")
    parser.add_argument("--difix3d_views", type=int, default=16,
                        help="Novel views per Difix3D+ fix cycle (default: 16)")
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

    # ── Pre-place external point cloud (dense init) ───────────
    if args.init_pcd:
        init_src = Path(args.init_pcd)
        if not init_src.exists():
            print(f"Warning: --init_pcd file not found: {init_src}")
        else:
            _process_dense_init(init_src, sparse_path, args.dense_init_pts)

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
        if args.resolution:
            cmd.extend(["-r", str(args.resolution)])
        if args.difix3d:
            cmd.extend(["--difix3d", "--difix3d_views", str(args.difix3d_views)])

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
