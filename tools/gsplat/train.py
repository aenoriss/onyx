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


def _process_dense_init(init_src, sparse_dir, target_pts=100_000):
    """Subsample a dense PLY cloud and inject it as points3D.bin.

    Replaces the symlinked points3D.bin with a real file containing dense
    MVS points. Same logic as MILo's train_wrapper._process_dense_init.
    """
    import struct
    import shutil
    import numpy as np

    try:
        import open3d as o3d
    except ImportError:
        # Fallback: read PLY with trimesh or plyfile
        import plyfile
        plydata = plyfile.PlyData.read(init_src)
        vertex = plydata['vertex']
        pts = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
        try:
            colors = np.column_stack([vertex['red'], vertex['green'], vertex['blue']]).astype(np.float64) / 255.0
        except ValueError:
            colors = np.ones((len(pts), 3), dtype=np.float64)
        print(f"[dense_init] Loaded {len(pts):,} points (plyfile)")
        # SOR not available without open3d, skip
        _center = pts.mean(axis=0)
        _dists = np.linalg.norm(pts - _center, axis=1)
        _clip = np.percentile(_dists, 99)
        mask = _dists <= _clip
        pts, colors = pts[mask], colors[mask]
        print(f"[dense_init] After p99 clip: {len(pts):,} points")
        if len(pts) > target_pts:
            idx = np.random.choice(len(pts), target_pts, replace=False)
            pts, colors = pts[idx], colors[idx]
        rgb = (colors * 255).clip(0, 255).astype(np.uint8)
        print(f"[dense_init] Final count: {len(pts):,} points")
        _write_points3d_bin(pts, rgb, sparse_dir)
        return

    print(f"[dense_init] Loading {init_src} ...")
    pcd = o3d.io.read_point_cloud(str(init_src))
    print(f"[dense_init] Loaded {len(pcd.points):,} points")

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"[dense_init] After SOR: {len(pcd.points):,} points")

    _pts = np.asarray(pcd.points)
    _center = _pts.mean(axis=0)
    _dists = np.linalg.norm(_pts - _center, axis=1)
    _clip = np.percentile(_dists, 99)
    _mask = _dists <= _clip
    _n_clipped = len(_pts) - _mask.sum()
    if _n_clipped > 0:
        pcd = pcd.select_by_index(np.where(_mask)[0])
        print(f"[dense_init] After p99 clip: {len(pcd.points):,} points")

    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(colors) == 0 or colors.shape[0] != len(pts):
        colors = np.ones((len(pts), 3), dtype=np.float32)
    if len(pts) > target_pts:
        idx = np.random.choice(len(pts), target_pts, replace=False)
        pts, colors = pts[idx], colors[idx]

    rgb = (colors * 255).clip(0, 255).astype(np.uint8)
    print(f"[dense_init] Final count: {len(pts):,} points")
    _write_points3d_bin(pts, rgb, sparse_dir)


def _write_points3d_bin(pts, rgb, sparse_dir):
    """Write COLMAP binary points3D.bin with empty tracks."""
    import struct
    import shutil

    bin_path = os.path.join(sparse_dir, "points3D.bin")
    backup_path = os.path.join(sparse_dir, "points3D.bin.sparse_backup")

    # Remove symlink and backup original if needed
    if os.path.islink(bin_path):
        real_path = os.path.realpath(bin_path)
        os.unlink(bin_path)
        if not os.path.exists(backup_path):
            shutil.copy2(real_path, backup_path)
            print(f"[dense_init] Backed up sparse cloud → {os.path.basename(backup_path)}")
    elif os.path.exists(bin_path) and not os.path.exists(backup_path):
        shutil.copy2(bin_path, backup_path)
        print(f"[dense_init] Backed up sparse cloud → {os.path.basename(backup_path)}")

    with open(bin_path, 'wb') as f:
        f.write(struct.pack('<Q', len(pts)))
        for i, (pt, color) in enumerate(zip(pts, rgb)):
            f.write(struct.pack('<QdddBBBd',
                i + 1,
                float(pt[0]), float(pt[1]), float(pt[2]),
                int(color[0]), int(color[1]), int(color[2]),
                0.0,
            ))
            f.write(struct.pack('<Q', 0))
    print(f"[dense_init] Written {len(pts):,} dense points → {bin_path}")


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
    parser.add_argument("--iterations", "-i", type=int, default=60000,
                        help="Number of training iterations (default: 60000)")
    parser.add_argument("--resolution", "-r", type=int, default=1,
                        help="Resolution scale factor (1=full, 2=half, 4=quarter)")
    parser.add_argument("--eval", action="store_true",
                        help="Enable eval mode (hold out test images)")
    parser.add_argument("--mcmc", action="store_true",
                        help="Use MCMC densification strategy instead of ADC")
    parser.add_argument("--init_pcd", default=None,
                        help="Path to dense point cloud PLY for initialization "
                             "(replaces sparse points3D.bin)")
    parser.add_argument("--dense_init_pts", type=int, default=100_000,
                        help="Target point count when subsampling dense init cloud")
    parser.add_argument("--bilateral-grid", action="store_true",
                        help="Enable bilateral grid for per-image exposure/WB correction "
                             "(useful for outdoor/360, causes shadow artifacts indoors)")
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
    print(f"Strategy:  {'MCMC' if args.mcmc else 'ADC (default)'}")
    print(f"BilGrid:   Yes")
    print(f"CamOptim:  SO3xR3")
    if args.init_pcd:
        print(f"Dense init: {args.init_pcd} (target {args.dense_init_pts:,} pts)")
    print("=" * 50)

    # ── Pre-place external point cloud (dense init) ───────────
    if args.init_pcd and os.path.exists(args.init_pcd):
        sparse_dir = os.path.join(ws, "colmap", "sparse", "0")
        _process_dense_init(args.init_pcd, sparse_dir, args.dense_init_pts)
    elif args.init_pcd:
        print(f"Warning: --init_pcd file not found: {args.init_pcd}")

    # ── Stage 1: Training ─────────────────────────────────────
    method = "splatfacto-mcmc" if args.mcmc else "splatfacto"
    cmd = [
        "ns-train", method,
        "--output-dir", output_path,
        "--max-num-iterations", str(args.iterations),
        "--vis", "tensorboard",
        "--steps-per-eval-batch", "0",
        "--steps-per-eval-image", "0",
        "--steps-per-eval-all-images", "0",
        "--steps-per-save", str(args.iterations),
        # Quality: lower cull threshold preserves subtle Gaussians (splatfacto-big default)
        "--pipeline.model.cull-alpha-thresh", "0.005",
        # Quality: lower densify threshold = more splits = more detail (splatfacto-big default)
        "--pipeline.model.densify-grad-thresh", "0.0005",
        # Quality: penalizes elongated spiky Gaussians (PhysGaussian)
        "--pipeline.model.use-scale-regularization", "True",
        # Quality: tighter max ratio = more uniform Gaussians, fewer needles
        "--pipeline.model.max-gauss-ratio", "5.0",
        # Quality: per-image exposure/WB correction (off by default, causes shadow artifacts on ceilings)
        "--pipeline.model.use-bilateral-grid", str(args.bilateral_grid),
        # Quality: 1M Gaussian cap (2M had 62% pruned at export — diminishing returns)
        "--pipeline.model.max-gs-num", "1000000",
        # Quality: allow densification through 80% of training
        "--pipeline.model.stop-split-at", "50000",
        # Quality: learn pose corrections from SfM imprecision
        "--pipeline.model.camera-optimizer.mode", "SO3xR3",
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

    # nerfstudio stores splatfacto-mcmc output under "splatfacto/" not "splatfacto-mcmc/"
    config_paths = glob.glob(os.path.join(output_path, "*", "*", "*", "config.yml"))
    if not config_paths:
        config_paths = glob.glob(os.path.join(output_path, "*", "*", "config.yml"))

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
