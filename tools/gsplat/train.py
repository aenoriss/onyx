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
EVAL_MANIFEST = ".eval_frames.json"


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


def separate_eval_frames(scene_path):
    """Move eval frames from images/ to images_eval/ before training.

    Reads the eval manifest written by the ingest step. Eval frames were
    kept in images/ during SfM so they got poses, but must be separated
    before gsplat training so they're not used as training data.

    Also parses COLMAP images.bin to save eval camera poses for DiFix
    progressive shift and post-training quality evaluation.

    Returns (eval_dir, eval_poses_path) or (None, None) if no eval frames.
    """
    import json
    import struct
    import shutil

    manifest_path = os.path.join(scene_path, EVAL_MANIFEST)
    if not os.path.exists(manifest_path):
        return None, None

    with open(manifest_path) as f:
        manifest = json.load(f)

    eval_filenames = set(manifest.get("eval_filenames", []))
    if not eval_filenames:
        return None, None

    # Find images directory
    src_colmap = os.path.join(scene_path, "colmap")
    if os.path.exists(src_colmap):
        images_dir = os.path.join(src_colmap, "images")
        sparse_dir = os.path.join(src_colmap, "sparse", "0")
    else:
        images_dir = os.path.join(scene_path, "images")
        sparse_dir = os.path.join(scene_path, "sparse", "0")

    eval_dir = os.path.join(scene_path, "images_eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Move eval images out of training set
    moved = 0
    for fname in eval_filenames:
        src = os.path.join(images_dir, fname)
        dst = os.path.join(eval_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
            moved += 1

    print(f"[EVAL] Moved {moved} eval images → {eval_dir}")

    # Parse eval camera poses FIRST (before rewriting images.bin)
    eval_poses_path = os.path.join(scene_path, "eval_cameras.pt")
    images_bin = os.path.join(sparse_dir, "images.bin")
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")

    # Reuse existing eval_cameras.pt if available (e.g. from a previous run
    # that already parsed and rewrote images.bin)
    if os.path.exists(eval_poses_path):
        print(f"[EVAL] Reusing existing {eval_poses_path}")
    elif os.path.exists(images_bin):
        try:
            eval_poses = _parse_eval_poses(images_bin, cameras_bin, eval_filenames)
            if eval_poses:
                import torch
                torch.save(eval_poses, eval_poses_path)
                print(f"[EVAL] Saved {len(eval_poses['c2ws'])} eval camera poses → {eval_poses_path}")
            else:
                eval_poses_path = None
        except Exception as e:
            print(f"[EVAL] Warning: could not parse eval poses: {e}")
            eval_poses_path = None
    else:
        eval_poses_path = None

    # Rewrite COLMAP images.bin WITHOUT eval entries.
    # nerfstudio crashes on FileNotFoundError if images.bin references
    # files that don't exist on disk. Must strip eval entries after parsing.
    if moved > 0 and os.path.exists(images_bin):
        _rewrite_images_bin_without(images_bin, eval_filenames)
        print(f"[EVAL] Rewrote images.bin (removed eval entries)")

    return eval_dir, eval_poses_path


def _parse_eval_poses(images_bin, cameras_bin, eval_filenames):
    """Parse COLMAP images.bin and cameras.bin to extract eval camera c2w + intrinsics."""
    import struct
    import numpy as np

    # Read cameras.bin → camera_id: (model, width, height, fx, fy, cx, cy)
    cameras = {}
    with open(cameras_bin, "rb") as f:
        n_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_cameras):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            # PINHOLE (1): fx, fy, cx, cy
            if model_id == 1:
                params = struct.unpack("<4d", f.read(32))
                cameras[cam_id] = {
                    "width": width, "height": height,
                    "fx": params[0], "fy": params[1],
                    "cx": params[2], "cy": params[3],
                }
            else:
                # SIMPLE_PINHOLE (0): f, cx, cy
                if model_id == 0:
                    params = struct.unpack("<3d", f.read(24))
                    cameras[cam_id] = {
                        "width": width, "height": height,
                        "fx": params[0], "fy": params[0],
                        "cx": params[1], "cy": params[2],
                    }
                else:
                    # COLMAP camera model param counts (from colmap/src/base/camera_models.h)
                    # COLMAP camera model param counts
                    # (from colmap/src/colmap/sensor/models.h)
                    param_counts = {
                        0: 3,   # SIMPLE_PINHOLE
                        1: 4,   # PINHOLE
                        2: 4,   # SIMPLE_RADIAL
                        3: 5,   # RADIAL
                        4: 8,   # OPENCV
                        5: 8,   # OPENCV_FISHEYE
                        6: 12,  # FULL_OPENCV
                        7: 5,   # FOV
                        8: 4,   # SIMPLE_RADIAL_FISHEYE
                        9: 5,   # RADIAL_FISHEYE
                        10: 12, # THIN_PRISM_FISHEYE
                    }
                    n_params = param_counts.get(model_id, 4)
                    f.read(n_params * 8)

    # Read images.bin → filter for eval filenames
    from scipy.spatial.transform import Rotation

    c2ws = []
    fxs, fys, cxs, cys = [], [], [], []
    widths, heights = [], []
    found_names = []

    with open(images_bin, "rb") as f:
        n_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_images):
            img_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]

            # Read image name (null-terminated string)
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c.decode("ascii"))
            name = "".join(name_chars)

            # Skip 2D points
            n_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n_pts2d * 24)  # each point: x(d), y(d), id(Q)

            if name not in eval_filenames:
                continue

            # Convert quaternion + translation to c2w [3, 4]
            # COLMAP stores w2c: R @ world_point + t = camera_point
            R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t_w2c = np.array([tx, ty, tz])
            # c2w = inverse of w2c
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ t_w2c
            c2w = np.zeros((3, 4), dtype=np.float32)
            c2w[:3, :3] = R_c2w
            c2w[:3, 3] = t_c2w

            c2ws.append(c2w)
            found_names.append(name)

            cam = cameras.get(cam_id, {})
            fxs.append(cam.get("fx", 1000))
            fys.append(cam.get("fy", 1000))
            cxs.append(cam.get("cx", cam.get("width", 1000) / 2))
            cys.append(cam.get("cy", cam.get("height", 1000) / 2))
            widths.append(cam.get("width", 1000))
            heights.append(cam.get("height", 1000))

    if not c2ws:
        return None

    import torch
    c2ws_tensor = torch.tensor(np.stack(c2ws), dtype=torch.float32)

    # NOTE: These c2w matrices are in raw COLMAP space. Nerfstudio applies
    # a coordinate transform + scale when loading cameras (saved in
    # dataparser_transforms.json). The ProgressiveShifter must apply this
    # same transform before using eval cameras. We store raw poses here
    # and the transform is applied at load time in difix_integration.py.
    return {
        "c2ws": c2ws_tensor,
        "fx": torch.tensor(fxs, dtype=torch.float32),
        "fy": torch.tensor(fys, dtype=torch.float32),
        "cx": torch.tensor(cxs, dtype=torch.float32),
        "cy": torch.tensor(cys, dtype=torch.float32),
        "width": torch.tensor(widths, dtype=torch.long),
        "height": torch.tensor(heights, dtype=torch.long),
        "filenames": found_names,
    }


def _rewrite_images_bin_without(images_bin_path, exclude_filenames):
    """Rewrite COLMAP images.bin excluding specified image entries.

    Reads the binary file, filters out entries whose name is in
    exclude_filenames, and writes back. This prevents nerfstudio from
    crashing on FileNotFoundError for moved eval images.
    """
    import struct
    import shutil

    exclude = set(exclude_filenames)

    # Read all entries
    entries = []
    with open(images_bin_path, "rb") as f:
        n_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_images):
            entry_start = f.tell()
            img_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]

            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c.decode("ascii"))
            name = "".join(name_chars)

            n_pts2d = struct.unpack("<Q", f.read(8))[0]
            pts2d_data = f.read(n_pts2d * 24)

            if name not in exclude:
                entries.append((img_id, qw, qx, qy, qz, tx, ty, tz,
                               cam_id, name, n_pts2d, pts2d_data))

    # Backup original
    backup = images_bin_path + ".with_eval_backup"
    if not os.path.exists(backup):
        shutil.copy2(images_bin_path, backup)

    # Write filtered entries
    with open(images_bin_path, "wb") as f:
        f.write(struct.pack("<Q", len(entries)))
        for (img_id, qw, qx, qy, qz, tx, ty, tz,
             cam_id, name, n_pts2d, pts2d_data) in entries:
            f.write(struct.pack("<I", img_id))
            f.write(struct.pack("<4d", qw, qx, qy, qz))
            f.write(struct.pack("<3d", tx, ty, tz))
            f.write(struct.pack("<I", cam_id))
            f.write(name.encode("ascii") + b"\x00")
            f.write(struct.pack("<Q", n_pts2d))
            f.write(pts2d_data)


def evaluate_quality(config_path, eval_dir, output_path, eval_poses_path=None):
    """Post-training quality evaluation: render at eval cameras, compute metrics.

    Loads the trained model, renders at held-out eval camera poses, and
    compares against ground-truth eval images. Reports per-image and
    aggregate PSNR and SSIM.

    Results saved to {output_path}/eval_metrics.json.
    """
    import json
    import torch
    import numpy as np
    from pathlib import Path
    from PIL import Image as PILImage

    print(f"\n{'='*50}")
    print("Post-Training Quality Evaluation")
    print(f"{'='*50}")

    # Need eval camera poses
    if not eval_poses_path or not os.path.exists(eval_poses_path):
        print("[EVAL] No eval_cameras.pt found, skipping quality evaluation")
        return None

    try:
        from nerfstudio.utils.eval_utils import eval_setup
        from nerfstudio.cameras.cameras import Cameras
    except ImportError:
        print("[EVAL] nerfstudio not available, skipping quality evaluation")
        return None

    try:
        # Load trained model
        _, pipeline, _, _ = eval_setup(Path(config_path))
        pipeline.eval()
        device = pipeline.device

        # Load eval camera data
        eval_data = torch.load(eval_poses_path, map_location="cpu")
        eval_c2ws = eval_data["c2ws"]
        eval_filenames = eval_data["filenames"]

        psnrs = []
        ssims = []
        eval_output = os.path.join(output_path, "eval_renders")
        os.makedirs(eval_output, exist_ok=True)

        for i, fname in enumerate(eval_filenames):
            gt_path = os.path.join(eval_dir, fname)
            if not os.path.exists(gt_path):
                continue

            # Load ground truth
            gt_img = np.array(PILImage.open(gt_path)).astype(np.float32) / 255.0
            gt_tensor = torch.tensor(gt_img, dtype=torch.float32)

            # Create eval camera
            cam = Cameras(
                camera_to_worlds=eval_c2ws[i].unsqueeze(0),
                fx=eval_data["fx"][i].unsqueeze(0),
                fy=eval_data["fy"][i].unsqueeze(0),
                cx=eval_data["cx"][i].unsqueeze(0),
                cy=eval_data["cy"][i].unsqueeze(0),
                width=eval_data["width"][i].unsqueeze(0),
                height=eval_data["height"][i].unsqueeze(0),
            ).to(device)

            # Render
            with torch.no_grad():
                outputs = pipeline.model.get_outputs(cam)
                rendered = outputs["rgb"].cpu()  # [H, W, 3]

            # Resize GT to match render if needed
            if gt_tensor.shape[:2] != rendered.shape[:2]:
                gt_pil = PILImage.fromarray((gt_tensor.numpy() * 255).astype(np.uint8))
                gt_pil = gt_pil.resize((rendered.shape[1], rendered.shape[0]), PILImage.LANCZOS)
                gt_tensor = torch.tensor(np.array(gt_pil).astype(np.float32) / 255.0)

            # PSNR
            mse = ((rendered - gt_tensor) ** 2).mean()
            psnr = (-10 * torch.log10(mse + 1e-8)).item()
            psnrs.append(psnr)

            # SSIM (simple structural similarity)
            try:
                from torchmetrics.functional import structural_similarity_index_measure
                ssim_val = structural_similarity_index_measure(
                    rendered.permute(2, 0, 1).unsqueeze(0),
                    gt_tensor.permute(2, 0, 1).unsqueeze(0),
                ).item()
            except ImportError:
                ssim_val = 0.0
            ssims.append(ssim_val)

            # Save render + GT side by side (first 8 only)
            if i < 8:
                render_pil = PILImage.fromarray(
                    (rendered.clamp(0, 1).numpy() * 255).astype(np.uint8))
                render_pil.save(os.path.join(eval_output, f"render_{i:02d}.jpg"))

        if not psnrs:
            print("[EVAL] No eval images matched, skipping")
            return None

        metrics = {
            "psnr_mean": float(np.mean(psnrs)),
            "psnr_std": float(np.std(psnrs)),
            "ssim_mean": float(np.mean(ssims)),
            "ssim_std": float(np.std(ssims)),
            "n_eval_images": len(psnrs),
        }

        # Save metrics
        metrics_path = os.path.join(output_path, "eval_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[EVAL] {len(psnrs)} eval images evaluated")
        print(f"[EVAL] PSNR:  {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
        print(f"[EVAL] SSIM:  {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
        print(f"[EVAL] Renders → {eval_output}")
        print(f"[EVAL] Metrics → {metrics_path}")

        return metrics

    except Exception as e:
        print(f"[EVAL] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


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


def downscale_images(ws, factor):
    """Pre-create downscaled images so nerfstudio doesn't prompt interactively.

    nerfstudio looks for {images_path}_{factor}/ (e.g. colmap/images_2/).
    If missing, it prompts via stdin which fails in non-interactive Docker.
    """
    from PIL import Image

    src_dir = os.path.join(ws, "colmap", "images")
    dst_dir = os.path.join(ws, "colmap", f"images_{factor}")

    if os.path.exists(dst_dir) and len(os.listdir(dst_dir)) > 0:
        print(f"[DOWNSCALE] Reusing existing images_{factor}/ ({len(os.listdir(dst_dir))} files)")
        return

    os.makedirs(dst_dir, exist_ok=True)
    exts = ('.png', '.jpg', '.jpeg')
    images = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(exts))
    print(f"[DOWNSCALE] Creating {len(images)} images at 1/{factor} resolution...")

    for fname in images:
        img = Image.open(os.path.join(src_dir, fname))
        w, h = img.size
        img_resized = img.resize((w // factor, h // factor), Image.LANCZOS)
        img_resized.save(os.path.join(dst_dir, fname))

    print(f"[DOWNSCALE] Done: {dst_dir} ({len(images)} files)")


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
    parser.add_argument("--mcmc", action="store_true",
                        help="Use MCMC densification strategy instead of ADC")
    parser.add_argument("--init_pcd", default=None,
                        help="Path to dense point cloud PLY for initialization "
                             "(replaces sparse points3D.bin)")
    parser.add_argument("--dense_init_pts", type=int, default=100_000,
                        help="Target point count when subsampling dense init cloud")
    parser.add_argument("--antialiased", action="store_true",
                        help="Enable Mip-Splatting anti-aliasing (adapts gaussian size to "
                             "pixel footprint — reduces blur at distance, improves sharpness)")
    parser.add_argument("--bilateral-grid", action="store_true",
                        help="Enable bilateral grid for per-image exposure/WB correction "
                             "(useful for outdoor/360, causes shadow artifacts indoors)")
    parser.add_argument("--depth", action="store_true",
                        help="Enable depth supervision with Depth Anything V2 priors "
                             "(constrains Gaussians to surfaces)")
    parser.add_argument("--depth-weight", type=float, default=0.2,
                        help="Depth loss weight (default: 0.2)")
    parser.add_argument("--normal-weight", type=float, default=None,
                        help="Normal regularization weight (default: 0.02 with depth, 0 without)")
    parser.add_argument("--max-gs-num", type=int, default=None,
                        help="Maximum number of Gaussians (default: 300K with depth, 1M without)")
    parser.add_argument("--difix3d", action="store_true",
                        help="Enable DiFix3D+ interleaved fix cycles during training")
    parser.add_argument("--difix3d-views", type=int, default=36,
                        help="Novel views per DiFix3D+ fix cycle (default: 36)")
    parser.add_argument("--difix3d-lambda", type=float, default=0.40,
                        help="Probability of novel data reuse per iteration (default: 0.40)")
    parser.add_argument("--difix3d-tau", type=int, default=200,
                        help="DiFix noise level tau (default: 200, NVIDIA default)")
    parser.add_argument("--tiles-per-frame", type=int, default=16,
                        help="Tiles per frame for 360 cameras (1=regular, 16=360 rig)")
    parser.add_argument("--holdout", action="store_true",
                        help="Hold out eval frames for quality metrics (reduces training data)")
    args = parser.parse_args()

    scene_path = os.path.abspath(args.scene)
    total_steps = 2

    # Set output path
    output_path = args.output or os.path.join(scene_path, "output", "splatfacto")
    os.makedirs(output_path, exist_ok=True)

    # ── Eval frame holdout (only with --holdout flag) ────────
    # By default, ALL images train. DiFix uses synthetic midpoint targets.
    # With --holdout: eval frames are separated for PSNR/SSIM metrics,
    # and their poses are used as DiFix progressive shift targets.
    eval_dir = None
    eval_poses_path = None
    if args.holdout:
        eval_dir, eval_poses_path = separate_eval_frames(scene_path)
        if eval_dir:
            n_eval = len([f for f in os.listdir(eval_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"[EVAL] {n_eval} eval images held out for quality metrics")

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
    print(f"BilGrid:   {'Yes' if args.bilateral_grid else 'No'}")
    print(f"Depth:     {'DAv2 (weight=' + str(args.depth_weight) + ')' if args.depth else 'No'}")
    # Normal loss defaults to 0.02 when depth is enabled
    if args.normal_weight is None:
        args.normal_weight = 0.02 if args.depth else 0.0
    print(f"Normals:   {'weight=' + str(args.normal_weight) if args.normal_weight > 0 else 'No'}")
    print(f"DiFix3D+:  {'Yes (views=' + str(args.difix3d_views) + ', tau=' + str(args.difix3d_tau) + ')' if args.difix3d else 'No'}")
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

    # ── Pre-stage: Generate depth maps (DAv2) ────────────────
    if args.depth:
        colmap_images = os.path.join(ws, "colmap", "images")
        depths_dir = os.path.join(ws, "colmap", "depths")
        if os.path.exists(depths_dir) and len(os.listdir(depths_dir)) > 0:
            print(f"[DEPTH] Reusing existing depth maps ({len(os.listdir(depths_dir))} files)")
        else:
            progress("depth_generation", "running", 0, step=1, total_steps=total_steps + 1)
            print("[DEPTH] Generating monocular depth priors with Depth Anything V2...")
            depth_cmd = [
                "python", "-u", "/workspace/generate_depths.py",
                "--images", colmap_images,
                "--output", depths_dir,
            ]
            subprocess.run(depth_cmd, check=True)
        total_steps += 1

    # ── Pre-stage: Downscale images if needed ───────────────
    if args.resolution > 1:
        downscale_images(ws, args.resolution)

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
        # MCMC relocation threshold (paper default 0.005 — higher values cause
        # excessive relocation churn → intersection buffer OOM spikes)
        "--pipeline.model.cull-alpha-thresh", "0.005",
        # Per-image exposure/WB correction (off by default, causes shadow artifacts on ceilings)
        "--pipeline.model.use-bilateral-grid", str(args.bilateral_grid),
        "--pipeline.model.rasterize-mode", "antialiased" if args.antialiased else "classic",
        # Gaussian cap — 500K with depth (MCMC never prunes, excess are dead weight
        # that inflate isect_tiles buffer → OOM spikes). 1M without depth.
        "--pipeline.model.max-gs-num", str(args.max_gs_num or 500000),
        # Learn pose corrections from SfM imprecision
        "--pipeline.model.camera-optimizer.mode", "SO3xR3",
        # Depth supervision (Depth Anything V2 priors)
        # Uses native RGB+ED render (depth from same rasterization, zero extra VRAM)
        # Gradient freeze (DNGaussian) prevents depth from inflating scales
        "--pipeline.model.depth-loss-mult", str(args.depth_weight if args.depth else 0.0),
        "--pipeline.model.normal-loss-mult", str(args.normal_weight),
        "--pipeline.model.output-depth-during-training", str(args.depth or args.normal_weight > 0),
        # DiFix3D+ interleaved fix cycles
        "--pipeline.model.difix3d-enabled", str(args.difix3d),
        "--pipeline.model.difix3d-views", str(args.difix3d_views),
        "--pipeline.model.difix3d-lambda", str(args.difix3d_lambda),
        "--pipeline.model.difix3d-tau", str(args.difix3d_tau),
        "--pipeline.model.difix3d-total-iters", str(args.iterations),
        "--pipeline.model.difix3d-tiles-per-frame", str(args.tiles_per_frame),
        "--pipeline.model.difix3d-eval-poses-path", str(eval_poses_path or ""),
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

    # ── Stage 3: Quality evaluation (eval frames) ────────────
    if eval_dir and os.path.exists(eval_dir):
        evaluate_quality(config_path, eval_dir, output_path,
                        eval_poses_path=eval_poses_path)

    progress("done", "completed", 100, step=total_steps, total_steps=total_steps)


if __name__ == "__main__":
    main()
