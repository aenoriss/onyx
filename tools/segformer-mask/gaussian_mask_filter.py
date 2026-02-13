#!/usr/bin/env python3
"""
Clean-GS Style Gaussian Mask Filter
====================================

Removes Gaussians that don't belong to the target object using semantic masks.
Based on Clean-GS (arXiv:2601.00913) three-stage approach.

STAGES:
-------
Stage 1: Whitelist Filtering
    - Project each Gaussian to sampled camera views
    - Keep if it projects to WHITE mask region in >= min_views views
    - Removes background and environment Gaussians

Stage 2: Color Validation (requires --images)
    - For each view, use depth buffering to find front-layer Gaussian at each pixel
    - Compare Gaussian's SH DC color to source image color: RGB = C0 * f_dc + 0.5
    - CONSERVATIVE: Keep if color matches (δ < threshold) in AT LEAST ONE view
    - Keeps occluded Gaussians (can't validate them)
    - Removes floaters with wrong colors

Stage 3: Outlier Removal
    - Neighbor-based: Remove Gaussians far from k nearest neighbors
    - Spatial: Remove Gaussians far from scene center
    - Cleans isolated floaters that passed earlier stages

CHECKPOINT FILTERING:
--------------------
When --checkpoint is provided, the script also filters the checkpoint tensors
so that mesh extraction can use the filtered Gaussians.

CAMERA FILTERING:
-----------------
Cameras with very low mask coverage (mostly sky/background) are automatically
excluded. This prevents mesh extraction from failing on views with no geometry.
A filtered cameras.json is saved alongside the output PLY.

USAGE:
------
Filter PLY only:
    python gaussian_mask_filter.py \
        --ply /path/to/point_cloud.ply \
        --cameras /path/to/cameras.json \
        --masks /path/to/masks_folder \
        --output /path/to/filtered.ply

Filter PLY + Checkpoint (for mesh extraction):
    python gaussian_mask_filter.py \
        --ply /path/to/point_cloud.ply \
        --cameras /path/to/cameras.json \
        --masks /path/to/masks_folder \
        --output /path/to/filtered.ply \
        --checkpoint /path/to/chkpnt18000.pth \
        --checkpoint-output /path/to/chkpnt18000_filtered.pth

Reference: Clean-GS (arXiv:2601.00913) - Semantic Mask-Guided Pruning for 3DGS
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
from plyfile import PlyData, PlyElement
import torch
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from tqdm import tqdm
import warnings


@dataclass
class Camera:
    """Camera with intrinsics and extrinsics (world-to-camera convention)."""
    id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R_w2c: np.ndarray  # 3x3 rotation (world to camera)
    t_w2c: np.ndarray  # 3x1 translation (world to camera)
    image_name: str


def load_ply(path: str) -> Tuple[np.ndarray, np.ndarray, PlyData]:
    """Load 3DGS PLY file and extract Gaussian positions and colors."""
    plydata = PlyData.read(path)
    vertex = plydata['vertex']

    positions = np.stack([
        vertex['x'],
        vertex['y'],
        vertex['z']
    ], axis=1).astype(np.float32)

    # Extract DC color components (SH band 0)
    colors_dc = np.stack([
        vertex['f_dc_0'],
        vertex['f_dc_1'],
        vertex['f_dc_2']
    ], axis=1).astype(np.float32)

    print(f"[INFO] Loaded {len(positions):,} Gaussians from {path}")
    return positions, colors_dc, plydata


def save_filtered_ply(plydata: PlyData, mask: np.ndarray, output_path: str):
    """Save filtered PLY with only the Gaussians where mask is True."""
    vertex = plydata['vertex']
    props = [p.name for p in vertex.properties]

    filtered_data = {prop: vertex[prop][mask] for prop in props}

    dtype = [(p.name, vertex[p.name].dtype) for p in vertex.properties]
    new_vertex = np.empty(mask.sum(), dtype=dtype)
    for prop in props:
        new_vertex[prop] = filtered_data[prop]

    new_element = PlyElement.describe(new_vertex, 'vertex')
    new_plydata = PlyData([new_element])
    new_plydata.write(output_path)

    print(f"[INFO] Saved {mask.sum():,} Gaussians to {output_path}")


def filter_checkpoint(checkpoint_path: str, mask: np.ndarray, output_path: str):
    """
    Filter a MILo/3DGS checkpoint to keep only Gaussians where mask is True.

    The checkpoint contains tensors for all Gaussian parameters. We apply the
    boolean mask to filter them, then save a new checkpoint.
    """
    print(f"\n[CHECKPOINT] Filtering checkpoint...")
    print(f"  - Input: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Convert numpy mask to torch
    mask_torch = torch.from_numpy(mask)
    n_original = mask.shape[0]
    n_filtered = mask.sum()

    # Keys that contain per-Gaussian data (need to be filtered)
    # These are the standard 3DGS/MILo parameter names
    gaussian_keys = [
        '_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity',
        'xyz', 'features_dc', 'features_rest', 'scaling', 'rotation', 'opacity',
    ]

    # Also filter optimizer state if present
    filtered_count = 0

    for key in list(checkpoint.keys()):
        if isinstance(checkpoint[key], torch.Tensor):
            if checkpoint[key].shape[0] == n_original:
                checkpoint[key] = checkpoint[key][mask_torch]
                filtered_count += 1
                print(f"  - Filtered tensor '{key}': {n_original} -> {n_filtered}")
        elif isinstance(checkpoint[key], dict):
            # Handle nested dicts (like optimizer state)
            for subkey in list(checkpoint[key].keys()):
                if isinstance(checkpoint[key][subkey], torch.Tensor):
                    if checkpoint[key][subkey].shape[0] == n_original:
                        checkpoint[key][subkey] = checkpoint[key][subkey][mask_torch]
                        filtered_count += 1

    # Save filtered checkpoint
    torch.save(checkpoint, output_path)
    print(f"  - Output: {output_path}")
    print(f"  - Filtered {filtered_count} tensors")

    return output_path


def read_colmap_cameras_binary(path: str) -> Dict:
    """Read COLMAP cameras.bin file."""
    cameras = {}
    with open(path, "rb") as f:
        import struct
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<I", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            model_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5, 6: 8, 7: 12, 8: 5, 9: 4, 10: 5}
            num_params = model_params.get(model_id, 4)
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[camera_id] = {
                'model_id': model_id, 'width': width, 'height': height, 'params': params
            }
    return cameras


def read_colmap_images_binary(path: str) -> Dict:
    """Read COLMAP images.bin file."""
    import struct
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            name = name.decode("utf-8")
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_points2D)
            images[image_id] = {
                'qvec': qvec, 'tvec': tvec, 'camera_id': camera_id, 'name': name
            }
    return images


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float32)


def load_cameras_colmap(sparse_path: str) -> List[Camera]:
    """Load cameras from COLMAP binary files."""
    sparse_path = Path(sparse_path)

    if (sparse_path / "cameras.bin").exists():
        cameras_file = sparse_path / "cameras.bin"
        images_file = sparse_path / "images.bin"
    elif (sparse_path / "0" / "cameras.bin").exists():
        cameras_file = sparse_path / "0" / "cameras.bin"
        images_file = sparse_path / "0" / "images.bin"
    elif (sparse_path / "sparse" / "0" / "cameras.bin").exists():
        cameras_file = sparse_path / "sparse" / "0" / "cameras.bin"
        images_file = sparse_path / "sparse" / "0" / "images.bin"
    else:
        raise FileNotFoundError(f"Could not find COLMAP cameras.bin in {sparse_path}")

    colmap_cameras = read_colmap_cameras_binary(str(cameras_file))
    colmap_images = read_colmap_images_binary(str(images_file))

    cameras = []
    for img_id, img_data in colmap_images.items():
        cam_data = colmap_cameras[img_data['camera_id']]
        R_w2c = qvec2rotmat(img_data['qvec'])
        t_w2c = np.array(img_data['tvec'], dtype=np.float32)

        width = cam_data['width']
        height = cam_data['height']
        params = cam_data['params']
        model_id = cam_data['model_id']

        if model_id in [0, 1]:
            fx = params[0]
            fy = params[1] if model_id == 1 else params[0]
            cx = params[1] if model_id == 0 else params[2]
            cy = params[2] if model_id == 0 else params[3]
        elif model_id in [2, 3, 4, 5]:
            fx = params[0]
            fy = params[1] if model_id >= 3 else params[0]
            cx = params[1] if model_id <= 2 else params[2]
            cy = params[2] if model_id <= 2 else params[3]
        else:
            fx = fy = params[0]
            cx = width / 2
            cy = height / 2

        img_name = Path(img_data['name']).stem

        cameras.append(Camera(
            id=img_id, width=width, height=height,
            fx=fx, fy=fy, cx=cx, cy=cy,
            R_w2c=R_w2c, t_w2c=t_w2c, image_name=img_name
        ))

    print(f"[INFO] Loaded {len(cameras)} cameras from COLMAP")
    return cameras


def load_cameras_json(path: str) -> List[Camera]:
    """Load cameras from MILo/3DGS cameras.json format."""
    with open(path, 'r') as f:
        data = json.load(f)

    cameras = []
    for cam in data:
        R_c2w = np.array(cam['rotation']).reshape(3, 3).astype(np.float32)
        R_w2c = R_c2w.T
        position = np.array(cam['position']).astype(np.float32)
        t_w2c = -R_w2c @ position

        width = cam['width']
        height = cam['height']

        if 'fx' in cam:
            fx, fy = cam['fx'], cam['fy']
        elif 'FovX' in cam:
            fovx, fovy = cam['FovX'], cam['FovY']
            fx = width / (2 * np.tan(fovx / 2))
            fy = height / (2 * np.tan(fovy / 2))
        else:
            raise ValueError("Camera missing focal length info")

        cx = width / 2
        cy = height / 2

        cameras.append(Camera(
            id=cam.get('id', len(cameras)),
            width=width, height=height,
            fx=fx, fy=fy, cx=cx, cy=cy,
            R_w2c=R_w2c, t_w2c=t_w2c,
            image_name=cam.get('img_name', f'image_{len(cameras):04d}')
        ))

    print(f"[INFO] Loaded {len(cameras)} cameras")
    return cameras


def project_points_batch(
    points: torch.Tensor,
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project N points to M cameras."""
    N = points.shape[0]
    M = R_w2c.shape[0]

    points_expanded = points.unsqueeze(0).expand(M, -1, -1)
    t_expanded = t_w2c.unsqueeze(1)
    points_cam = torch.bmm(points_expanded, R_w2c.transpose(1, 2)) + t_expanded

    x_cam = points_cam[:, :, 0]
    y_cam = points_cam[:, :, 1]
    z_cam = points_cam[:, :, 2]

    depth = z_cam
    z_safe = z_cam.clamp(min=1e-6)

    u = fx.view(M, 1) * (x_cam / z_safe) + cx.view(M, 1)
    v = fy.view(M, 1) * (y_cam / z_safe) + cy.view(M, 1)

    uv = torch.stack([u, v], dim=2)

    valid = (depth > 0) & \
            (u >= 0) & (u < widths.view(M, 1)) & \
            (v >= 0) & (v < heights.view(M, 1))

    return uv, depth, valid


def filter_low_coverage_cameras(
    cameras: List[Camera],
    mask_folder: str,
    min_coverage: float = 5.0,
) -> Tuple[List[Camera], List[Camera]]:
    """
    Filter out cameras where mask has low object coverage (mostly sky/background).

    These cameras would cause mesh extraction to fail because there's no geometry
    visible in those views.

    Returns:
        (kept_cameras, excluded_cameras)
    """
    mask_folder = Path(mask_folder)
    kept = []
    excluded = []

    for cam in cameras:
        # Find mask file
        mask_name = cam.image_name
        if not mask_name.endswith('.png'):
            mask_name += '.png'
        mask_path = mask_folder / mask_name

        if not mask_path.exists():
            mask_path = mask_folder / (Path(cam.image_name).stem + '.png')

        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert('L'))
            white_pct = (mask > 127).sum() / mask.size * 100

            if white_pct >= min_coverage:
                kept.append(cam)
            else:
                excluded.append(cam)
        else:
            # No mask = keep camera (conservative)
            kept.append(cam)

    return kept, excluded


def save_filtered_cameras_json(
    cameras: List[Camera],
    original_json_path: str,
    output_path: str,
):
    """Save filtered cameras.json for mesh extraction."""
    # Load original to preserve all fields
    with open(original_json_path, 'r') as f:
        original_data = json.load(f)

    # Create set of kept camera names
    kept_names = {cam.image_name for cam in cameras}

    # Filter original data
    filtered_data = [
        cam for cam in original_data
        if cam.get('img_name', '') in kept_names
    ]

    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"[INFO] Saved {len(filtered_data)} cameras to {output_path}")


def stage1_whitelist_filtering(
    positions: np.ndarray,
    cameras: List[Camera],
    mask_folder: str,
    num_samples: int = 30,
    min_views: int = 1,
    device: str = 'cuda',
    batch_size: int = 100000,
) -> np.ndarray:
    """Stage 1: Whitelist Filtering - Keep Gaussians in object regions."""
    N = len(positions)
    print(f"\n[STAGE 1] Whitelist Filtering")
    print(f"  - {N:,} Gaussians to process")

    if num_samples < len(cameras):
        indices = np.linspace(0, len(cameras) - 1, num_samples, dtype=int)
        sampled_cameras = [cameras[i] for i in indices]
    else:
        sampled_cameras = cameras
        num_samples = len(cameras)

    print(f"  - Using {num_samples} sampled views")

    mask_folder = Path(mask_folder)
    masks = []
    valid_cameras = []

    for cam in sampled_cameras:
        mask_name = cam.image_name.replace('.jpg', '.png').replace('.JPG', '.png')
        if not mask_name.endswith('.png'):
            mask_name += '.png'
        mask_path = mask_folder / mask_name

        if not mask_path.exists():
            mask_path = mask_folder / (Path(cam.image_name).stem + '.png')

        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert('L'))
            if mask.shape != (cam.height, cam.width):
                mask = np.array(Image.open(mask_path).convert('L').resize(
                    (cam.width, cam.height), Image.NEAREST))
            masks.append(mask)
            valid_cameras.append(cam)

    if len(masks) == 0:
        raise ValueError("No masks found!")

    M = len(masks)
    print(f"  - Loaded {M} masks")

    R_w2c = torch.stack([torch.from_numpy(c.R_w2c) for c in valid_cameras]).to(device)
    t_w2c = torch.stack([torch.from_numpy(c.t_w2c) for c in valid_cameras]).to(device)
    fx = torch.tensor([c.fx for c in valid_cameras], device=device, dtype=torch.float32)
    fy = torch.tensor([c.fy for c in valid_cameras], device=device, dtype=torch.float32)
    cx = torch.tensor([c.cx for c in valid_cameras], device=device, dtype=torch.float32)
    cy = torch.tensor([c.cy for c in valid_cameras], device=device, dtype=torch.float32)
    widths = torch.tensor([c.width for c in valid_cameras], device=device, dtype=torch.float32)
    heights = torch.tensor([c.height for c in valid_cameras], device=device, dtype=torch.float32)

    object_votes = np.zeros(N, dtype=np.int32)
    positions_tensor = torch.from_numpy(positions).to(device)

    for start in tqdm(range(0, N, batch_size), desc="  Projecting"):
        end = min(start + batch_size, N)
        batch_pos = positions_tensor[start:end]

        uv, depth, valid = project_points_batch(
            batch_pos, R_w2c, t_w2c, fx, fy, cx, cy, widths, heights
        )

        uv_cpu = uv.cpu().numpy()
        valid_cpu = valid.cpu().numpy()
        batch_votes = np.zeros(end - start, dtype=np.int32)

        for m in range(M):
            u = uv_cpu[m, :, 0].astype(np.int32)
            v = uv_cpu[m, :, 1].astype(np.int32)
            is_valid = valid_cpu[m]

            u = np.clip(u, 0, valid_cameras[m].width - 1)
            v = np.clip(v, 0, valid_cameras[m].height - 1)

            mask_vals = masks[m][v, u]
            is_object = is_valid & (mask_vals > 0)
            batch_votes += is_object.astype(np.int32)

        object_votes[start:end] = batch_votes

    whitelist = object_votes >= min_views

    n_kept = whitelist.sum()
    n_removed = N - n_kept
    print(f"  - Whitelisted: {n_kept:,} ({100*n_kept/N:.1f}%)")
    print(f"  - Removed: {n_removed:,} ({100*n_removed/N:.1f}%)")

    return whitelist


def sh_dc_to_rgb(colors_dc: np.ndarray) -> np.ndarray:
    """Convert SH DC coefficients to RGB values."""
    C0 = 0.28209479177387814
    rgb = colors_dc * C0 + 0.5
    return np.clip(rgb, 0, 1)


def stage2_color_validation(
    positions: np.ndarray,
    colors_dc: np.ndarray,
    whitelist: np.ndarray,
    cameras: List[Camera],
    mask_folder: str,
    image_folder: str,
    color_threshold: float = 0.40,
    num_samples: int = 10,
    device: str = 'cuda',
) -> np.ndarray:
    """Stage 2: Color Validation - Remove Gaussians with wrong colors."""
    print(f"\n[STAGE 2] Color Validation (threshold={color_threshold})")

    if image_folder is None:
        print("  - Skipping (no image folder provided)")
        return whitelist.copy()

    image_folder = Path(image_folder)
    mask_folder = Path(mask_folder)

    whitelist_idx = np.where(whitelist)[0]
    n_whitelist = len(whitelist_idx)

    if n_whitelist == 0:
        print("  - Skipping (no whitelisted Gaussians)")
        return whitelist.copy()

    gaussian_rgb = sh_dc_to_rgb(colors_dc[whitelist_idx])
    sample_cams = cameras[:min(num_samples, len(cameras))]
    print(f"  - Validating {n_whitelist:,} Gaussians against {len(sample_cams)} views")

    was_front_layer = np.zeros(n_whitelist, dtype=bool)
    has_color_match = np.zeros(n_whitelist, dtype=bool)
    pos_whitelist = positions[whitelist_idx]

    for cam in tqdm(sample_cams, desc="  Color validation", leave=False):
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = image_folder / (Path(cam.image_name).stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        img = np.array(Image.open(img_path).convert('RGB')) / 255.0
        H, W = img.shape[:2]

        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = mask_folder / (Path(cam.image_name).stem + ext)
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            continue

        mask_img = Image.open(mask_path).convert('L')
        if mask_img.size != (W, H):
            mask_img = mask_img.resize((W, H), Image.NEAREST)
        mask = np.array(mask_img) > 127

        p_cam = (cam.R_w2c @ pos_whitelist.T).T + cam.t_w2c
        depths = p_cam[:, 2]
        valid_depth = depths > 0.01

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = (p_cam[:, 0] * cam.fx / depths + cam.cx).astype(np.int32)
            v = (p_cam[:, 1] * cam.fy / depths + cam.cy).astype(np.int32)

        valid_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid = valid_depth & valid_bounds

        if not valid.any():
            continue

        u_valid = u[valid]
        v_valid = v[valid]
        d_valid = depths[valid]
        valid_idx = np.where(valid)[0]

        mask_at_proj = mask[v_valid, u_valid]
        in_mask = mask_at_proj

        if not in_mask.any():
            continue

        pixel_keys = v_valid * W + u_valid
        in_mask_idx = np.where(in_mask)[0]
        pixel_keys_masked = pixel_keys[in_mask_idx]
        depths_masked = d_valid[in_mask_idx]
        gauss_idx_masked = valid_idx[in_mask_idx]

        unique_pixels, inverse = np.unique(pixel_keys_masked, return_inverse=True)

        for pix_i, pix_key in enumerate(unique_pixels):
            at_pixel = inverse == pix_i
            if not at_pixel.any():
                continue

            depths_at_pixel = depths_masked[at_pixel]
            gauss_at_pixel = gauss_idx_masked[at_pixel]
            front_idx = gauss_at_pixel[np.argmin(depths_at_pixel)]

            was_front_layer[front_idx] = True

            pix_v = pix_key // W
            pix_u = pix_key % W
            img_color = img[pix_v, pix_u]
            gauss_color = gaussian_rgb[front_idx]
            diff = np.sqrt(np.sum((img_color - gauss_color) ** 2))

            if diff < color_threshold:
                has_color_match[front_idx] = True

    keep_color = (~was_front_layer) | has_color_match

    n_front_layer = np.sum(was_front_layer)
    n_matched = np.sum(has_color_match)
    n_removed = np.sum(~keep_color)

    print(f"  - Front-layer Gaussians: {n_front_layer:,}")
    print(f"  - Color matched (δ < {color_threshold}): {n_matched:,}")
    print(f"  - Removed: {n_removed:,} ({100*n_removed/n_whitelist:.1f}%)")

    result = whitelist.copy()
    result[whitelist_idx[~keep_color]] = False

    return result


def stage3_outlier_removal(
    positions: np.ndarray,
    keep_mask: np.ndarray,
    k_neighbors: int = 10,
    percentile: float = 95.0,
    spatial_percentile: float = 99.0,
) -> np.ndarray:
    """Stage 3: Outlier Removal - Remove isolated floaters."""
    print(f"\n[STAGE 3] Neighbor-based Outlier Removal")

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("  - Skipping (scipy not available)")
        return keep_mask

    kept_positions = positions[keep_mask]
    n_kept = len(kept_positions)

    if n_kept < k_neighbors + 1:
        print("  - Skipping (too few points)")
        return keep_mask

    print(f"  - Analyzing {n_kept:,} points with k={k_neighbors}")

    tree = cKDTree(kept_positions)
    distances, _ = tree.query(kept_positions, k=k_neighbors + 1)
    mean_distances = distances[:, 1:].mean(axis=1)

    threshold = np.percentile(mean_distances, percentile)
    is_inlier = mean_distances <= threshold

    center = kept_positions.mean(axis=0)
    dist_to_center = np.linalg.norm(kept_positions - center, axis=1)
    spatial_threshold = np.percentile(dist_to_center, spatial_percentile)
    is_spatial_inlier = dist_to_center <= spatial_threshold

    final_inlier = is_inlier & is_spatial_inlier

    kept_indices = np.where(keep_mask)[0]
    outlier_indices = kept_indices[~final_inlier]

    new_mask = keep_mask.copy()
    new_mask[outlier_indices] = False

    n_removed = (~final_inlier).sum()
    print(f"  - Removed {n_removed:,} outliers")
    print(f"  - Neighbor threshold: {threshold:.4f}")
    print(f"  - Spatial threshold: {spatial_threshold:.4f}")

    return new_mask


def main():
    parser = argparse.ArgumentParser(
        description="Clean-GS style Gaussian mask filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--ply', required=True, help='Input PLY file')
    parser.add_argument('--cameras', required=True, help='cameras.json file OR COLMAP sparse folder')
    parser.add_argument('--masks', required=True, help='Mask folder (PNG, 0=background, 255=object)')
    parser.add_argument('--images', default=None, help='Image folder (optional, for color validation)')
    parser.add_argument('--output', required=True, help='Output filtered PLY')

    # Checkpoint filtering for mesh extraction compatibility
    parser.add_argument('--checkpoint', default=None, help='Input checkpoint (.pth) to filter')
    parser.add_argument('--checkpoint-output', default=None, help='Output filtered checkpoint')

    parser.add_argument('--num-views', type=int, default=30, help='Views to sample (default: 30)')
    parser.add_argument('--min-views', type=int, default=2, help='Min views for whitelist (default: 2)')
    parser.add_argument('--color-threshold', type=float, default=0.40, help='Color match threshold (default: 0.40)')
    parser.add_argument('--k-neighbors', type=int, default=10, help='K neighbors for outlier detection (default: 10)')
    parser.add_argument('--neighbor-percentile', type=float, default=95.0, help='Neighbor distance percentile (default: 95)')
    parser.add_argument('--spatial-percentile', type=float, default=99.0, help='Spatial distance percentile (default: 99)')

    parser.add_argument('--skip-outlier-removal', action='store_true', help='Skip Stage 3')
    parser.add_argument('--skip-color-validation', action='store_true', help='Skip Stage 2')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--dry-run', action='store_true', help='Preview without saving')

    # Camera filtering
    parser.add_argument('--min-mask-coverage', type=float, default=5.0,
                        help='Min mask coverage %% to keep camera (default: 5.0). '
                             'Cameras with less coverage (mostly sky) are excluded.')
    parser.add_argument('--cameras-output', default=None,
                        help='Output filtered cameras.json (auto-generated if not specified)')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        args.device = 'cpu'

    print("=" * 60)
    print("Clean-GS Style Gaussian Mask Filter")
    print("=" * 60)

    # Load data
    positions, colors_dc, plydata = load_ply(args.ply)

    # Auto-detect cameras format
    cameras_path = Path(args.cameras)
    is_json = cameras_path.suffix == '.json'

    if is_json:
        cameras = load_cameras_json(args.cameras)
    elif cameras_path.is_dir():
        cameras = load_cameras_colmap(args.cameras)
    else:
        try:
            cameras = load_cameras_json(args.cameras)
            is_json = True
        except:
            cameras = load_cameras_colmap(args.cameras)

    # Filter low-coverage cameras (prevents mesh extraction failures)
    excluded_cameras = []
    if args.min_mask_coverage > 0:
        print(f"\n[CAMERA FILTER] Excluding cameras with <{args.min_mask_coverage}% mask coverage")
        cameras, excluded_cameras = filter_low_coverage_cameras(
            cameras, args.masks, min_coverage=args.min_mask_coverage
        )
        print(f"  - Kept: {len(cameras)} cameras")
        print(f"  - Excluded: {len(excluded_cameras)} cameras (mostly sky/background)")

        if len(excluded_cameras) > 0 and len(excluded_cameras) <= 10:
            for cam in excluded_cameras:
                print(f"    - {cam.image_name}")
        elif len(excluded_cameras) > 10:
            for cam in excluded_cameras[:5]:
                print(f"    - {cam.image_name}")
            print(f"    ... and {len(excluded_cameras) - 5} more")

    N = len(positions)

    # Stage 1: Whitelist Filtering
    whitelist = stage1_whitelist_filtering(
        positions, cameras, args.masks,
        num_samples=args.num_views,
        min_views=args.min_views,
        device=args.device,
    )

    # Stage 2: Color Validation
    if args.skip_color_validation:
        print("\n[STAGE 2] Color Validation - SKIPPED")
        keep_mask = whitelist.copy()
    else:
        keep_mask = stage2_color_validation(
            positions, colors_dc, whitelist, cameras,
            args.masks, args.images,
            color_threshold=args.color_threshold,
            num_samples=args.num_views,
            device=args.device,
        )

    # Stage 3: Outlier Removal
    if not args.skip_outlier_removal:
        keep_mask = stage3_outlier_removal(
            positions, keep_mask,
            k_neighbors=args.k_neighbors,
            percentile=args.neighbor_percentile,
            spatial_percentile=args.spatial_percentile,
        )

    # Summary
    n_final = keep_mask.sum()
    compression = 100 * (N - n_final) / N

    print("\n" + "=" * 60)
    print("[RESULTS]")
    print(f"  Original:   {N:,} Gaussians")
    print(f"  Final:      {n_final:,} Gaussians")
    print(f"  Removed:    {N - n_final:,} ({compression:.1f}% compression)")
    print("=" * 60)

    # Save
    if args.dry_run:
        print("\n[DRY RUN] No files saved")
    else:
        save_filtered_ply(plydata, keep_mask, args.output)
        print(f"\n[DONE] Saved PLY to: {args.output}")

        # Filter checkpoint if provided
        if args.checkpoint:
            if args.checkpoint_output is None:
                # Auto-generate output name
                ckpt_path = Path(args.checkpoint)
                args.checkpoint_output = str(ckpt_path.parent / f"{ckpt_path.stem}_filtered{ckpt_path.suffix}")

            filter_checkpoint(args.checkpoint, keep_mask, args.checkpoint_output)
            print(f"[DONE] Saved checkpoint to: {args.checkpoint_output}")

        # Save filtered cameras.json if cameras were excluded
        if is_json and args.min_mask_coverage > 0 and len(excluded_cameras) > 0:
            if args.cameras_output is None:
                # Auto-generate: cameras.json -> cameras_filtered.json
                cam_path = Path(args.cameras)
                args.cameras_output = str(cam_path.parent / f"{cam_path.stem}_filtered.json")

            save_filtered_cameras_json(cameras, args.cameras, args.cameras_output)
            print(f"[DONE] Saved filtered cameras to: {args.cameras_output}")
            print(f"       Use this for mesh extraction to avoid empty-geometry views")


if __name__ == '__main__':
    main()
