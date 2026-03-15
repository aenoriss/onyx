"""Difix3D+ integration for nerfstudio/splatfacto training.

Provides render-enhance-train fix cycles using NVIDIA Difix3D+ (SD-Turbo based).
Loaded only when --difix3d is passed to gsplat's train.py.

Two-phase approach (matching the Difix3D+ baseline):

Phase 1 — Fix cycles (~28 cycles, NVIDIA-matched schedule):
  1. Progressive shift: generate novel cameras that march from training
     poses toward inter-frame midpoints (360) or random SLERP (non-360).
  2. Render the current scene at novel poses (no_grad).
  3. Run Difix to produce enhanced pseudo-GT images.
  4. Compute binary alpha masks (NVIDIA approach).
  5. Store (camera, enhanced_image_cpu, alpha_mask_cpu) in the novel_data_pool.

Phase 2 — Novel data reuse (every iteration after first fix cycle):
  With probability novel_lambda (~0.40), sample a random entry from
  novel_data_pool and add extra gradient from that enhanced image,
  with binary alpha masking on transparent regions.

Key alignment with NVIDIA official implementation:
  - Schedule: ~28 cycles from 5% to 98%, every ~2K steps (NVIDIA-matched)
  - Progressive shift: NVIDIA approach — novel views march from training poses
    toward target poses across cycles (compound refinement)
  - 360-aware: groups cameras by frame (16 tiles), targets inter-frame midpoints
  - Ref selection: direction (60%) + position (40%) scoring (360-safe)
  - Alpha masking: binary threshold at 0.5 during loss (NVIDIA approach)
  - Loss: L1 + SSIM at 0.3x weight (matches NVIDIA novel_data_lambda=0.3)
  - Enhancement: 512px, single-step, tau=200 (NVIDIA default)
  - No immediate gradient steps during fix cycle (NVIDIA approach)
"""

import gc
import os
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

_DIFIX_PROMPT = "remove degradation"
# Default tau=200 (NVIDIA official default, timestep=199).
# Configurable via --difix3d-tau CLI arg. Higher tau (e.g. 400) gives stronger
# correction for scenes with severe mid-training artifacts.
_DIFIX_TAU_DEFAULT = 200
# Maximum entries kept in the novel data pool (oldest evicted beyond this limit).
_MAX_POOL_SIZE = 200


# ── Schedule ──────────────────────────────────────────────────────────────────

def compute_difix_schedule(
    total_iters: int,
    views_per_cycle: int = 36,
    novel_lambda: float = 0.40,
) -> List[int]:
    """Compute fix cycle iterations matching NVIDIA's official schedule.

    NVIDIA's approach (60K baseline):
      - First fix at step 3K (5% of training)
      - Then every ~2K steps until step 60K
      - 29 total cycles for progressive compound refinement

    Each cycle builds on the previous: better model → better renders →
    better DiFix output → better training signal. Starting early and
    running frequently is critical for this feedback loop.

    Adapts to any total_iters by scaling the interval proportionally:
      interval = total_iters / 30  (~2K for 60K, ~1K for 30K)
      start = total_iters * 0.05   (5% of training)

    Returns sorted list of iteration numbers for fix cycles.
    """
    # Start after nerfstudio's resolution schedule completes (default: 6K steps).
    # num_downscales=2, resolution_schedule=3000 → full res at step 6000.
    # Starting earlier produces low-res renders → garbage DiFix → pool pollution.
    start_iter = max(int(total_iters * 0.12), 7000)  # ~12%, after resolution ramp
    end_iter = total_iters

    # Match NVIDIA's ~2K interval for 60K baseline, scale proportionally
    interval = max(int(total_iters / 30), 500)  # ~2K for 60K, floor 500
    schedule = list(range(start_iter, end_iter + 1, interval))

    # Ensure we don't schedule during the very last few steps
    schedule = [s for s in schedule if s <= int(total_iters * 0.98)]

    num_cycles = len(schedule)
    reuse_window = total_iters - start_iter
    reuse_steps = novel_lambda * reuse_window
    actual_synth = num_cycles * views_per_cycle + reuse_steps
    actual_pct = actual_synth / (total_iters + actual_synth) * 100
    print(f"[DIFIX] Schedule: {num_cycles} cycles × {views_per_cycle} views, "
          f"interval={interval} steps, start={start_iter}")
    print(f"[DIFIX] + {reuse_steps:.0f} reuse steps = {actual_pct:.1f}% synthetic")

    return schedule


# ── Difix model ───────────────────────────────────────────────────────────────

def init_difix(device: str = "cpu"):
    """Load DifixPipeline from HuggingFace (nvidia/difix_ref).

    Kept on CPU by default — moved to GPU only during fix cycles.
    """
    difix_src = "/workspace/Difix3D/src"
    if difix_src not in sys.path:
        sys.path.insert(0, difix_src)

    from pipeline_difix import DifixPipeline  # noqa: PLC0415

    print("[DIFIX] Loading nvidia/difix_ref (fp16) …")
    pipe = DifixPipeline.from_pretrained(
        "nvidia/difix_ref", trust_remote_code=True, torch_dtype=torch.float16,
    )
    pipe.set_progress_bar_config(disable=True)
    print("[DIFIX] DifixPipeline ready (CPU, moved to GPU on demand).")
    return pipe


# ── Image conversion helpers ──────────────────────────────────────────────────

def _tensor_to_pil(t: torch.Tensor) -> PILImage.Image:
    """Float [H, W, 3] or [3, H, W] tensor in [0, 1] → RGB PIL Image."""
    if t.dim() == 3 and t.shape[0] == 3:
        t = t.permute(1, 2, 0)
    arr = (t.clamp(0.0, 1.0) * 255).byte().cpu().numpy()
    return PILImage.fromarray(arr, mode="RGB")


def _pil_to_tensor(img: PILImage.Image, device: str = "cuda") -> torch.Tensor:
    """RGB PIL Image → float [H, W, 3] tensor in [0, 1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).to(device)


def _apply_dataparser_transform(raw_c2ws, train_c2ws):
    """Transform raw COLMAP c2w matrices into nerfstudio's coordinate space.

    Nerfstudio applies a 3x4 transform matrix + scale to all cameras during
    data loading (saved in dataparser_transforms.json). Eval cameras parsed
    from COLMAP are in raw space and need the same transform.

    Strategy: find the transform file from the nerfstudio output directory.
    Falls back to returning raw poses if transform can't be found.
    """
    import json
    import glob as _glob

    # Find dataparser_transforms.json (saved by nerfstudio during training)
    transform_files = _glob.glob(
        "/data/output/splatfacto/unnamed/splatfacto/*/dataparser_transforms.json")
    if not transform_files:
        transform_files = _glob.glob(
            "/data/output/splatfacto/*/*/dataparser_transforms.json")
    if not transform_files:
        print("[DIFIX] WARNING: No dataparser_transforms.json found, "
              "eval cameras may be in wrong coordinate space")
        return raw_c2ws

    with open(sorted(transform_files)[-1]) as f:
        transforms = json.load(f)

    transform_matrix = torch.tensor(transforms["transform"], dtype=torch.float32)  # [3, 4]
    scale = transforms["scale"]

    # Build 4x4 transform
    T = torch.eye(4, dtype=torch.float32)
    T[:3, :] = transform_matrix

    # Apply to each c2w:
    # 1. Flip Y and Z columns (COLMAP OpenCV → nerfstudio OpenGL convention)
    # 2. Apply dataparser transform (computed on already-flipped poses)
    # 3. Scale translation
    transformed = []
    for c2w in raw_c2ws:
        # COLMAP → OpenGL: flip Y and Z axes of the rotation
        c2w = c2w.clone()
        c2w[:3, 1:3] *= -1
        # Pad c2w [3,4] to [4,4]
        c2w_4x4 = torch.eye(4, dtype=torch.float32)
        c2w_4x4[:3, :] = c2w

        # Apply transform
        new_c2w = T @ c2w_4x4
        new_c2w[:3, 3] *= scale  # scale translation
        transformed.append(new_c2w[:3, :])  # back to [3,4]

    result = torch.stack(transformed)
    print(f"[DIFIX] Applied dataparser transform (scale={scale:.4f}) to {len(result)} eval cameras")
    return result


def _slerp_c2w(c2w_a: torch.Tensor, c2w_b: torch.Tensor, t: float) -> np.ndarray:
    """Interpolate two [3,4] camera-to-world matrices: SLERP rotation + LERP translation."""
    from scipy.spatial.transform import Rotation, Slerp  # noqa: PLC0415

    R_a = c2w_a[:3, :3].cpu().float().numpy()
    R_b = c2w_b[:3, :3].cpu().float().numpy()
    rots = Rotation.from_matrix(np.stack([R_a, R_b]))
    slerp = Slerp([0.0, 1.0], rots)
    R_interp = slerp([float(t)]).as_matrix()[0]

    t_a = c2w_a[:3, 3].cpu().float().numpy()
    t_b = c2w_b[:3, 3].cpu().float().numpy()
    t_interp = (1.0 - t) * t_a + t * t_b

    c2w = np.zeros((3, 4), dtype=np.float32)
    c2w[:3, :3] = R_interp
    c2w[:3, 3] = t_interp
    return c2w


# ── Progressive Shifter (NVIDIA approach, 360-aware) ─────────────────────────

class ProgressiveShifter:
    """NVIDIA-style progressive pose shifting for novel view generation.

    For 360 scenes (tiles_per_frame > 1):
      - Groups cameras by frame (16 tiles each)
      - Computes target poses at midpoints between consecutive frames
        (these are truly unseen viewpoints — no camera ever observes them)
      - Each fix cycle shifts current poses closer to midpoint targets
      - Over ~28 cycles, novel views progressively fill inter-frame gaps

    For regular scenes (tiles_per_frame == 1):
      - Uses all training cameras as starting poses
      - Computes targets at midpoints between adjacent training cameras
      - Same progressive shift mechanism

    Mirrors NVIDIA's CameraPoseInterpolator.shift_poses():
      - Each cycle: current_pose += distance * (target - current) / ||target - current||
      - distance=0.5 scene units per cycle (NVIDIA default)
    """

    def __init__(self, train_cameras, tiles_per_frame: int = 16,
                 eval_poses_path: str = ""):
        from nerfstudio.cameras.cameras import Cameras  # noqa: PLC0415

        c2ws = train_cameras.camera_to_worlds  # [N, 3, 4]
        n_cams = len(train_cameras)
        self.train_cameras = train_cameras
        self.tiles_per_frame = tiles_per_frame

        # ── Try loading real eval camera poses (held-out frames from ingest) ──
        # These are camera poses from frames that were discarded during ingest,
        # kept through SfM for poses, then separated before training.
        # They represent truly unseen viewpoints — ideal shift targets.
        if eval_poses_path and os.path.exists(eval_poses_path):
            eval_data = torch.load(eval_poses_path, map_location="cpu")
            raw_c2ws = eval_data["c2ws"]  # [E, 3, 4] in raw COLMAP space

            # Apply nerfstudio's dataparser transform to bring eval cameras
            # into the same coordinate space as training cameras.
            # The transform (3x4 matrix + scale) is applied by nerfstudio when
            # loading COLMAP data. Without this, eval cameras point in wrong
            # directions at wrong positions.
            self.targets = _apply_dataparser_transform(
                raw_c2ws, train_cameras.camera_to_worlds)
            # Store eval intrinsics for camera construction
            self._eval_fx = eval_data["fx"]
            self._eval_fy = eval_data["fy"]
            self._eval_cx = eval_data["cx"]
            self._eval_cy = eval_data["cy"]
            self._eval_width = eval_data["width"]
            self._eval_height = eval_data["height"]
            self._has_eval_intrinsics = True
            self.current_poses = None
            print(f"[DIFIX] ProgressiveShifter: {len(self.targets)} eval camera targets "
                  f"(real held-out frames)")
            return

        self._has_eval_intrinsics = False

        # ── Default: synthetic midpoint targets (no holdout needed) ──
        # Groups cameras by spatial proximity to find frame clusters,
        # then computes midpoints between consecutive clusters.
        # Works for both 360 rigs (16 tiles/frame) and regular cameras.
        centres = c2ws[:, :3, 3]  # [N, 3]

        if tiles_per_frame > 1 and n_cams >= tiles_per_frame * 2:
            # 360 rig: cluster cameras by position to find frame groups.
            # Cameras from the same frame are at nearly identical positions.
            # Use agglomerative clustering instead of assuming contiguous indices.
            from scipy.cluster.hierarchy import fcluster, linkage

            pos_np = centres.cpu().float().numpy()
            Z = linkage(pos_np, method="average")
            # Cut at a distance that separates frame positions
            # (tiles within a frame are ~0 apart, frames are far apart)
            dists_sorted = sorted(Z[:, 2])
            # Find the gap between intra-frame and inter-frame distances
            gap_idx = len(dists_sorted) // 2
            cut_dist = (dists_sorted[gap_idx] + dists_sorted[min(gap_idx + 1, len(dists_sorted) - 1)]) / 2
            labels = fcluster(Z, t=cut_dist, criterion="distance")

            # Compute cluster centroids (frame positions)
            unique_labels = sorted(set(labels))
            frame_positions = []
            frame_c2ws = []
            for lab in unique_labels:
                mask = labels == lab
                idxs = [i for i, m in enumerate(mask) if m]
                centroid_pos = centres[idxs].mean(dim=0)
                frame_positions.append(centroid_pos)
                # Use the camera closest to centroid as the frame's representative c2w
                dists_to_center = torch.norm(centres[idxs] - centroid_pos, dim=1)
                best = idxs[dists_to_center.argmin().item()]
                frame_c2ws.append(c2ws[best])

            n_frames = len(frame_positions)
            if n_frames < 2:
                self.targets = None
                print("[DIFIX] ProgressiveShifter: too few frame clusters, "
                      "will fall back to SLERP")
                return

            # Sort frames along principal movement axis for consistent ordering
            frame_pos_tensor = torch.stack(frame_positions)
            principal = torch.pca_lowrank(
                frame_pos_tensor - frame_pos_tensor.mean(0), q=1)[0].squeeze()
            order = torch.argsort(principal)

            # Midpoints between consecutive frames (+ wraparound for 360)
            targets = []
            for k in range(len(order)):
                i = order[k].item()
                j = order[(k + 1) % len(order)].item()
                mid = _slerp_c2w(frame_c2ws[i], frame_c2ws[j], 0.5)
                targets.append(torch.tensor(mid, dtype=torch.float32))

            self.targets = torch.stack(targets)
            print(f"[DIFIX] ProgressiveShifter: {n_frames} frame clusters detected, "
                  f"{len(targets)} midpoint targets (synthetic, 360-aware)")
        else:
            # Regular cameras: midpoints between adjacent cameras sorted by position
            principal = torch.pca_lowrank(
                centres - centres.mean(0), q=1)[0].squeeze()
            order = torch.argsort(principal)

            targets = []
            for k in range(len(order) - 1):
                i, j = order[k].item(), order[k + 1].item()
                mid = _slerp_c2w(c2ws[i], c2ws[j], 0.5)
                targets.append(torch.tensor(mid, dtype=torch.float32))

            self.targets = torch.stack(targets) if targets else None
            if self.targets is not None:
                print(f"[DIFIX] ProgressiveShifter: {n_cams} cameras, "
                      f"{len(targets)} midpoint targets (synthetic)")

        # Current poses start at nearest training cameras to each target
        self.current_poses = None  # lazy init on first cycle

    def _init_current_poses(self):
        """Initialize current poses as nearest training cameras to each target."""
        c2ws = self.train_cameras.camera_to_worlds
        centres = c2ws[:, :3, 3]

        current = []
        for target in self.targets:
            target_pos = target[:3, 3]
            dists = torch.norm(centres - target_pos, dim=1)
            nearest = dists.argmin().item()
            current.append(c2ws[nearest].clone())
        self.current_poses = torch.stack(current)

    def generate_views(self, n_views: int, distance: float = 0.5):
        """Generate novel views by shifting current poses toward targets.

        Args:
            n_views: max views to generate this cycle
            distance: shift distance in scene units per cycle (NVIDIA default: 0.5)

        Returns list of (nerfstudio Cameras, ref_cam_idx) tuples.
        """
        from nerfstudio.cameras.cameras import Cameras  # noqa: PLC0415

        if self.targets is None:
            return None  # signal to fallback to SLERP

        if self.current_poses is None:
            self._init_current_poses()

        c2ws = self.train_cameras.camera_to_worlds
        centres = c2ws[:, :3, 3]
        forwards = -c2ws[:, :3, 2]
        forwards = F.normalize(forwards, dim=1)

        # Shift each current pose toward its target
        new_poses = []
        for current, target in zip(self.current_poses, self.targets):
            t_current = current[:3, 3]
            t_target = target[:3, 3]
            dist = torch.norm(t_target - t_current).item()
            t_ratio = min(distance / (dist + 1e-8), 1.0)
            shifted = _slerp_c2w(current, target, t_ratio)
            new_poses.append(torch.tensor(shifted, dtype=torch.float32))

        self.current_poses = torch.stack(new_poses)

        # Select up to n_views from shifted poses
        n_available = len(self.current_poses)
        if n_available > n_views:
            indices = torch.randperm(n_available)[:n_views].tolist()
        else:
            indices = list(range(n_available))

        novel_cameras = []
        for idx in indices:
            c2w = self.current_poses[idx]
            interp_pos = c2w[:3, 3]

            # Ref camera: direction + position scoring (360-safe)
            pos_dists = torch.norm(centres - interp_pos, dim=1)
            pos_score = 1.0 - pos_dists / (pos_dists.max() + 1e-8)

            interp_fwd = -c2w[:3, 2]
            interp_fwd = F.normalize(interp_fwd, dim=0)
            dir_score = (forwards * interp_fwd.unsqueeze(0)).sum(dim=1)

            ref_idx = (0.6 * dir_score + 0.4 * pos_score).argmax().item()

            # Intrinsics: use eval camera intrinsics if available, else ref camera
            if self._has_eval_intrinsics:
                from nerfstudio.cameras.cameras import Cameras as _C
                novel_cam = _C(
                    camera_to_worlds=c2w.unsqueeze(0),
                    fx=self._eval_fx[idx].unsqueeze(0),
                    fy=self._eval_fy[idx].unsqueeze(0),
                    cx=self._eval_cx[idx].unsqueeze(0),
                    cy=self._eval_cy[idx].unsqueeze(0),
                    width=self._eval_width[idx].unsqueeze(0),
                    height=self._eval_height[idx].unsqueeze(0),
                )
                novel_cameras.append((novel_cam, ref_idx))
                continue

            ref_cam = self.train_cameras[ref_idx].reshape(())
            novel_cam = Cameras(
                camera_to_worlds=c2w.unsqueeze(0),
                fx=ref_cam.fx.unsqueeze(0),
                fy=ref_cam.fy.unsqueeze(0),
                cx=ref_cam.cx.unsqueeze(0),
                cy=ref_cam.cy.unsqueeze(0),
                width=ref_cam.width.unsqueeze(0),
                height=ref_cam.height.unsqueeze(0),
            )
            novel_cameras.append((novel_cam, ref_idx))

        return novel_cameras


# ── Fallback SLERP interpolation (non-360, no progressive shift) ─────────────

def interpolate_cameras_slerp(datamanager, n_views: int = 36) -> list:
    """Generate n_views via random distant-pair SLERP (CuriGS strategy).

    Fallback when ProgressiveShifter returns None (too few cameras/frames).
    Returns list of (nerfstudio Cameras, ref_cam_idx) tuples.
    """
    from nerfstudio.cameras.cameras import Cameras  # noqa: PLC0415

    train_cameras = datamanager.train_dataset.cameras
    n_cams = len(train_cameras)
    if n_cams < 2:
        return []

    c2ws = train_cameras.camera_to_worlds
    centres = c2ws[:, :3, 3]
    forwards = -c2ws[:, :3, 2]
    forwards = F.normalize(forwards, dim=1)

    novel_cameras = []
    n_sample = min(n_views, n_cams)
    perm = torch.randperm(n_cams).tolist()
    top_k = min(3, n_cams - 1)

    for i in range(n_sample):
        a_idx = perm[i % len(perm)]
        dists = torch.norm(centres - centres[a_idx], dim=1)
        dists[a_idx] = -1.0
        _, topk_idx = torch.topk(dists, top_k)
        b_idx = topk_idx[torch.randint(top_k, (1,)).item()].item()
        t = random.uniform(0.15, 0.85)

        c2w_interp = _slerp_c2w(c2ws[a_idx], c2ws[b_idx], t)
        c2w_t = torch.tensor(c2w_interp, dtype=torch.float32)

        # Ref camera: direction + position scoring
        interp_pos = c2w_t[:3, 3]
        pos_dists = torch.norm(centres - interp_pos, dim=1)
        pos_score = 1.0 - pos_dists / (pos_dists.max() + 1e-8)
        interp_fwd = (1.0 - t) * forwards[a_idx] + t * forwards[b_idx]
        interp_fwd = F.normalize(interp_fwd, dim=0)
        dir_score = (forwards * interp_fwd.unsqueeze(0)).sum(dim=1)
        ref_idx = (0.6 * dir_score + 0.4 * pos_score).argmax().item()

        cam_a = train_cameras[a_idx].reshape(())
        novel_cam = Cameras(
            camera_to_worlds=c2w_t.unsqueeze(0),
            fx=cam_a.fx.unsqueeze(0),
            fy=cam_a.fy.unsqueeze(0),
            cx=cam_a.cx.unsqueeze(0),
            cy=cam_a.cy.unsqueeze(0),
            width=cam_a.width.unsqueeze(0),
            height=cam_a.height.unsqueeze(0),
        )
        novel_cameras.append((novel_cam, ref_idx))

    return novel_cameras


# ── Fix step (Phase 1) ───────────────────────────────────────────────────────

def difix_fix_step(
    model,
    datamanager,
    difix_pipe,
    novel_pool: List[Tuple],
    shifter: Optional["ProgressiveShifter"],
    n_views: int = 36,
    difix_tau: int = _DIFIX_TAU_DEFAULT,
    step: int = 0,
) -> None:
    """Run one Difix3D+ fix cycle within splatfacto training.

    Uses ProgressiveShifter for novel camera generation (NVIDIA approach).
    Falls back to random SLERP if shifter is unavailable.
    """
    difix_timesteps = [difix_tau - 1]
    print(f"[DIFIX] Fix cycle @ step {step} — {n_views} novel views, tau={difix_tau}")

    # Free VRAM before loading Difix
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"[DIFIX] VRAM free before Difix load: {free_gb:.1f} GB")

    # Enable memory-efficient attention
    try:
        difix_pipe.enable_attention_slicing(1)
    except Exception:
        pass
    try:
        difix_pipe.enable_vae_slicing()
    except Exception:
        pass

    # Enforce fp16 during GPU transfer
    difix_pipe.to(device="cuda", dtype=torch.float16)
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"[DIFIX] Pipeline on GPU (fp16), VRAM free: {free_gb:.1f} GB")

    # Generate novel cameras (progressive shift or SLERP fallback)
    novel_cameras = None
    if shifter is not None:
        novel_cameras = shifter.generate_views(n_views)
    if novel_cameras is None:
        novel_cameras = interpolate_cameras_slerp(datamanager, n_views)
    if not novel_cameras:
        difix_pipe.to("cpu")
        torch.cuda.empty_cache()
        return

    # Flush pool — only use latest cycle's data (NVIDIA approach).
    # Older enhanced images were rendered from a worse model and are stale.
    novel_pool.clear()

    _DIFIX_SIZE = 512
    device = model.device
    generated = 0

    for cam_idx, (novel_cam, ref_idx) in enumerate(novel_cameras):
        novel_cam_device = novel_cam.to(device)
        with torch.no_grad():
            outputs = model.get_outputs(novel_cam_device)
            rendered = outputs["rgb"]  # [H, W, 3]
            render_h, render_w = rendered.shape[0], rendered.shape[1]
            base_pil = _tensor_to_pil(rendered)

            del outputs, rendered

        torch.cuda.empty_cache()

        # Downscale to 512px for Difix (SD-Turbo native resolution)
        base_small = base_pil.resize((_DIFIX_SIZE, _DIFIX_SIZE), PILImage.LANCZOS)

        # Get reference image (nearest by direction+position scoring)
        ref_data = datamanager.train_dataset[ref_idx]
        ref_img = ref_data["image"]
        ref_pil = _tensor_to_pil(ref_img)
        ref_small = ref_pil.resize((_DIFIX_SIZE, _DIFIX_SIZE), PILImage.LANCZOS)

        # Difix enhancement
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            try:
                enhanced_small = difix_pipe(
                    _DIFIX_PROMPT,
                    image=base_small,
                    ref_image=ref_small,
                    num_inference_steps=1,
                    timesteps=difix_timesteps,
                    guidance_scale=0.0,
                ).images[0]
            except Exception as exc:
                print(f"[DIFIX] Enhancement failed for view {cam_idx}: {exc}")
                continue
            enhanced_pil = enhanced_small.resize((render_w, render_h), PILImage.LANCZOS)
            enhanced_tensor = _pil_to_tensor(enhanced_pil, device="cpu")

        # Save debug images (first 4 views per cycle)
        if cam_idx < 4:
            debug_dir = f"/data/output/splatfacto/difix_debug/step_{step}"
            os.makedirs(debug_dir, exist_ok=True)
            base_pil.save(f"{debug_dir}/view_{cam_idx:02d}_render.jpg")
            ref_pil.save(f"{debug_dir}/view_{cam_idx:02d}_ref.jpg")
            enhanced_pil.save(f"{debug_dir}/view_{cam_idx:02d}_enhanced.jpg")
            if cam_idx == 0:
                print(f"[DIFIX] Debug images → {debug_dir}")

        # Store in pool (flushed at start of each cycle — only latest data used)
        novel_pool.append((novel_cam, enhanced_tensor))
        generated += 1

    # Move Difix back to CPU
    difix_pipe.to("cpu")
    torch.cuda.empty_cache()
    print(f"[DIFIX] Fix cycle done. Generated: {generated} | Pool size: {len(novel_pool)}")


# ── Novel data reuse (Phase 2) ───────────────────────────────────────────────

_novel_step_count = 0


def difix_novel_step(
    model,
    novel_pool: List[Tuple],
    novel_weight: float = 0.3,
    lambda_dssim: float = 0.2,
) -> None:
    """One extra gradient step using a random entry from the novel data pool.

    Called probabilistically (novel_lambda ≈ 0.40) every training iteration
    after the first fix cycle. Accumulates gradients — the main training loop's
    optimizer.step() applies them together with the normal training gradients.

    Uses stored alpha masks from Phase 1 to zero out transparent regions
    (NVIDIA approach: colors * (alpha > 0.5) before loss computation).
    """
    global _novel_step_count
    _novel_step_count += 1

    idx = random.randint(0, len(novel_pool) - 1)
    novel_cam, enhanced_cpu = novel_pool[idx]

    device = model.device
    novel_cam_device = novel_cam.to(device)
    enhanced = enhanced_cpu.to(device)

    outputs = model.get_outputs(novel_cam_device)
    rendered = outputs["rgb"]  # [H, W, 3]

    # Resize enhanced/alpha to match render resolution (nerfstudio may
    # downscale during training, and eval cameras have original resolution)
    if enhanced.shape[:2] != rendered.shape[:2]:
        enhanced = F.interpolate(
            enhanced.permute(2, 0, 1).unsqueeze(0),
            size=rendered.shape[:2], mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        if alpha_mask_cpu is not None:
            alpha_mask_cpu = F.interpolate(
                alpha_mask_cpu.unsqueeze(0).unsqueeze(0),
                size=rendered.shape[:2], mode="nearest",
            ).squeeze(0).squeeze(0).cpu()

    # Binary alpha masking (NVIDIA approach) — always use FRESH alpha from
    # current render, not stale stored mask. Coverage changes as training progresses.
    alpha = outputs.get("accumulation")
    alpha_mask = (alpha.squeeze(-1) > 0.5).float() if alpha is not None else None

    if alpha_mask is not None:
        rendered = rendered * alpha_mask.unsqueeze(-1)
        enhanced = enhanced * alpha_mask.unsqueeze(-1)

    # L1 + SSIM loss (same formula as splatfacto main loss)
    l1 = torch.abs(rendered - enhanced).mean()

    try:
        from fused_ssim import fused_ssim  # noqa: PLC0415
        ssim_val = fused_ssim(
            rendered.permute(2, 0, 1).unsqueeze(0),
            enhanced.permute(2, 0, 1).unsqueeze(0),
        )
    except ImportError:
        from torchmetrics.functional import structural_similarity_index_measure as ssim_fn  # noqa
        ssim_val = ssim_fn(
            rendered.permute(2, 0, 1).unsqueeze(0),
            enhanced.permute(2, 0, 1).unsqueeze(0),
        )

    loss = novel_weight * ((1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_val))
    loss.backward()

    if _novel_step_count % 500 == 0:
        print(f"[DIFIX] Novel reuse: {_novel_step_count} steps "
              f"(pool={len(novel_pool)}, loss={loss.item():.4f})")
