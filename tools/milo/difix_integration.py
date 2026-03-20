"""Difix3D+ integration for MILo Mesh-In-the-Loop training.

Provides render-enhance-train fix cycles using NVIDIA Difix3D+ (SD-Turbo based).
Loaded only when --difix3d is passed to MILo's train.py.

Two-phase approach (matching the Difix3D+ baseline):

Phase 1 — Fix cycles (NVIDIA-matched schedule, starts at ~12%):
  1. Progressive shift: generate novel cameras that march from training
     poses toward inter-frame midpoints (NVIDIA approach).
  2. Render the current scene at novel poses (no_grad).
  3. Run Difix to produce enhanced pseudo-GT images.
  4. Perform n_grad_steps immediate gradient updates (MILo custom optimizer).
  5. Flush the novel_data_pool and refill with new cycle's data.

Phase 2 — Novel data reuse (every iteration after first fix cycle):
  With probability novel_data_lambda (~0.40), sample a random entry from
  novel_data_pool and perform one extra gradient step against that stored
  enhanced image.

Alignment with NVIDIA paper:
  - tau=200 (NVIDIA default)
  - Schedule starts at ~12% of training (early start for compound refinement)
  - ProgressiveShifter: poses march from training cameras toward midpoints
  - Pool flush: stale data discarded each cycle, only latest retained

MILo-specific (preserved):
  - Custom gaussian optimizer (step(radii > 0, n)) with immediate grad steps
  - Depth-based confidence weighting in loss
  - Mask penalty for ref camera selection
"""

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image as PILImage

_DIFIX_PROMPT = "remove degradation"
# tau=200: NVIDIA paper default. Higher values (e.g. 400) give stronger
# corrections but reduce texture coherence. Configurable via --difix3d_tau.
_DIFIX_TAU_DEFAULT = 200
# Maximum pool entries (pool is flushed each cycle, so this is a safety cap).
_MAX_POOL_SIZE = 200

# Module-level lazy-initialized ProgressiveShifter (persists across fix cycles).
_shifter: Optional["ProgressiveShifter"] = None


# ── Schedule ──────────────────────────────────────────────────────────────────

def compute_difix_schedule(
    total_iters: int,
    views_per_cycle: int = 48,
    start_pct: float = 0.12,
    end_pct: float = 0.98,
    target_synthetic_pct: float = 0.23,
    novel_lambda: float = 0.40,
) -> List[int]:
    """Compute fix cycle iterations to match a target synthetic data ratio.

    Matches the official Difix3D+ baseline (~23% synthetic training steps).
    Starts early (~12%) so the progressive shift feedback loop has time to
    compound across many cycles — key to the NVIDIA approach.

    Returns sorted list of iteration numbers for fix cycles.
    """
    start_iter = int(total_iters * start_pct)
    end_iter = int(total_iters * end_pct)
    reuse_window = total_iters - start_iter

    # Solve for num_cycles:
    #   total_synth = num_cycles * views_per_cycle + novel_lambda * reuse_window
    #   target = total_synth / (total_iters + total_synth)
    total_synth_needed = target_synthetic_pct * total_iters / (1 - target_synthetic_pct)
    reuse_steps = novel_lambda * reuse_window
    direct_steps_needed = max(total_synth_needed - reuse_steps, views_per_cycle)
    num_cycles = max(3, round(direct_steps_needed / views_per_cycle))

    if num_cycles == 1:
        return [start_iter]
    spacing = (end_iter - start_iter) / (num_cycles - 1)
    schedule = [int(start_iter + i * spacing) for i in range(num_cycles)]

    actual_synth = num_cycles * views_per_cycle + reuse_steps
    actual_pct = actual_synth / (total_iters + actual_synth) * 100
    print(f"[difix] Schedule: {num_cycles} cycles × {views_per_cycle} views "
          f"+ {reuse_steps:.0f} reuse steps = {actual_pct:.1f}% synthetic")

    return schedule


# ── Synthetic camera ──────────────────────────────────────────────────────────

@dataclass
class _SyntheticCamera:
    """Minimal camera-like object for novel-view rendering in fix cycles."""
    world_view_transform: torch.Tensor   # [4, 4]
    projection_matrix: torch.Tensor      # [4, 4]
    full_proj_transform: torch.Tensor    # [4, 4]
    camera_center: torch.Tensor          # [3]
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    uid: int = -1


# ── Camera helpers ─────────────────────────────────────────────────────────────

def _interpolate_camera(
    cam_a, cam_b, t: float = 0.5, uid: int = -1
) -> "_SyntheticCamera":
    """Interpolate between cam_a and cam_b at parameter t ∈ [0, 1].

    Rotation: spherical linear interpolation (SLERP) via scipy.
    Translation / camera centre: linear interpolation (LERP).

    In standard 3DGS / MILo the world_view_transform (WVT) is the
    world-to-camera matrix stored in column-major (transposed) form:
        WVT = W2C^T  →  WVT[:3,:3] = R^T,  WVT[3,:3] = t_view = -R @ cc
    where cc is the camera centre in world space.
    """
    try:
        from scipy.spatial.transform import Rotation, Slerp  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "[difix] scipy is required for camera interpolation. "
            "Install it with: pip install scipy"
        )

    device = cam_a.world_view_transform.device

    R_T_a = cam_a.world_view_transform[:3, :3].cpu().float().numpy()
    R_T_b = cam_b.world_view_transform[:3, :3].cpu().float().numpy()

    rots = Rotation.from_matrix(np.stack([R_T_a.T, R_T_b.T]))
    slerp = Slerp([0.0, 1.0], rots)
    R_interp = slerp([float(t)]).as_matrix()[0]   # [3, 3]
    R_T_interp = R_interp.T

    cc_a = cam_a.camera_center.cpu().float().numpy()
    cc_b = cam_b.camera_center.cpu().float().numpy()
    cc_interp = (1.0 - t) * cc_a + t * cc_b

    wvt_np = np.zeros((4, 4), dtype=np.float32)
    wvt_np[:3, :3] = R_T_interp
    wvt_np[3, :3]  = -R_interp @ cc_interp
    wvt_np[3, 3]   = 1.0

    wvt = torch.tensor(wvt_np, dtype=torch.float32, device=device)

    proj = getattr(cam_a, "projection_matrix", None)
    if proj is None:
        proj = torch.inverse(cam_a.world_view_transform) @ cam_a.full_proj_transform

    fpt = wvt.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    cc  = torch.tensor(cc_interp, dtype=torch.float32, device=device)

    return _SyntheticCamera(
        world_view_transform=wvt,
        projection_matrix=proj,
        full_proj_transform=fpt,
        camera_center=cc,
        image_height=cam_a.image_height,
        image_width=cam_a.image_width,
        FoVx=cam_a.FoVx,
        FoVy=cam_a.FoVy,
        uid=uid,
    )


def _slerp_c2w(c2w_a: torch.Tensor, c2w_b: torch.Tensor, t: float) -> np.ndarray:
    """Interpolate two [3,4] c2w matrices: SLERP rotation + LERP translation."""
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


def _camera_from_c2w(
    c2w: torch.Tensor, ref_cam, uid: int = -1
) -> "_SyntheticCamera":
    """Build _SyntheticCamera from a [3,4] cam-to-world matrix.

    Uses ref_cam for intrinsics (FoV, resolution, projection matrix).
    """
    device = ref_cam.world_view_transform.device
    R_np = c2w[:3, :3].float().numpy()   # cam-to-world rotation
    cc_np = c2w[:3, 3].float().numpy()   # camera center in world space

    # WVT = W2C in column-major form:
    #   upper-left 3×3 = R^T (world-to-cam rotation transposed)
    #   bottom row [0:3] = t_view = -R @ cc
    wvt_np = np.zeros((4, 4), dtype=np.float32)
    wvt_np[:3, :3] = R_np.T
    wvt_np[3, :3]  = -R_np @ cc_np
    wvt_np[3, 3]   = 1.0

    wvt = torch.tensor(wvt_np, dtype=torch.float32, device=device)

    proj = getattr(ref_cam, "projection_matrix", None)
    if proj is None:
        proj = torch.inverse(ref_cam.world_view_transform) @ ref_cam.full_proj_transform

    fpt = wvt.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    cc  = torch.tensor(cc_np, dtype=torch.float32, device=device)

    return _SyntheticCamera(
        world_view_transform=wvt,
        projection_matrix=proj,
        full_proj_transform=fpt,
        camera_center=cc,
        image_height=ref_cam.image_height,
        image_width=ref_cam.image_width,
        FoVx=ref_cam.FoVx,
        FoVy=ref_cam.FoVy,
        uid=uid,
    )


# ── Progressive Shifter (NVIDIA approach) ─────────────────────────────────────

class ProgressiveShifter:
    """NVIDIA-style progressive pose shifting for novel view generation.

    Poses start at training camera positions and march toward inter-camera
    midpoints across fix cycles. Over ~N cycles the novel views progressively
    fill in gaps between training cameras — compound refinement.

    Mirrors NVIDIA's CameraPoseInterpolator.shift_poses():
      each cycle: current_pose += distance * (target - current) / ||target - current||
    """

    def __init__(self, train_cams):
        import torch.nn.functional as F

        # Extract [N, 3, 4] c2w matrices from MILo cameras.
        # MILo WVT[:3,:3] = R^T  →  cam-to-world R = WVT[:3,:3].T
        c2ws = []
        for cam in train_cams:
            R = cam.world_view_transform[:3, :3].float().cpu().T  # cam-to-world
            cc = cam.camera_center.float().cpu()
            c2w = torch.zeros(3, 4)
            c2w[:3, :3] = R
            c2w[:3, 3] = cc
            c2ws.append(c2w)
        self.c2ws = torch.stack(c2ws)       # [N, 3, 4]
        self.train_cams = train_cams

        centres = self.c2ws[:, :3, 3]       # [N, 3]
        n_cams = len(train_cams)

        # Sort cameras along principal movement axis, compute midpoints.
        if n_cams < 2:
            self.targets = None
            self.current_poses = None
            return

        principal = torch.pca_lowrank(
            centres - centres.mean(0), q=1)[0].squeeze()
        order = torch.argsort(principal)

        targets = []
        for k in range(len(order) - 1):
            i, j = order[k].item(), order[k + 1].item()
            mid = _slerp_c2w(self.c2ws[i], self.c2ws[j], 0.5)
            targets.append(torch.tensor(mid, dtype=torch.float32))

        self.targets = torch.stack(targets) if targets else None
        self.current_poses = None   # lazy init on first cycle

        if self.targets is not None:
            print(f"[difix] ProgressiveShifter: {n_cams} cameras, "
                  f"{len(targets)} midpoint targets")

    def _init_current_poses(self):
        """Initialise current poses as the nearest training camera to each target."""
        centres = self.c2ws[:, :3, 3]
        current = []
        for target in self.targets:
            target_pos = target[:3, 3]
            dists = torch.norm(centres - target_pos, dim=1)
            nearest = dists.argmin().item()
            current.append(self.c2ws[nearest].clone())
        self.current_poses = torch.stack(current)

    def generate_views(
        self,
        n_views: int,
        distance: float = 0.5,
        mask_penalty: Optional[torch.Tensor] = None,
    ) -> Optional[List[Tuple["_SyntheticCamera", int]]]:
        """Shift current poses toward targets, return novel (_SyntheticCamera, ref_idx) pairs.

        Returns None if targets are unavailable (triggers SLERP fallback).
        """
        import torch.nn.functional as F

        if self.targets is None:
            return None

        if self.current_poses is None:
            self._init_current_poses()

        centres = self.c2ws[:, :3, 3]
        forwards = -self.c2ws[:, :3, 2]   # camera forward = -Z axis in world space
        forwards = F.normalize(forwards, dim=1)

        # Shift each current pose toward its target by `distance` scene units.
        new_poses = []
        for current, target in zip(self.current_poses, self.targets):
            t_current = current[:3, 3]
            t_target = target[:3, 3]
            dist = torch.norm(t_target - t_current).item()
            t_ratio = min(distance / (dist + 1e-8), 1.0)
            shifted = _slerp_c2w(current, target, t_ratio)
            new_poses.append(torch.tensor(shifted, dtype=torch.float32))

        self.current_poses = torch.stack(new_poses)

        # Sub-sample to n_views if we have more targets than requested.
        n_available = len(self.current_poses)
        if n_available > n_views:
            indices = torch.randperm(n_available)[:n_views].tolist()
        else:
            indices = list(range(n_available))

        novel_cameras = []
        for idx in indices:
            c2w = self.current_poses[idx]
            interp_pos = c2w[:3, 3]

            # Ref camera: 60% direction + 40% position scoring (mask-penalty aware).
            pos_dists = torch.norm(centres - interp_pos, dim=1)
            pos_score = 1.0 - pos_dists / (pos_dists.max() + 1e-8)

            interp_fwd = -c2w[:3, 2]
            interp_fwd = F.normalize(interp_fwd, dim=0)
            dir_score = (forwards * interp_fwd.unsqueeze(0)).sum(dim=1)

            combined = 0.6 * dir_score + 0.4 * pos_score
            if mask_penalty is not None:
                combined = combined - mask_penalty
            ref_idx = combined.argmax().item()

            synth_cam = _camera_from_c2w(c2w, self.train_cams[ref_idx], uid=-(idx + 1))
            novel_cameras.append((synth_cam, ref_idx))

        return novel_cameras


# ── Difix model ───────────────────────────────────────────────────────────────

def init_difix(device: str = "cuda"):
    """Load DifixPipeline from HuggingFace (nvidia/difix_ref).

    DifixPipeline is defined in /workspace/Difix3D/src/pipeline_difix.py.
    The model is ~2 GB in fp16 and cached under HF_HOME.
    """
    difix_src = "/workspace/Difix3D/src"
    if difix_src not in sys.path:
        sys.path.insert(0, difix_src)

    from pipeline_difix import DifixPipeline  # noqa: PLC0415

    print("[difix] Loading nvidia/difix_ref (fp16) …")
    pipe = DifixPipeline.from_pretrained(
        "nvidia/difix_ref", trust_remote_code=True, torch_dtype=torch.float16,
    )
    # Keep on CPU by default — moved to GPU only during fix cycles to avoid OOM
    pipe.set_progress_bar_config(disable=True)
    print("[difix] DifixPipeline ready (CPU, moved to GPU on demand).")
    return pipe


# ── Image conversion helpers ──────────────────────────────────────────────────

def _tensor_to_pil(t: torch.Tensor) -> PILImage.Image:
    """Float [3, H, W] tensor in [0, 1] → RGB PIL Image."""
    arr = (t.clamp(0.0, 1.0) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return PILImage.fromarray(arr, mode="RGB")


def _pil_to_tensor(img: PILImage.Image, device: str = "cuda") -> torch.Tensor:
    """RGB PIL Image → float [3, H, W] tensor in [0, 1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


# ── Phase 1: fix cycle ────────────────────────────────────────────────────────

def difix_fix_step(
    gaussians,
    scene,
    difix_pipe,
    render_func,
    pipe,                    # MILo PipelineParams object (NOT the Difix pipeline)
    gaussians_optimizer,
    opt,
    background: torch.Tensor,
    iteration: int,
    novel_data_pool: List[Tuple],
    n_views: int = 48,
    n_grad_steps: int = 1,
    save_debug_images: bool = True,
    lambda_dssim: float = 0.2,
    difix_tau: int = _DIFIX_TAU_DEFAULT,
) -> None:
    """Run one Difix3D+ fix cycle within MILo training.

    Novel cameras are generated via ProgressiveShifter (NVIDIA approach):
    poses progressively march from training cameras toward midpoints across
    cycles. Falls back to random distant-pair SLERP if shifter unavailable.

    Pool is flushed at the start of each cycle (NVIDIA approach): stale
    data from worse model states is discarded, only the latest cycle's
    enhanced images are retained for reuse.
    """
    from utils.loss_utils import l1_loss    # noqa: PLC0415
    from fused_ssim import fused_ssim       # noqa: PLC0415
    import gc

    global _shifter

    difix_timesteps = [difix_tau - 1]
    print(
        f"[difix] Fix cycle @ iter {iteration} — "
        f"{n_views} novel views × {n_grad_steps} grad step(s), tau={difix_tau}"
    )

    # Free VRAM before moving Difix pipeline to GPU
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"[difix] VRAM free before Difix load: {free_gb:.1f} GB")

    try:
        difix_pipe.enable_attention_slicing(1)
    except Exception:
        pass
    try:
        difix_pipe.enable_vae_slicing()
    except Exception:
        pass

    difix_pipe.to(device="cuda", dtype=torch.float16)
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"[difix] Pipeline on GPU (fp16), VRAM free: {free_gb:.1f} GB")

    train_cams = scene.getTrainCameras()
    if len(train_cams) < 2:
        print("[difix] Need ≥2 training cameras — skipping.")
        return

    # ── Flush pool (NVIDIA approach) ──────────────────────────────────────────
    # Each cycle replaces stale data from a worse model with fresh renders.
    novel_data_pool.clear()

    # ── Compute mask penalty for ref selection ────────────────────────────────
    centres = torch.stack([c.camera_center for c in train_cams], dim=0)  # [N, 3]
    forwards = torch.stack(
        [-c.world_view_transform[:3, 2] for c in train_cams], dim=0
    )
    forwards = torch.nn.functional.normalize(forwards, dim=1)

    mask_penalty = torch.zeros(len(train_cams), device=centres.device)
    for ci, cam in enumerate(train_cams):
        gt_mask = getattr(cam, "gt_mask", None)
        if gt_mask is not None:
            mask_penalty[ci] = (1.0 - gt_mask.mean().item()) * 2.0

    # ── Generate novel cameras (ProgressiveShifter or SLERP fallback) ─────────
    if _shifter is None:
        _shifter = ProgressiveShifter(train_cams)

    novel_cameras = _shifter.generate_views(n_views, mask_penalty=mask_penalty)

    if novel_cameras is None:
        # Fallback: random distant-pair SLERP (original MILo strategy)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(iteration)
        n_sample = min(n_views, len(train_cams))
        perm = torch.randperm(len(train_cams), generator=rng).tolist()
        top_k = min(3, len(train_cams) - 1)

        novel_cameras = []
        for i in range(n_sample):
            a_idx = perm[i % len(perm)]
            dists = torch.norm(centres - centres[a_idx], dim=1)
            dists[a_idx] = -1.0
            _, topk_idx = torch.topk(dists, top_k)
            b_slot = torch.randint(top_k, (1,), generator=rng).item()
            b_idx = topk_idx[b_slot].item()
            t = float(torch.rand(1, generator=rng).item() * 0.70 + 0.15)

            synth_cam = _interpolate_camera(
                train_cams[a_idx], train_cams[b_idx], t=t,
                uid=-(iteration * 1000 + i),
            )
            interp_centre = synth_cam.camera_center
            pos_dists = torch.norm(centres - interp_centre, dim=1)
            pos_score = 1.0 - pos_dists / (pos_dists.max() + 1e-8)
            interp_fwd = torch.nn.functional.normalize(
                (1.0 - t) * forwards[a_idx] + t * forwards[b_idx], dim=0)
            dir_score = (forwards * interp_fwd.unsqueeze(0)).sum(dim=1)
            combined = 0.6 * dir_score + 0.4 * pos_score - mask_penalty
            ref_idx = combined.argmax().item()
            novel_cameras.append((synth_cam, ref_idx))

    # ── Render → Difix → grad update → pool ──────────────────────────────────
    total_steps = 0
    _DIFIX_SIZE = 512

    for pair_idx, (synth_cam, ref_cam_idx) in enumerate(novel_cameras):
        # Render novel view (no grad)
        with torch.no_grad():
            render_pkg_base = render_func(
                synth_cam, gaussians, pipe, background,
                require_coord=False, require_depth=False,
            )
            rendered_base = render_pkg_base["render"]   # [3, H, W] ∈ [0, 1]
            render_h, render_w = rendered_base.shape[1], rendered_base.shape[2]
            base_pil = _tensor_to_pil(rendered_base)
            del render_pkg_base, rendered_base

        torch.cuda.empty_cache()

        base_small = base_pil.resize((_DIFIX_SIZE, _DIFIX_SIZE), PILImage.LANCZOS)

        ref_cam = train_cams[ref_cam_idx]
        ref_pil = _tensor_to_pil(ref_cam.original_image)
        ref_small = ref_pil.resize((_DIFIX_SIZE, _DIFIX_SIZE), PILImage.LANCZOS)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            enhanced_small = difix_pipe(
                _DIFIX_PROMPT,
                image=base_small,
                ref_image=ref_small,
                num_inference_steps=1,
                timesteps=difix_timesteps,
                guidance_scale=0.0,
            ).images[0]
            enhanced_pil = enhanced_small.resize((render_w, render_h), PILImage.LANCZOS)
            enhanced_gpu = _pil_to_tensor(enhanced_pil, device="cuda")

        # Save debug images (first 4 views per cycle)
        if save_debug_images and pair_idx < 4:
            import os
            debug_dir = f"/data/output/milo/difix_debug/iter_{iteration}"
            os.makedirs(debug_dir, exist_ok=True)
            base_pil.save(f"{debug_dir}/view_{pair_idx:02d}_render.jpg")
            ref_pil.save(f"{debug_dir}/view_{pair_idx:02d}_ref.jpg")
            enhanced_pil.save(f"{debug_dir}/view_{pair_idx:02d}_enhanced.jpg")
            if pair_idx == 0:
                print(f"[difix] Debug images → {debug_dir}")

        # Immediate gradient update(s) with depth-based confidence weighting.
        for _ in range(n_grad_steps):
            render_pkg_grad = render_func(
                synth_cam, gaussians, pipe, background,
                require_coord=False, require_depth=True,
            )
            img   = render_pkg_grad["render"]
            radii = render_pkg_grad["radii"]
            depth = render_pkg_grad.get("expected_depth")   # [1, H, W] or None

            # Depth-based confidence: trust diffusion more where depth is missing
            if depth is not None:
                valid = (depth > 1e-4).float()
                weight = 1.0 - torch.nn.functional.avg_pool2d(
                    valid, kernel_size=11, stride=1, padding=5
                ) * 0.7   # range [0.3, 1.0]
            else:
                weight = 1.0

            Ll1      = (weight * torch.abs(img - enhanced_gpu)).mean()
            ssim_val = fused_ssim(img.unsqueeze(0), enhanced_gpu.unsqueeze(0))
            loss     = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_val)
            loss.backward()

            if gaussians.use_appearance_network:
                gaussians_optimizer.step()
            else:
                gaussians_optimizer.step(radii > 0, radii.shape[0])
            gaussians_optimizer.zero_grad(set_to_none=True)
            total_steps += 1

        # Store in pool for reuse (CPU to free VRAM)
        weight_cpu = weight.cpu() if torch.is_tensor(weight) else None
        novel_data_pool.append((synth_cam, enhanced_gpu.cpu(), weight_cpu))

    difix_pipe.to("cpu")
    torch.cuda.empty_cache()
    print(
        f"[difix] Fix cycle done. "
        f"Grad steps: {total_steps}  |  Pool size: {len(novel_data_pool)}"
    )


# ── Phase 2: novel data reuse ─────────────────────────────────────────────────

_novel_step_count = 0


def difix_novel_step(
    gaussians,
    novel_data_pool: List[Tuple],
    render_func,
    pipe,
    gaussians_optimizer,
    background: torch.Tensor,
    lambda_dssim: float = 0.2,
) -> None:
    """One extra gradient step using a random entry from the novel data pool.

    Called probabilistically (novel_data_lambda ≈ 0.40) every training
    iteration after the first fix cycle, mirroring the --novel_data_lambda
    mechanism in Difix3D+'s simple_trainer_difix3d.py.
    """
    from utils.loss_utils import l1_loss   # noqa: PLC0415
    from fused_ssim import fused_ssim      # noqa: PLC0415

    global _novel_step_count
    _novel_step_count += 1

    idx = torch.randint(len(novel_data_pool), (1,)).item()
    entry = novel_data_pool[idx]
    if len(entry) == 3:
        synth_cam, enhanced_cpu, weight_cpu = entry
    else:
        synth_cam, enhanced_cpu = entry
        weight_cpu = None
    enhanced = enhanced_cpu.to(background.device)
    weight = weight_cpu.to(background.device) if weight_cpu is not None else 1.0

    render_pkg = render_func(
        synth_cam, gaussians, pipe, background,
        require_coord=False, require_depth=False,
    )
    img   = render_pkg["render"]
    radii = render_pkg["radii"]

    Ll1      = (weight * torch.abs(img - enhanced)).mean()
    ssim_val = fused_ssim(img.unsqueeze(0), enhanced.unsqueeze(0))
    loss     = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_val)
    loss.backward()

    if gaussians.use_appearance_network:
        gaussians_optimizer.step()
    else:
        gaussians_optimizer.step(radii > 0, radii.shape[0])
    gaussians_optimizer.zero_grad(set_to_none=True)

    if _novel_step_count % 500 == 0:
        print(f"[difix] Novel data reuse: {_novel_step_count} steps so far "
              f"(pool={len(novel_data_pool)}, loss={loss.item():.4f})")
