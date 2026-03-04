"""Difix3D+ integration for MILo Mesh-In-the-Loop training.

Provides render-enhance-train fix cycles using NVIDIA Difix3D+ (SD-Turbo based).
Loaded only when --difix3d is passed to MILo's train.py.

Two-phase approach (matching the Difix3D+ baseline):

Phase 1 — Fix cycles (at scheduled iterations 9000 / 13000 / 17000):
  1. Sample n_views pairs of distant training cameras.
  2. SLERP rotation + LERP translation to obtain a novel-view pose.
  3. Render the current scene at the novel pose (no_grad).
  4. Run Difix on the render to produce an enhanced pseudo-GT image.
  5. Perform n_grad_steps immediate gradient updates against the enhanced image.
  6. Append (synthetic_camera, enhanced_image_cpu) to the novel_data_pool
     so it can be reused in subsequent training iterations.

Phase 2 — Novel data reuse (every iteration after first fix cycle):
  With probability novel_data_lambda (~0.3), sample a random entry from
  novel_data_pool and perform one extra gradient step against that stored
  enhanced image.  This mirrors the --novel_data_lambda mechanism in
  Difix3D+'s simple_trainer_difix3d.py.

Mesh-in-the-loop regularization is NOT applied during either phase;
only the Gaussian appearance parameters are updated.
"""

import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image as PILImage

_DIFIX_PROMPT = "remove degradation"
_DIFIX_TIMESTEPS = [199]
# Maximum entries kept in the novel data pool (oldest evicted beyond this limit).
# At 800×1200 px, 200 images ≈ 2.3 GB on CPU RAM — acceptable for 32 GB hosts.
_MAX_POOL_SIZE = 200


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


# ── Camera interpolation (SLERP + LERP) ──────────────────────────────────────

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

    # WVT[:3, :3] = R^T  →  R = (R^T)^T
    R_T_a = cam_a.world_view_transform[:3, :3].cpu().float().numpy()
    R_T_b = cam_b.world_view_transform[:3, :3].cpu().float().numpy()

    rots = Rotation.from_matrix(np.stack([R_T_a.T, R_T_b.T]))
    slerp = Slerp([0.0, 1.0], rots)
    R_interp = slerp([float(t)]).as_matrix()[0]   # [3, 3]
    R_T_interp = R_interp.T                        # back to R^T for WVT

    cc_a = cam_a.camera_center.cpu().float().numpy()
    cc_b = cam_b.camera_center.cpu().float().numpy()
    cc_interp = (1.0 - t) * cc_a + t * cc_b        # [3]

    # WVT[:3,:3] = R^T,  WVT[3,:3] = t_view = -R @ cc
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

    print("[difix] Loading nvidia/difix_ref …")
    pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print("[difix] DifixPipeline ready.")
    return pipe


# ── image conversion helpers ──────────────────────────────────────────────────

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
    novel_data_pool: List[Tuple],   # mutable list; (synth_cam, enhanced_cpu) appended here
    n_views: int = 48,
    n_grad_steps: int = 1,
    lambda_dssim: float = 0.2,
) -> None:
    """Run one Difix3D+ fix cycle within MILo training.

    Generates n_views novel-view cameras (diverse distant-pair SLERP),
    then for each:
      - renders the current scene (no_grad),
      - enhances with Difix → pseudo-GT,
      - performs n_grad_steps immediate gradient updates,
      - appends (synth_cam, enhanced_image_cpu) to novel_data_pool for reuse.

    Camera-pair strategy (CuriGS CVPR 2025 + InstantSplat):
      - Anchor: shuffled permutation of training cameras.
      - Partner: drawn from top-3 FURTHEST cameras from the anchor.
      - t: uniform random in [0.15, 0.85] for varied interpolation depth.
    """
    from utils.loss_utils import l1_loss    # noqa: PLC0415
    from fused_ssim import fused_ssim       # noqa: PLC0415

    print(
        f"[difix] Fix cycle @ iter {iteration} — "
        f"{n_views} novel views × {n_grad_steps} grad step(s)"
    )

    train_cams = scene.getTrainCameras()
    if len(train_cams) < 2:
        print("[difix] Need ≥2 training cameras for interpolation — skipping.")
        return

    centres = torch.stack([c.camera_center for c in train_cams], dim=0)  # [N, 3]
    rng = torch.Generator(device="cpu")
    rng.manual_seed(iteration)

    n_sample = min(n_views, len(train_cams))
    perm = torch.randperm(len(train_cams), generator=rng).tolist()
    top_k = min(3, len(train_cams) - 1)

    pairs: list = []
    t_vals: list = []
    for i in range(n_sample):
        a_idx = perm[i % len(perm)]
        dists = torch.norm(centres - centres[a_idx], dim=1)
        dists[a_idx] = -1.0
        _, topk_idx = torch.topk(dists, top_k)
        b_slot = torch.randint(top_k, (1,), generator=rng).item()
        b_idx = topk_idx[b_slot].item()
        t = float(torch.rand(1, generator=rng).item() * 0.70 + 0.15)
        pairs.append((a_idx, b_idx))
        t_vals.append(t)

    total_steps = 0
    base_uid = -(iteration * 1000)

    for pair_idx, ((idx_a, idx_b), t) in enumerate(zip(pairs, t_vals)):
        cam_a = train_cams[idx_a]
        cam_b = train_cams[idx_b]
        synth_uid = base_uid - pair_idx

        try:
            synth_cam = _interpolate_camera(cam_a, cam_b, t=t, uid=synth_uid)
        except Exception as exc:
            print(f"[difix] Camera interpolation failed ({idx_a},{idx_b}): {exc}")
            continue

        # Render (no grad)
        with torch.no_grad():
            render_pkg_base = render_func(
                synth_cam, gaussians, pipe, background,
                require_coord=False, require_depth=False,
            )
            rendered_base = render_pkg_base["render"]   # [3, H, W] ∈ [0, 1]

        # Difix enhancement
        with torch.no_grad():
            enhanced_pil = difix_pipe(
                _DIFIX_PROMPT,
                image=_tensor_to_pil(rendered_base),
                num_inference_steps=1,
                timesteps=_DIFIX_TIMESTEPS,
                guidance_scale=0.0,
            ).images[0]
            enhanced_gpu = _pil_to_tensor(enhanced_pil, device=rendered_base.device)

        # Immediate gradient update(s)
        for _ in range(n_grad_steps):
            render_pkg_grad = render_func(
                synth_cam, gaussians, pipe, background,
                require_coord=False, require_depth=False,
            )
            img   = render_pkg_grad["render"]
            radii = render_pkg_grad["radii"]

            Ll1      = l1_loss(img, enhanced_gpu)
            ssim_val = fused_ssim(img.unsqueeze(0), enhanced_gpu.unsqueeze(0))
            loss     = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_val)
            loss.backward()

            if gaussians.use_appearance_network:
                gaussians_optimizer.step()
            else:
                gaussians_optimizer.step(radii > 0, radii.shape[0])
            gaussians_optimizer.zero_grad(set_to_none=True)
            total_steps += 1

        # Store in pool for later reuse (CPU to avoid holding VRAM)
        novel_data_pool.append((synth_cam, enhanced_gpu.cpu()))
        if len(novel_data_pool) > _MAX_POOL_SIZE:
            # Evict oldest entry
            novel_data_pool.pop(0)

    print(
        f"[difix] Fix cycle done. "
        f"Grad steps: {total_steps}  |  Pool size: {len(novel_data_pool)}"
    )
    torch.cuda.empty_cache()


# ── Phase 2: novel data reuse ─────────────────────────────────────────────────

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

    Called probabilistically (novel_data_lambda ≈ 0.3) every training
    iteration after the first fix cycle, mirroring the --novel_data_lambda
    mechanism in Difix3D+'s simple_trainer_difix3d.py.
    """
    from utils.loss_utils import l1_loss   # noqa: PLC0415
    from fused_ssim import fused_ssim      # noqa: PLC0415

    # Random entry from pool
    idx = torch.randint(len(novel_data_pool), (1,)).item()
    synth_cam, enhanced_cpu = novel_data_pool[idx]
    enhanced = enhanced_cpu.to(background.device)

    render_pkg = render_func(
        synth_cam, gaussians, pipe, background,
        require_coord=False, require_depth=False,
    )
    img   = render_pkg["render"]
    radii = render_pkg["radii"]

    Ll1      = l1_loss(img, enhanced)
    ssim_val = fused_ssim(img.unsqueeze(0), enhanced.unsqueeze(0))
    loss     = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_val)
    loss.backward()

    if gaussians.use_appearance_network:
        gaussians_optimizer.step()
    else:
        gaussians_optimizer.step(radii > 0, radii.shape[0])
    gaussians_optimizer.zero_grad(set_to_none=True)
