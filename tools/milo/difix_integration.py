"""Difix3D+ integration for MILo Mesh-In-the-Loop training.

Provides render-enhance-train fix cycles using NVIDIA Difix3D+ (SD-Turbo based).
Loaded only when --difix3d is passed to MILo's train.py.

Fix cycle at each scheduled iteration:
  1. Sample n_views pairs of distant training cameras (diverse coverage).
  2. SLERP rotation + LERP translation to obtain a midpoint novel-view pose.
  3. Render the current scene at the novel pose (no_grad).
  4. Run Difix on the render to produce an enhanced pseudo-GT image.
  5. Perform n_grad_steps gradient updates (L1 + D-SSIM) against the enhanced image.

Mesh-in-the-loop regularization is NOT applied during these extra steps;
only the Gaussian appearance parameters are updated.
"""

import sys
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image as PILImage

_DIFIX_PROMPT = "remove degradation"
_DIFIX_TIMESTEPS = [199]


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

    # ── rotation: extract R (not R^T) from each WVT ──────────────────────────
    # WVT[:3, :3] = R^T  →  R = (R^T)^T
    R_T_a = cam_a.world_view_transform[:3, :3].cpu().float().numpy()
    R_T_b = cam_b.world_view_transform[:3, :3].cpu().float().numpy()

    rots = Rotation.from_matrix(np.stack([R_T_a.T, R_T_b.T]))
    slerp = Slerp([0.0, 1.0], rots)
    R_interp = slerp([float(t)]).as_matrix()[0]  # [3, 3]
    R_T_interp = R_interp.T                       # back to R^T for WVT

    # ── camera centre: LERP in world space ────────────────────────────────────
    cc_a = cam_a.camera_center.cpu().float().numpy()
    cc_b = cam_b.camera_center.cpu().float().numpy()
    cc_interp = (1.0 - t) * cc_a + t * cc_b       # [3]

    # ── rebuild WVT ───────────────────────────────────────────────────────────
    # WVT[:3,:3] = R^T,  WVT[3,:3] = t_view = -R @ cc
    wvt_np = np.zeros((4, 4), dtype=np.float32)
    wvt_np[:3, :3] = R_T_interp
    wvt_np[3, :3]  = -R_interp @ cc_interp
    wvt_np[3, 3]   = 1.0

    wvt = torch.tensor(wvt_np, dtype=torch.float32, device=device)

    # ── projection matrix: copy / extract from cam_a ─────────────────────────
    proj = getattr(cam_a, "projection_matrix", None)
    if proj is None:
        # Extract from FPT = WVT @ proj  →  proj = WVT^{-1} @ FPT
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


# ── main fix-cycle function ───────────────────────────────────────────────────

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
    n_views: int = 4,
    n_grad_steps: int = 1,
    lambda_dssim: float = 0.2,
) -> None:
    """Run one Difix3D+ fix cycle within MILo training.

    Generates n_views novel-view cameras by SLERP-interpolating between
    randomly sampled pairs of training cameras, then:
      - renders each novel view with the current Gaussians (no_grad),
      - enhances the render with Difix to get a pseudo-GT image,
      - performs n_grad_steps gradient update steps against the pseudo-GT.

    Parameters
    ----------
    render_func : callable
        The rasterizer render function (render_radegs or render_gof), with
        signature: render_func(cam, gaussians, pipe, background,
                               require_coord, require_depth) → render_pkg.
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

    # ── pair sampling: diverse DISTANT-pair selection ────────────────────────
    # Strategy (CuriGS CVPR 2025 + InstantSplat co-visibility):
    #   - Anchor cam_a: different for each view (shuffled permutation)
    #   - Partner cam_b: drawn from the top-3 FURTHEST cameras from cam_a;
    #     picking from top-3 (not always #1) avoids all views collapsing to
    #     the same extreme pair, maximising angular diversity.
    #   - Interpolation t: uniform in [0.15, 0.85] so synthetic poses
    #     sample varied positions between the pair, not just the midpoint.
    centres = torch.stack([c.camera_center for c in train_cams], dim=0)  # [N, 3]
    rng = torch.Generator(device="cpu")
    rng.manual_seed(iteration)                  # deterministic per fix cycle

    n_sample = min(n_views, len(train_cams))
    # Shuffle anchors so each synthetic view starts from a different camera
    perm = torch.randperm(len(train_cams), generator=rng).tolist()

    pairs: list[tuple[int, int]] = []
    t_vals: list[float] = []
    top_k = min(3, len(train_cams) - 1)

    for i in range(n_sample):
        a_idx = perm[i % len(perm)]
        dists = torch.norm(centres - centres[a_idx], dim=1)
        dists[a_idx] = -1.0                                     # exclude self
        _, topk_idx = torch.topk(dists, top_k)
        b_slot = torch.randint(top_k, (1,), generator=rng).item()
        b_idx = topk_idx[b_slot].item()
        # t ∈ [0.15, 0.85]
        t = float(torch.rand(1, generator=rng).item() * 0.70 + 0.15)
        pairs.append((a_idx, b_idx))
        t_vals.append(t)

    total_steps = 0
    base_uid = -(iteration * 1000)   # negative UIDs to avoid collision with real cams

    for pair_idx, ((idx_a, idx_b), t) in enumerate(zip(pairs, t_vals)):
        cam_a = train_cams[idx_a]
        cam_b = train_cams[idx_b]
        synth_uid = base_uid - pair_idx

        # Build the interpolated novel-view camera
        try:
            synth_cam = _interpolate_camera(cam_a, cam_b, t=t, uid=synth_uid)
        except Exception as exc:
            print(f"[difix] Camera interpolation failed for pair ({idx_a},{idx_b}): {exc}")
            continue

        # 1. Render at the novel viewpoint (no gradients needed)
        with torch.no_grad():
            render_pkg_base = render_func(
                synth_cam, gaussians, pipe, background,
                require_coord=False, require_depth=False,
            )
            rendered_base = render_pkg_base["render"]   # [3, H, W] ∈ [0, 1]

        # 2. Difix enhancement (inference only)
        with torch.no_grad():
            rendered_pil = _tensor_to_pil(rendered_base)
            enhanced_pil = difix_pipe(
                _DIFIX_PROMPT,
                image=rendered_pil,
                num_inference_steps=1,
                timesteps=_DIFIX_TIMESTEPS,
                guidance_scale=0.0,
            ).images[0]
            enhanced = _pil_to_tensor(enhanced_pil, device=rendered_base.device)

        # 3. Gradient update(s) — enhanced image is the pseudo-GT
        for _ in range(n_grad_steps):
            render_pkg_grad = render_func(
                synth_cam, gaussians, pipe, background,
                require_coord=False, require_depth=False,
            )
            img   = render_pkg_grad["render"]
            radii = render_pkg_grad["radii"]

            Ll1      = l1_loss(img, enhanced)
            ssim_val = fused_ssim(img.unsqueeze(0), enhanced.unsqueeze(0))
            loss     = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_val)
            loss.backward()

            if gaussians.use_appearance_network:
                gaussians_optimizer.step()
            else:
                visible = radii > 0
                gaussians_optimizer.step(visible, radii.shape[0])
            gaussians_optimizer.zero_grad(set_to_none=True)
            total_steps += 1

    print(f"[difix] Fix cycle done. Extra gradient steps: {total_steps}")
    torch.cuda.empty_cache()
