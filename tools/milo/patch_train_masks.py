#!/usr/bin/env python3
"""
Apply mask-aware training patch to MILo's train.py.

Run once at Docker build time (AFTER patch_train_difix.py):
    python3 /workspace/patch_train_masks.py

Makes five targeted edits to /workspace/MILo/milo/train.py:
  1. Adds --masks argparse argument.
  2. Loads mask PNGs onto camera objects after Scene construction.
  3. Fixes SSIM with zero-error infill (replaces naive skip-SSIM approach).
  4. Suppresses densification gradients in masked regions.
  5. Delays densification start when masks are active.

All changes are conditional on --masks being provided, so the default MILo
flow is completely unchanged when --masks is not passed.
"""

import sys

TRAIN_PY = "/workspace/MILo/milo/train.py"

with open(TRAIN_PY, "r") as f:
    code = f.read()

# ── Patch A: argparser — add --masks arg ────────────────────────────────────
# Inserted after the existing --decoupled_appearance argument.

_PA_OLD = (
    '    # ----- Appearance Network for Exposure-aware loss -----\n'
    '    # > Inspired by GOF.\n'
    '    parser.add_argument("--decoupled_appearance", action="store_true")'
)

_PA_NEW = (
    '    # ----- Appearance Network for Exposure-aware loss -----\n'
    '    # > Inspired by GOF.\n'
    '    parser.add_argument("--decoupled_appearance", action="store_true")\n'
    '\n'
    '    # ----- Mask-Aware Training -----\n'
    '    parser.add_argument(\n'
    '        "--masks", type=str, default=None,\n'
    '        help="Path to mask directory (PNG, white=include, black=exclude)",\n'
    '    )'
)

assert _PA_OLD in code, "Patch A anchor not found in train.py"
code = code.replace(_PA_OLD, _PA_NEW, 1)
print("[patch-masks] A/5  --masks argparse arg added")

# ── Patch B: load masks onto cameras after Scene construction ───────────────
# Inserted right after:  scene = Scene(dataset, gaussians, resolution_scales=[1,2])

_PB_OLD = (
    '    scene = Scene(dataset, gaussians, resolution_scales=[1,2])\n'
    '    gaussians.training_setup(opt)'
)

_PB_NEW = (
    '    scene = Scene(dataset, gaussians, resolution_scales=[1,2])\n'
    '\n'
    '    # --- Mask-aware training: store mask paths (lazy load to save RAM) ---\n'
    '    _masks_loaded = 0\n'
    '    if getattr(args, "masks", None) and os.path.isdir(args.masks):\n'
    '        _masks_dir = args.masks\n'
    '        _unique_stems = set()\n'
    '        for _scale, _cams in scene.train_cameras.items():\n'
    '            for _cam in _cams:\n'
    '                _stem = os.path.splitext(os.path.basename(_cam.image_name))[0]\n'
    '                _mp = os.path.join(_masks_dir, f"{_stem}.png")\n'
    '                if os.path.exists(_mp):\n'
    '                    _cam._gt_mask_path = _mp\n'
    '                    _unique_stems.add(_stem)\n'
    '                else:\n'
    '                    _cam._gt_mask_path = None\n'
    '        _masks_loaded = len(_unique_stems)\n'
    '        for _scale, _cams in scene.test_cameras.items():\n'
    '            for _cam in _cams:\n'
    '                _cam._gt_mask_path = None\n'
    '        print(f"[Mask] Found {_masks_loaded} masks in {_masks_dir} (lazy load)")\n'
    '    else:\n'
    '        for _scale, _cams in scene.train_cameras.items():\n'
    '            for _cam in _cams:\n'
    '                if not hasattr(_cam, "_gt_mask_path"):\n'
    '                    _cam._gt_mask_path = None\n'
    '        for _scale, _cams in scene.test_cameras.items():\n'
    '            for _cam in _cams:\n'
    '                if not hasattr(_cam, "_gt_mask_path"):\n'
    '                    _cam._gt_mask_path = None\n'
    '\n'
    '    def _load_gt_mask(cam):\n'
    '        """Load mask from disk on demand — no RAM overhead at startup."""\n'
    '        mp = getattr(cam, "_gt_mask_path", None)\n'
    '        if mp is None:\n'
    '            return None\n'
    '        import torchvision.transforms.functional as TF\n'
    '        from PIL import Image\n'
    '        _mimg = Image.open(mp).convert("L")\n'
    '        _mimg = _mimg.resize((cam.image_width, cam.image_height), Image.NEAREST)\n'
    '        return TF.to_tensor(_mimg)\n'
    '\n'
    '    gaussians.training_setup(opt)'
)

assert _PB_OLD in code, "Patch B anchor not found in train.py"
code = code.replace(_PB_OLD, _PB_NEW, 1)
print("[patch-masks] B/5  Mask loading after Scene construction added")

# ── Patch C: fix SSIM with zero-error infill ────────────────────────────────
# The key quality fix. Replaces naive "skip SSIM" with infill approach that
# preserves full SSIM quality for unmasked pixels.
#
# Anchor: the loss computation block. This needs to match the ORIGINAL MILo
# code (post-difix-patch), since this is where we ADD mask support.

_PC_OLD = (
    '        # Rendering loss\n'
    '        if args.decoupled_appearance:\n'
    '            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.uid)\n'
    '        else:\n'
    '            Ll1 = l1_loss(image, gt_image)\n'
    '        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))\n'
    '        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)'
)

_PC_NEW = (
    '        # Rendering loss (mask-aware when gt_mask is available)\n'
    '        gt_mask = _load_gt_mask(viewpoint_cam)\n'
    '        if gt_mask is not None:\n'
    '            gt_mask = gt_mask.cuda()\n'
    '            if gt_mask.max() > 1:\n'
    '                gt_mask = gt_mask / 255.0\n'
    '\n'
    '        if args.decoupled_appearance:\n'
    '            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.uid)\n'
    '        else:\n'
    '            if gt_mask is not None:\n'
    '                # Masked L1: only penalize unmasked (white) pixels\n'
    '                _diff = torch.abs(image - gt_image)\n'
    '                Ll1 = (_diff * gt_mask).sum() / (gt_mask.sum() * 3 + 1e-6)\n'
    '            else:\n'
    '                Ll1 = l1_loss(image, gt_image)\n'
    '\n'
    '        if gt_mask is not None:\n'
    '            # Zero-error infill: copy rendered values into GT at masked pixels\n'
    '            # so SSIM sees zero difference there. image.detach() prevents\n'
    '            # self-reinforcing gradients in masked regions.\n'
    '            _inv_mask = 1.0 - gt_mask\n'
    '            _gt_infill = gt_image * gt_mask + image.detach() * _inv_mask\n'
    '            ssim_value = fused_ssim(image.unsqueeze(0), _gt_infill.unsqueeze(0))\n'
    '        else:\n'
    '            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))\n'
    '\n'
    '        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)'
)

assert _PC_OLD in code, "Patch C anchor not found in train.py"
code = code.replace(_PC_OLD, _PC_NEW, 1)
print("[patch-masks] C/5  SSIM zero-error infill loss added")

# ── Patch D: suppress densification gradients in masked regions ─────────────
# After add_densification_stats, zero out gradients for Gaussians projecting
# into masked pixels. This prevents split/clone from spawning Gaussians for
# transient objects.

_PD_OLD = (
    '                if gaussians._culling[:,viewpoint_cam.uid].sum()==0:\n'
    '                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)\n'
    '                else:\n'
    '                    # normalize xy gradient after culling\n'
    '                    gaussians.add_densification_stats_culling(viewspace_point_tensor, visibility_filter, gaussians.factor_culling)'
)

_PD_NEW = (
    '                if gaussians._culling[:,viewpoint_cam.uid].sum()==0:\n'
    '                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)\n'
    '                else:\n'
    '                    # normalize xy gradient after culling\n'
    '                    gaussians.add_densification_stats_culling(viewspace_point_tensor, visibility_filter, gaussians.factor_culling)\n'
    '\n'
    '                # Suppress densification in masked regions\n'
    '                _gt_mask_d = _load_gt_mask(viewpoint_cam)\n'
    '                if _gt_mask_d is not None and visibility_filter.sum() > 0:\n'
    '                    _mask_gpu = _gt_mask_d.cuda()\n'
    '                    if _mask_gpu.dim() == 3:\n'
    '                        _mask_gpu = _mask_gpu[0]  # [H, W]\n'
    '                    _H, _W = _mask_gpu.shape\n'
    '                    # Only process visible Gaussians to avoid huge allocations\n'
    '                    _vis_idx = visibility_filter.nonzero(as_tuple=True)[0]\n'
    '                    _xy = viewspace_point_tensor.detach()[_vis_idx, :2]\n'
    '                    _u = _xy[:, 0].long().clamp(0, _W - 1)\n'
    '                    _v = _xy[:, 1].long().clamp(0, _H - 1)\n'
    '                    _in_masked_vis = _mask_gpu[_v, _u] < 0.5\n'
    '                    if _in_masked_vis.any():\n'
    '                        _suppress_idx = _vis_idx[_in_masked_vis]\n'
    '                        gaussians.xyz_gradient_accum[_suppress_idx] = 0\n'
    '                        gaussians.denom[_suppress_idx] = 0\n'
    '                    del _mask_gpu'
)

assert _PD_OLD in code, "Patch D anchor not found in train.py"
code = code.replace(_PD_OLD, _PD_NEW, 1)
print("[patch-masks] D/5  Densification gradient suppression added")

# ── Patch E: delay densification start when masks are active ────────────────
# Push densify_from_iter from 500 → 3000 when --masks is set. Early iterations
# have noisy positions, gradient suppression alone isn't reliable until ~3k iters.
# Inserted after mask loading (Patch B), in the training setup section.

_PE_OLD = (
    '    print(f"[INFO] Using 3D Mip Filter: {gaussians.use_mip_filter}")'
)

_PE_NEW = (
    '    # Delay densification when masks are active (safety net for early noisy iterations)\n'
    '    if _masks_loaded > 0 and opt.densify_from_iter < 3000:\n'
    '        print(f"[Mask] Delayed densification: {opt.densify_from_iter} -> 3000")\n'
    '        opt.densify_from_iter = 3000\n'
    '\n'
    '    print(f"[INFO] Using 3D Mip Filter: {gaussians.use_mip_filter}")'
)

assert _PE_OLD in code, "Patch E anchor not found in train.py"
code = code.replace(_PE_OLD, _PE_NEW, 1)
print("[patch-masks] E/5  Delayed densification start added")

with open(TRAIN_PY, "w") as f:
    f.write(code)

print(f"\n[patch-masks] Successfully patched {TRAIN_PY}")
print("[patch-masks] Default MILo flow is unchanged — masks only active with --masks")
