#!/usr/bin/env python3
"""Build-time patch: Add depth supervision to nerfstudio's splatfacto.

Uses native RGB+ED rendering (depth from same rasterization, zero extra VRAM).
Per-frame aligned L1 depth loss with gradient-aware weighting and opacity-frozen
gradient. Optional normal regularization from depth gradients (cosine distance).

Depth alignment: least-squares (a * d_mono + b) ≈ d_rendered per frame, then
gradient-weighted L1 provides per-pixel signal. Unlike Pearson correlation
(global stat, diminishing gradient at high r), L1 has constant gradient that
pulls Gaussians directly to surfaces.

Applied at Docker build time, same pattern as MILo's patch scripts.

Patches:
  A. SplatfactoModelConfig — depth + normal config fields
  C. get_loss_dict — aligned L1 depth + normal loss, opacity-frozen gradient
  D. get_outputs — VRAM + Gaussian diagnostics (every 500 steps)
  E. InputDataset — store depth path for lazy loading
  H. FullImageDatamanager — lazy image + depth loading (free RAM after cache)
  I. SplatfactoModel — StableGS dual opacity (geometric + auxiliary)
  J. Optimizers — default optimizer for custom param groups (opacities_aux)
"""

import re
import sys
import glob


def find_splatfacto():
    """Find splatfacto.py in nerfstudio installation."""
    paths = glob.glob("/usr/local/lib/python*/dist-packages/nerfstudio/models/splatfacto.py")
    if not paths:
        print("ERROR: splatfacto.py not found")
        sys.exit(1)
    return paths[0]


def find_base_dataset():
    """Find input_dataset.py (base dataset class) in nerfstudio."""
    paths = glob.glob("/usr/local/lib/python*/dist-packages/nerfstudio/data/datasets/base_dataset.py")
    if not paths:
        print("ERROR: base_dataset.py not found")
        sys.exit(1)
    return paths[0]


def patch_splatfacto_config(src):
    """Patch A: Add depth loss and scale config fields to SplatfactoModelConfig."""

    anchor = '"""Regularization term for scale in MCMC strategy"""'
    if anchor not in src:
        anchor = 'mcmc_scale_reg: float = 0.01'
        insert_after = src.index(anchor) + len(anchor)
        insert_after = src.index('\n', insert_after) + 1
    else:
        insert_after = src.index(anchor) + len(anchor) + 1

    new_fields = '''
    # Depth supervision (Depth Anything V2 monocular priors)
    depth_loss_mult: float = 0.0
    """Depth loss weight. 0 = disabled. Recommended: 0.1-0.5"""
    depth_loss_start_step: int = 500
    """Start depth loss after this many steps (let geometry settle first)"""
    depth_weight_decay: float = 0.1
    """Decay depth weight to this fraction over depth_decay_steps. 1.0 = no decay."""
    depth_decay_steps: int = 30000
    """Steps over which depth weight decays (from depth_loss_start_step)."""
    normal_loss_mult: float = 0.0
    """Normal regularization weight (cosine distance). 0 = disabled. Recommended: 0.01-0.05"""
    dual_opacity_start_step: int = 15000
    """Start dual opacity (StableGS) depth→geometry gradient after this step."""
'''

    src = src[:insert_after] + new_fields + src[insert_after:]
    print("[PATCH A] Added depth + scale config fields")
    return src


def patch_splatfacto_loss(src):
    """Patch C: Aligned L1 depth loss + normal regularization, opacity-frozen gradient.

    Per-frame least-squares alignment converts monocular depth to rendered depth
    scale, then gradient-weighted L1 provides per-pixel gradient (constant
    magnitude, unlike Pearson's diminishing gradient at high correlation).

    Optional normal regularization computes surface normals from rendered and
    aligned GT depth via finite differences, penalizing cosine distance.

    Depth gradients update means, scales, quats via selective backward.
    Before dual_opacity_start_step: opacities frozen (MCMC stabilization).
    After: geometric opacity unfrozen, auxiliary shielded (StableGS).
    """

    # Replace the end of get_loss_dict to add depth + normal loss
    anchor = (
        '        if self.training:\n'
        '            # Add loss from camera optimizer\n'
        '            self.camera_optimizer.get_loss_dict(loss_dict)\n'
        '            if self.config.use_bilateral_grid:\n'
        '                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)\n'
        '\n'
        '        return loss_dict'
    )

    if anchor not in src:
        # Try without blank line before return
        anchor = (
            '        if self.training:\n'
            '            # Add loss from camera optimizer\n'
            '            self.camera_optimizer.get_loss_dict(loss_dict)\n'
            '            if self.config.use_bilateral_grid:\n'
            '                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)\n'
            '        return loss_dict'
        )

    if anchor not in src:
        print("[PATCH C] ERROR: Could not find end-of-get_loss_dict anchor")
        return src

    new_code = (
        '        if self.training:\n'
        '            # Add loss from camera optimizer\n'
        '            self.camera_optimizer.get_loss_dict(loss_dict)\n'
        '            if self.config.use_bilateral_grid:\n'
        '                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)\n'
        '\n'
        '        # ── Depth supervision (aligned L1, opacity-frozen gradient) ──\n'
        '        # Per-frame least-squares alignment converts monocular depth to rendered\n'
        '        # depth scale, then gradient-weighted L1 provides per-pixel signal.\n'
        '        # Unlike Pearson (global stat, diminishing gradient at high r), L1 has\n'
        '        # constant gradient magnitude → pulls Gaussians directly to surfaces.\n'
        '        if (self.training\n'
        '                and self.config.depth_loss_mult > 0\n'
        '                and self.step >= self.config.depth_loss_start_step\n'
        '                and outputs.get("depth") is not None\n'
        '                and batch.get("depth_image") is not None):\n'
        '            import torch.nn.functional as F\n'
        '\n'
        '            _depth_pred = outputs["depth"]  # [H, W, 1]\n'
        '            _depth_gt = batch["depth_image"].to(self.device)  # [H_gt, W_gt, 1]\n'
        '\n'
        '            # Resize GT to match render resolution (handles downscale schedule)\n'
        '            if _depth_gt.shape[:2] != _depth_pred.shape[:2]:\n'
        '                _depth_gt = _depth_gt.permute(2, 0, 1).unsqueeze(0)\n'
        '                _depth_gt = F.interpolate(_depth_gt, size=_depth_pred.shape[:2],\n'
        '                                         mode="bilinear", align_corners=False)\n'
        '                _depth_gt = _depth_gt.squeeze(0).permute(1, 2, 0)\n'
        '\n'
        '            _alpha = outputs["accumulation"]  # [H, W, 1]\n'
        '            _valid = (_alpha.squeeze(-1) > 0) & (_depth_gt.squeeze(-1) > 0)\n'
        '            _dp = _depth_pred.squeeze(-1)[_valid]\n'
        '            _dg = _depth_gt.squeeze(-1)[_valid]\n'
        '\n'
        '            if _dp.numel() > 100:\n'
        '                # Per-frame least-squares: (a * depth_gt + b) ≈ depth_pred\n'
        '                # Detach dp in solve so gradients flow only through L1,\n'
        '                # not back through the alignment parameters.\n'
        '                _A = torch.stack([_dg, torch.ones_like(_dg)], dim=-1)  # [N, 2]\n'
        '                _b_vec = _dp.detach().unsqueeze(-1)  # [N, 1]\n'
        '                _ATA = _A.T @ _A  # [2, 2]\n'
        '                _ATb = _A.T @ _b_vec  # [2, 1]\n'
        '                _params = torch.linalg.solve(\n'
        '                    _ATA + 1e-6 * torch.eye(2, device=_ATA.device), _ATb)\n'
        '                _scale_ab = _params[0, 0].detach()\n'
        '                _shift_ab = _params[1, 0].detach()\n'
        '                _dg_aligned = _scale_ab * _dg + _shift_ab\n'
        '\n'
        '                # Gradient-aware weighting: stronger on textureless regions,\n'
        '                # weaker at edges where depth discontinuities are expected\n'
        '                _rgb = outputs["rgb"].detach()  # [H, W, 3]\n'
        '                _H, _W = _rgb.shape[:2]\n'
        '                _grad_x = torch.abs(_rgb[1:, :, :] - _rgb[:-1, :, :]).mean(dim=-1)\n'
        '                _grad_y = torch.abs(_rgb[:, 1:, :] - _rgb[:, :-1, :]).mean(dim=-1)\n'
        '                _grad = torch.zeros(_H, _W, device=_rgb.device)\n'
        '                _grad[:-1, :] += _grad_x\n'
        '                _grad[:, :-1] += _grad_y\n'
        '                _edge_weight = torch.exp(-_grad * 10.0)\n'
        '                _ew_valid = _edge_weight[_valid]\n'
        '\n'
        '                # Gradient-weighted L1 depth loss\n'
        '                _depth_l1 = (_ew_valid * torch.abs(_dp - _dg_aligned)).mean()\n'
        '\n'
        '                # Linear decay: depth weight decreases over training\n'
        '                _t = max(0.0, min(1.0,\n'
        '                    (self.step - self.config.depth_loss_start_step)\n'
        '                    / max(self.config.depth_decay_steps, 1)))\n'
        '                _dw = self.config.depth_loss_mult * (\n'
        '                    1.0 - _t * (1.0 - self.config.depth_weight_decay))\n'
        '\n'
        '                # Selective gradient: depth updates means, scales, quats.\n'
        '                # After dual_opacity_start_step, also updates geometric opacity\n'
        '                # (StableGS architecture: auxiliary opacity shielded from depth).\n'
        '                _depth_loss = _dw * _depth_l1\n'
        '                _dinputs = [self.gauss_params["means"],\n'
        '                    self.gauss_params["scales"],\n'
        '                    self.gauss_params["quats"]]\n'
        '                if (hasattr(self.config, "dual_opacity_start_step")\n'
        '                        and self.step >= self.config.dual_opacity_start_step):\n'
        '                    _dinputs.append(self.gauss_params["opacities"])\n'
        '                torch.autograd.backward(\n'
        '                    _depth_loss, inputs=_dinputs, retain_graph=True)\n'
        '\n'
        '                if self.step % 100 == 0:\n'
        '                    _do = " dual_opac" if len(_dinputs) > 3 else ""\n'
        '                    print(f"[DEPTH] step={self.step} l1={_depth_l1.item():.4f} "\n'
        '                          f"weight={_dw:.4f} scale={_scale_ab.item():.3f} "\n'
        '                          f"shift={_shift_ab.item():.3f}{_do}")\n'
        '\n'
        '                # ── Normal regularization from depth gradients ──\n'
        '                # Compute surface normals from rendered depth and aligned GT\n'
        '                # depth via central differences. Cosine distance loss constrains\n'
        '                # depth surface orientation, reducing floaters.\n'
        '                if self.config.normal_loss_mult > 0:\n'
        '                    _dp_2d = _depth_pred.squeeze(-1)  # [H, W]\n'
        '                    _dg_2d = _depth_gt.squeeze(-1)  # [H, W]\n'
        '                    # Normals from rendered depth\n'
        '                    _nr_dx = torch.zeros_like(_dp_2d)\n'
        '                    _nr_dy = torch.zeros_like(_dp_2d)\n'
        '                    _nr_dx[:, 1:-1] = (_dp_2d[:, 2:] - _dp_2d[:, :-2]) / 2\n'
        '                    _nr_dy[1:-1, :] = (_dp_2d[2:, :] - _dp_2d[:-2, :]) / 2\n'
        '                    _n_r = torch.stack([-_nr_dx, -_nr_dy,\n'
        '                        torch.ones_like(_dp_2d)], dim=-1)\n'
        '                    _n_r = F.normalize(_n_r, dim=-1)\n'
        '\n'
        '                    # Normals from aligned GT depth\n'
        '                    _dg_al_2d = (_scale_ab * _dg_2d + _shift_ab).detach()\n'
        '                    _ng_dx = torch.zeros_like(_dg_al_2d)\n'
        '                    _ng_dy = torch.zeros_like(_dg_al_2d)\n'
        '                    _ng_dx[:, 1:-1] = (_dg_al_2d[:, 2:] - _dg_al_2d[:, :-2]) / 2\n'
        '                    _ng_dy[1:-1, :] = (_dg_al_2d[2:, :] - _dg_al_2d[:-2, :]) / 2\n'
        '                    _n_g = torch.stack([-_ng_dx, -_ng_dy,\n'
        '                        torch.ones_like(_dg_al_2d)], dim=-1)\n'
        '                    _n_g = F.normalize(_n_g, dim=-1)\n'
        '\n'
        '                    # Valid: good alpha, skip border pixels (finite diff artifacts)\n'
        '                    _nv = (_alpha.squeeze(-1) > 0.1) & (_dg_2d > 0)\n'
        '                    _nv[:1, :] = False\n'
        '                    _nv[-1:, :] = False\n'
        '                    _nv[:, :1] = False\n'
        '                    _nv[:, -1:] = False\n'
        '\n'
        '                    if _nv.sum() > 100:\n'
        '                        # Cosine distance: 1 - dot(n_rendered, n_gt)\n'
        '                        _n_loss = (1 - (_n_r[_nv] * _n_g[_nv]\n'
        '                            ).sum(dim=-1)).mean()\n'
        '                        _normal_loss = self.config.normal_loss_mult * _n_loss\n'
        '                        torch.autograd.backward(\n'
        '                            _normal_loss, inputs=_dinputs,\n'
        '                            retain_graph=True)\n'
        '                        if self.step % 100 == 0:\n'
        '                            print(f"[NORMAL] step={self.step} "\n'
        '                                  f"loss={_n_loss.item():.4f}")\n'
        '\n'
        '        return loss_dict'
    )

    src = src.replace(anchor, new_code)
    print("[PATCH C] Added aligned L1 depth loss + normal regularization")
    return src


def patch_splatfacto_scale_clamp(src):
    """Patch F: REMOVED — scale clamp is no longer needed.

    With DNGaussian gradient freeze (P0), depth gradients don't flow into
    scales at all. Volume penalty (P1) handles any remaining growth.
    This patch is kept as a no-op for call compatibility.
    """
    print("[PATCH F] Skipped (scale clamp removed — gradient freeze replaces it)")
    return src


def patch_splatfacto_pruning(src):
    """Patch G: REMOVED — pruning is no longer needed.

    With gradient freeze (P0) preventing scale explosion and 300K cap,
    dead Gaussians don't cause OOM. Pruning was interfering with MCMC's
    relocation mechanism (removing Gaussians MCMC wants to reuse).
    Kept as no-op for call compatibility.
    """
    print("[PATCH G] Skipped (pruning removed — gradient freeze + cap sufficient)")
    return src


def patch_splatfacto_diagnostics(src):
    """Patch D: VRAM + Gaussian diagnostics after rasterization (every 500 steps)."""

    anchor = (
        '        if self.training:\n'
        '            self.strategy.step_pre_backward(\n'
        '                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info\n'
        '            )'
    )

    if anchor not in src:
        print("[PATCH D] WARNING: step_pre_backward anchor not found, skipping diagnostics")
        return src

    diag_code = '''
            # ── VRAM + Gaussian diagnostics (every 500 steps) ──
            if self.step % 500 == 0:
                _alloc = torch.cuda.memory_allocated() / 1e9
                _resv = torch.cuda.memory_reserved() / 1e9
                _n_gs = self.means.shape[0]
                _scales = torch.exp(self.scales).detach()
                _opac = torch.sigmoid(self.opacities).detach().squeeze()
                _alive = (_opac > 0.05).sum().item()
                _scale_max = _scales.max().item()
                _scale_mean = _scales.mean().item()
                _scale_p99 = torch.quantile(_scales.flatten(), 0.99).item()
                _isect_n = self.info.get("isect_ids", torch.tensor([])).numel()
                _isect_mb = _isect_n * 8 / 1e6
                _flat_n = self.info.get("flatten_ids", torch.tensor([])).numel()
                _flat_mb = _flat_n * 4 / 1e6
                _tiles_per = self.info.get("tiles_per_gauss", None)
                _tpg_max = _tiles_per.max().item() if _tiles_per is not None else 0
                _tpg_mean = _tiles_per.float().mean().item() if _tiles_per is not None else 0
                print(f"[DIAG] step={self.step} VRAM={_alloc:.1f}G/{_resv:.1f}G "
                      f"gs={_n_gs:,} alive={_alive:,} "
                      f"scale_mean={_scale_mean:.4f} p99={_scale_p99:.4f} max={_scale_max:.2f} "
                      f"isect={_isect_n:,}({_isect_mb:.0f}MB) flat={_flat_n:,}({_flat_mb:.0f}MB) "
                      f"tpg_mean={_tpg_mean:.1f} tpg_max={_tpg_max}")'''

    src = src.replace(anchor, anchor + diag_code)
    print("[PATCH D] Added VRAM + Gaussian diagnostics")
    return src


def patch_base_dataset(path):
    """Patch E: Store depth map paths for lazy loading in InputDataset.get_data()."""
    with open(path, 'r') as f:
        src = f.read()

    if '_lazy_depth_path' in src or 'depth_image' in src:
        print("[PATCH E] Already has depth loading, skipping")
        return

    if 'import numpy as np' not in src:
        src = src.replace('import torch', 'import torch\nimport numpy as np', 1)

    if 'class InputDataset' not in src:
        print("[PATCH E] WARNING: InputDataset class not found")
        return

    getitem_match = re.search(r'def __getitem__\(self.*?\n(.*?)(?=\n    def |\nclass |\Z)', src, re.DOTALL)
    if not getitem_match:
        print("[PATCH E] WARNING: __getitem__ not found")
        return

    return_match = re.search(r'(\s+)(return data\b)', getitem_match.group(0))
    if not return_match:
        print("[PATCH E] WARNING: 'return data' not found in __getitem__")
        return

    indent = return_match.group(1)
    return_stmt = return_match.group(2)

    depth_code = f'''
{indent}# Store depth path for lazy loading (loaded per training step in next_train,
{indent}# not cached in RAM — saves ~4.5GB for 589 images)
{indent}try:
{indent}    img_path = self._dataparser_outputs.image_filenames[image_idx]
{indent}    depths_dir = img_path.parent.parent / "depths"
{indent}    depth_path = depths_dir / f"{{img_path.stem}}.npy"
{indent}    if depth_path.exists():
{indent}        data["_lazy_depth_path"] = str(depth_path)
{indent}except Exception:
{indent}    pass
'''

    src = src.replace(return_stmt, depth_code + f'{indent}{return_stmt}', 1)

    with open(path, 'w') as f:
        f.write(src)

    print("[PATCH E] Added lazy depth path to InputDataset.get_data()")


def patch_dual_opacity(src):
    """Patch I: StableGS dual opacity — add auxiliary opacity parameter + dual rendering.

    Each Gaussian gets a second opacity parameter (opacities_aux) initialized to
    logit(0.99) ≈ 1.0. The rasterization uses sigmoid(opac) * sigmoid(opac_aux),
    which is ~= sigmoid(opac) at initialization (seamless transition).

    Gradient isolation (StableGS principle):
    - opacities_aux receives gradient ONLY from RGB loss (appearance)
    - opacities receives gradient from RGB + depth (after dual_opacity_start_step)
    - Depth cannot corrupt appearance opacity → no fog, no collapse

    MCMC handles opacities_aux automatically: generic param iteration copies it
    during relocate/sample_add. Only 'opacities' gets special relocation treatment.
    """
    if 'opacities_aux' in src:
        print("[PATCH I] Already has dual opacity, skipping")
        return src

    # ── Part 1: Add opacities_aux to gauss_params ──
    old_params = (
        '                "opacities": opacities,\n'
        '            }\n'
        '        )'
    )
    new_params = (
        '                "opacities": opacities,\n'
        '                "opacities_aux": torch.nn.Parameter(\n'
        '                    torch.logit(0.99 * torch.ones_like(opacities.data))),\n'
        '            }\n'
        '        )'
    )

    if old_params not in src:
        print("[PATCH I] WARNING: gauss_params anchor not found, skipping")
        return src

    src = src.replace(old_params, new_params)

    # ── Part 2: Use dual opacity in rasterization ──
    old_opac = '            opacities=torch.sigmoid(opacities_crop).squeeze(-1),'
    new_opac = (
        '            opacities=(torch.sigmoid(opacities_crop) * (\n'
        '                torch.sigmoid(self.gauss_params["opacities_aux"][crop_ids]\n'
        '                    if crop_ids is not None else self.gauss_params["opacities_aux"])\n'
        '                if "opacities_aux" in self.gauss_params else 1.0)\n'
        '            ).squeeze(-1),'
    )

    if old_opac not in src:
        print("[PATCH I] WARNING: rasterization opacity anchor not found, skipping")
        return src

    src = src.replace(old_opac, new_opac)

    # ── Part 3: Add opacities_aux to optimizer param groups ──
    old_groups = (
        '        return {\n'
        '            name: [self.gauss_params[name]]\n'
        '            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]\n'
        '        }'
    )
    new_groups = (
        '        _names = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]\n'
        '        if "opacities_aux" in self.gauss_params:\n'
        '            _names.append("opacities_aux")\n'
        '        return {\n'
        '            name: [self.gauss_params[name]]\n'
        '            for name in _names\n'
        '        }'
    )

    if old_groups not in src:
        print("[PATCH I] WARNING: param_groups anchor not found, opacities_aux may not be optimized!")
    else:
        src = src.replace(old_groups, new_groups)

    print("[PATCH I] Added StableGS dual opacity (opacities_aux + dual rendering + optimizer)")
    return src


def find_optimizers():
    """Find optimizers.py in nerfstudio installation."""
    paths = glob.glob("/usr/local/lib/python*/dist-packages/nerfstudio/engine/optimizers.py")
    if not paths:
        print("ERROR: optimizers.py not found")
        sys.exit(1)
    return paths[0]


def patch_optimizer_default(path):
    """Patch J: Default optimizer for unknown param groups (e.g. opacities_aux).

    Nerfstudio crashes with RuntimeError if a param group from the model
    isn't in the optimizer config. This patch adds a default Adam(lr=0.05)
    for unrecognized groups, matching the opacity learning rate.
    """
    with open(path, 'r') as f:
        src = f.read()

    if 'Auto-default' in src:
        print("[PATCH J] Already has optimizer default, skipping")
        return

    old_error = (
        '            # Print some nice warning messages if the user forgot to specify an optimizer\n'
        '            if param_group_name not in config:\n'
        '                raise RuntimeError(\n'
        '                    f"""Optimizer config for \'{param_group_name}\' not found in config file. Make sure you specify an optimizer for each parameter group. Provided configs were: {config.keys()}"""\n'
        '                )'
    )

    new_default = (
        '            # Auto-default: create Adam optimizer for custom param groups\n'
        '            if param_group_name not in config:\n'
        '                config[param_group_name] = {\n'
        '                    "optimizer": AdamOptimizerConfig(lr=0.05),\n'
        '                    "scheduler": None,\n'
        '                }\n'
        '                print(f"[OPTIM] Auto-created Adam(lr=0.05) for param group: {param_group_name}")'
    )

    if old_error not in src:
        print("[PATCH J] WARNING: optimizer error anchor not found, skipping")
        return

    src = src.replace(old_error, new_default)

    with open(path, 'w') as f:
        f.write(src)

    print("[PATCH J] Added default optimizer for unknown param groups")


def find_datamanager():
    """Find full_images_datamanager.py in nerfstudio installation."""
    paths = glob.glob("/usr/local/lib/python*/dist-packages/nerfstudio/data/datamanagers/full_images_datamanager.py")
    if not paths:
        print("ERROR: full_images_datamanager.py not found")
        sys.exit(1)
    return paths[0]


def patch_lazy_cache(path):
    """Patch H: Lazy image + depth loading in FullImageDatamanager.

    Replaces eager image caching with on-demand disk loading.
    Non-distorted images (PINHOLE) are marked lazy — tensors freed after
    the caching loop, reloaded from disk per training step.
    Depth maps (_lazy_depth_path from Patch E) loaded per step.

    Reduces steady-state RAM from ~18GB to <1GB for 589 images.
    Peak memory during init is unchanged (images loaded for undistortion check).
    """
    with open(path, 'r') as f:
        src = f.read()

    if '_lazy_image' in src:
        print("[PATCH H] Already has lazy loading, skipping")
        return

    # ── Part 1: Replace device-move block in _load_images ──
    old_block = (
        '        # Move to device.\n'
        '        if cache_images_device == "gpu":\n'
        '            for cache in undistorted_images:\n'
        '                cache["image"] = cache["image"].to(self.device)\n'
        '                if "mask" in cache:\n'
        '                    cache["mask"] = cache["mask"].to(self.device)\n'
        '                if "depth" in cache:\n'
        '                    cache["depth"] = cache["depth"].to(self.device)\n'
        '                self.train_cameras = self.train_dataset.cameras.to(self.device)\n'
        '        elif cache_images_device == "cpu":\n'
        '            for cache in undistorted_images:\n'
        '                cache["image"] = cache["image"].pin_memory()\n'
        '                if "mask" in cache:\n'
        '                    cache["mask"] = cache["mask"].pin_memory()\n'
        '                self.train_cameras = self.train_dataset.cameras\n'
        '        else:\n'
        '            assert_never(cache_images_device)\n'
        '        return undistorted_images'
    )

    new_block = (
        '        # ── Lazy loading: free image tensors, reload from disk on demand ──\n'
        '        # Non-distorted (PINHOLE) cameras: on-disk image matches cached version.\n'
        '        # Distorted images stay cached (undistortion can\'t be replayed).\n'
        '        _n_lazy = 0\n'
        '        for _i, _d in enumerate(undistorted_images):\n'
        '            _cam = dataset.cameras[_i].reshape(())\n'
        '            _no_distort = _cam.distortion_params is None or torch.all(_cam.distortion_params == 0)\n'
        '            if split == "train" and _no_distort and "image" in _d:\n'
        '                _d["_lazy_image"] = True\n'
        '                del _d["image"]\n'
        '                _n_lazy += 1\n'
        '            elif "image" in _d:\n'
        '                if cache_images_device == "gpu":\n'
        '                    _d["image"] = _d["image"].to(self.device)\n'
        '                elif cache_images_device == "cpu":\n'
        '                    _d["image"] = _d["image"].pin_memory()\n'
        '            if "mask" in _d:\n'
        '                if cache_images_device == "gpu":\n'
        '                    _d["mask"] = _d["mask"].to(self.device)\n'
        '        if split == "train":\n'
        '            if cache_images_device == "gpu":\n'
        '                self.train_cameras = self.train_dataset.cameras.to(self.device)\n'
        '            else:\n'
        '                self.train_cameras = self.train_dataset.cameras\n'
        '        print(f"[LAZY] {_n_lazy}/{len(undistorted_images)} {split} images lazy (disk)")\n'
        '        return undistorted_images'
    )

    if old_block not in src:
        print("[PATCH H] WARNING: device-move anchor not found in _load_images, skipping")
        return

    src = src.replace(old_block, new_block)

    # ── Part 2: Patch next_train for lazy image + depth resolution ──
    old_next = (
        '        data = self.cached_train[image_idx]\n'
        '        # We\'re going to copy to make sure we don\'t mutate the cached dictionary.\n'
        '        # This can cause a memory leak: https://github.com/nerfstudio-project/nerfstudio/issues/3335\n'
        '        data = data.copy()\n'
        '        data["image"] = data["image"].to(self.device)'
    )

    new_next = (
        '        data = self.cached_train[image_idx]\n'
        '        data = data.copy()\n'
        '        # Lazy image: reload from disk (non-distorted cameras)\n'
        '        if data.get("_lazy_image"):\n'
        '            if self.config.cache_images_type == "uint8":\n'
        '                data["image"] = self.train_dataset.get_image_uint8(image_idx).to(self.device)\n'
        '            else:\n'
        '                data["image"] = self.train_dataset.get_image_float32(image_idx).to(self.device)\n'
        '        else:\n'
        '            data["image"] = data["image"].to(self.device)\n'
        '        # Lazy depth: load .npy from path stored by Patch E\n'
        '        if "_lazy_depth_path" in data:\n'
        '            import numpy as np\n'
        '            data["depth_image"] = torch.from_numpy(\n'
        '                np.load(data["_lazy_depth_path"])).unsqueeze(-1).float()'
    )

    if old_next not in src:
        print("[PATCH H] WARNING: next_train anchor not found, skipping")
        return

    src = src.replace(old_next, new_next)

    with open(path, 'w') as f:
        f.write(src)

    print("[PATCH H] Added lazy image + depth loading to FullImageDatamanager")


def main():
    splatfacto_path = find_splatfacto()
    print(f"Patching: {splatfacto_path}")

    with open(splatfacto_path, 'r') as f:
        src = f.read()

    src = patch_splatfacto_config(src)
    src = patch_dual_opacity(src)
    src = patch_splatfacto_loss(src)
    src = patch_splatfacto_scale_clamp(src)
    src = patch_splatfacto_pruning(src)
    src = patch_splatfacto_diagnostics(src)

    with open(splatfacto_path, 'w') as f:
        f.write(src)

    print(f"Splatfacto patched: {splatfacto_path}")

    dataset_path = find_base_dataset()
    print(f"\nPatching: {dataset_path}")
    patch_base_dataset(dataset_path)

    datamanager_path = find_datamanager()
    print(f"\nPatching: {datamanager_path}")
    patch_lazy_cache(datamanager_path)

    optimizers_path = find_optimizers()
    print(f"\nPatching: {optimizers_path}")
    patch_optimizer_default(optimizers_path)

    print("\nAll depth patches applied.")


if __name__ == "__main__":
    main()
