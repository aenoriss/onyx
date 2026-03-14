#!/usr/bin/env python3
"""Build-time patch: Add depth supervision to nerfstudio's splatfacto.

Uses native RGB+ED rendering (depth from same rasterization, zero extra VRAM).
Pearson correlation depth loss with opacity-frozen gradient (depth gradients
update positions, scales, and rotations but NOT opacities). This is critical for
MCMC: depth opacity gradients kill the population, causing scale growth → OOM.

Applied at Docker build time, same pattern as MILo's patch scripts.

Patches:
  A. SplatfactoModelConfig — depth config fields
  C. get_loss_dict — Pearson depth loss with means-only gradient freeze
  D. get_outputs — VRAM + Gaussian diagnostics (every 500 steps)
  E. InputDataset — load .npy depth maps from depths/ directory
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
'''

    src = src[:insert_after] + new_fields + src[insert_after:]
    print("[PATCH A] Added depth + scale config fields")
    return src


def patch_splatfacto_loss(src):
    """Patch C: Pearson depth loss with opacity-frozen gradient.

    Depth gradients update means, scales, and quats via selective backward.
    Opacities are frozen — critical for MCMC compatibility: depth gradients
    on opacities kill the population (200K→12K alive), causing RGB to
    compensate with larger scales → intersection buffer explosion → OOM.
    """

    # Replace the end of get_loss_dict to add depth loss
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
        '        # ── Depth supervision (Pearson correlation, means-only gradient freeze) ──\n'
        '        # Depth gradients only update positions — NOT scales, quats, or opacities.\n'
        '        # Critical for MCMC: opacity gradients from depth kill the population,\n'
        '        # causing RGB to grow scales to compensate → intersection buffer OOM.\n'
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
        '                # Pearson correlation (scale-invariant, ideal for monocular depth)\n'
        '                _dp_c = _dp - _dp.mean()\n'
        '                _dg_c = _dg - _dg.mean()\n'
        '                _pearson = (_dp_c * _dg_c).sum() / (\n'
        '                    torch.sqrt((_dp_c ** 2).sum() * (_dg_c ** 2).sum()) + 1e-8)\n'
        '\n'
        '                # Linear decay: depth weight decreases over training\n'
        '                _t = max(0.0, min(1.0,\n'
        '                    (self.step - self.config.depth_loss_start_step)\n'
        '                    / max(self.config.depth_decay_steps, 1)))\n'
        '                _dw = self.config.depth_loss_mult * (\n'
        '                    1.0 - _t * (1.0 - self.config.depth_weight_decay))\n'
        '\n'
        '                # Opacity-frozen gradient: depth updates means, scales, quats\n'
        '                # but NOT opacities. Opacity gradients from depth kill MCMC\n'
        '                # population (200K→12K alive) → scale explosion → OOM.\n'
        '                _depth_loss = _dw * (1 - _pearson)\n'
        '                torch.autograd.backward(\n'
        '                    _depth_loss,\n'
        '                    inputs=[self.gauss_params["means"],\n'
        '                            self.gauss_params["scales"],\n'
        '                            self.gauss_params["quats"]],\n'
        '                    retain_graph=True\n'
        '                )\n'
        '\n'
        '                if self.step % 100 == 0:\n'
        '                    print(f"[DEPTH] step={self.step} pearson={_pearson.item():.4f} weight={_dw:.4f}")\n'
        '\n'
        '        return loss_dict'
    )

    src = src.replace(anchor, new_code)
    print("[PATCH C] Added Pearson depth loss with means-only gradient freeze")
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
    """Patch E: Load .npy depth maps alongside images in InputDataset."""
    with open(path, 'r') as f:
        src = f.read()

    if 'depth_image' in src:
        print("[PATCH E] Already has depth_image loading, skipping")
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
{indent}# Load depth map if available (Onyx depth supervision)
{indent}try:
{indent}    img_path = self._dataparser_outputs.image_filenames[image_idx]
{indent}    depths_dir = img_path.parent.parent / "depths"
{indent}    depth_path = depths_dir / f"{{img_path.stem}}.npy"
{indent}    if depth_path.exists():
{indent}        depth_np = np.load(str(depth_path))
{indent}        data["depth_image"] = torch.from_numpy(depth_np).unsqueeze(-1).float()
{indent}except Exception:
{indent}    pass
'''

    src = src.replace(return_stmt, depth_code + f'{indent}{return_stmt}', 1)

    with open(path, 'w') as f:
        f.write(src)

    print("[PATCH E] Added depth_image loading to InputDataset")


def main():
    splatfacto_path = find_splatfacto()
    print(f"Patching: {splatfacto_path}")

    with open(splatfacto_path, 'r') as f:
        src = f.read()

    src = patch_splatfacto_config(src)
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

    print("\nAll depth patches applied.")


if __name__ == "__main__":
    main()
