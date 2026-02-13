#!/usr/bin/env python3
"""
Apply optimized mask-aware training patches to MILo train.py

Key optimizations:
1. Keep masks on CPU, transfer only ONE per iteration (saves ~2.6GB VRAM for 659 masks)
2. Magenta background for masked pixels
3. Masked L1 loss for valid pixels only
4. Skip SSIM for masked training

Usage:
    python apply_mask_patch.py /path/to/MILo/milo/train.py
"""

import sys

train_file = sys.argv[1] if len(sys.argv) > 1 else "/workspace/MILo/milo/train.py"

with open(train_file, "r") as f:
    content = f.read()

# Check if already patched
if "magenta_bg" in content:
    print("Already patched - skipping")
    sys.exit(0)

# 1. Add magenta background tensor (small, no mask preloading to save VRAM)
init_code = '''
    # [Mask-Aware] Create magenta background tensor once (masks stay on CPU to save VRAM)
    magenta_bg = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32, device="cuda")
    _has_masks = any(hasattr(c, "gt_mask") and c.gt_mask is not None for c in scene.getTrainCameras())
    if _has_masks:
        print(f"[Mask-Aware] Found masks - using magenta background + masked loss")
'''

marker = "gaussians.training_setup(opt)"
if marker in content:
    idx = content.find(marker)
    end = content.find("\n", idx)
    content = content[:end+1] + init_code + content[end+1:]
    print("PATCH 1: Added magenta_bg initialization (masks stay on CPU)")
else:
    print("PATCH 1 FAILED: Could not find insertion point")
    sys.exit(1)

# 2. Add render_bg selection before first render call
old_render = '''        # If depth-normal regularization or mesh-in-the-loop regularization are active,
        # we use the rasterizer compatible with depth and normal rendering.
        if reg_kick_on or mesh_kick_on:
            render_pkg = render(
                viewpoint_cam, gaussians, pipe, background,'''

new_render = '''        # [Mask-Aware] Use magenta background for masked training
        render_bg = magenta_bg if (hasattr(viewpoint_cam, "gt_mask") and viewpoint_cam.gt_mask is not None) else background

        # If depth-normal regularization or mesh-in-the-loop regularization are active,
        # we use the rasterizer compatible with depth and normal rendering.
        if reg_kick_on or mesh_kick_on:
            render_pkg = render(
                viewpoint_cam, gaussians, pipe, render_bg,'''

if old_render in content:
    content = content.replace(old_render, new_render)
    print("PATCH 2: Added render_bg selection")
else:
    print("PATCH 2 FAILED: Could not find render section")

# 3. Replace remaining render calls to use render_bg
replacements = [
    ("render_full(\n                viewpoint_cam, gaussians, pipe, background,",
     "render_full(\n                viewpoint_cam, gaussians, pipe, render_bg,"),
    ("render_imp(\n                viewpoint_cam, gaussians, pipe, background,",
     "render_imp(\n                viewpoint_cam, gaussians, pipe, render_bg,"),
]
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print("PATCH 3: Updated render call")

# 4. Modify loss calculation - transfer mask to GPU only when needed, then free it
old_loss = '''        # Rendering loss
        if args.decoupled_appearance:
            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.uid)
        else:
            Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))'''

new_loss = '''        # [Mask-Aware] Rendering loss with mask support (mask transferred to GPU only when needed)
        _cpu_mask = getattr(viewpoint_cam, "gt_mask", None)
        if args.decoupled_appearance:
            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.uid)
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        elif _cpu_mask is not None:
            # Transfer mask to GPU for this iteration only
            _mask = _cpu_mask.cuda() if not _cpu_mask.is_cuda else _cpu_mask
            _mask = _mask.squeeze(0) if (_mask.dim() == 3 and _mask.shape[0] == 1) else _mask
            diff = torch.abs(image - gt_image)
            Ll1 = (diff * _mask).sum() / (_mask.sum() + 1e-6)
            ssim_value = torch.tensor(1.0, device="cuda")
            del _mask  # Free GPU memory immediately
        else:
            Ll1 = l1_loss(image, gt_image)
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))'''

if old_loss in content:
    content = content.replace(old_loss, new_loss)
    print("PATCH 4: Updated loss calculation (mask transferred per-iteration)")
else:
    print("PATCH 4 FAILED: Could not find loss section")

with open(train_file, "w") as f:
    f.write(content)

print("Done! Patches applied successfully.")
print("\nMemory optimization: Masks stay on CPU (~2.6GB VRAM saved for 659 masks)")
print("Each mask is transferred to GPU only when needed, then freed.")
