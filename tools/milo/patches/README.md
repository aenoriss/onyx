# MILo Mask-Aware Training Patches

These patches add mask-aware training support to MILo for RGBA images where alpha channel masks out sky/unwanted regions.

## Files

- `train.py` - Main training script with:
  - Fixed magenta background `[1,0,1]` for masked pixels
  - Skip SSIM for masked regions
  - Periodic opacity pruning (every 1000 iters, opacity < 0.05)

- `scene_init.py` - Scene class with:
  - Auto-filter at save time (opacity + projection + magenta color)
  - Copy to `scene/__init__.py` in MILo installation

## Installation

```bash
# From MILo root directory
cp /path/to/patches/train.py milo/train.py
cp /path/to/patches/scene_init.py milo/scene/__init__.py
```

## Usage

Train with RGBA images (alpha = mask, 0=sky, 255=valid):

```bash
python train.py -s /path/to/colmap -m /path/to/output \
    --iterations 30000 \
    --imp_metric outdoor \
    --densify_grad_threshold 0.0003 \
    --lambda_depth_normal 0.1
```

## How It Works

1. **Magenta background**: Sky pixels render as magenta, so sky Gaussians learn magenta color
2. **Periodic pruning**: Removes low-opacity Gaussians during training (reduces memory)
3. **Auto-filter**: At save time, removes Gaussians that are:
   - Low opacity AND project primarily to masked regions
   - Magenta color AND project to masked regions

See `MASK_AWARE_TRAINING.md` in results/milo for full documentation.
