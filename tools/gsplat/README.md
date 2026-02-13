# gsplat Training Container

Fast 3D Gaussian Splatting with MCMC densification and Mip-Splatting anti-aliasing.

Based on [gsplat](https://github.com/nerfstudio-project/gsplat) by Nerfstudio.

**Output:** Standard PLY format compatible with Unreal, Unity, web viewers, Supersplat, etc.

## Build

```bash
cd /home/aenoris/projects/onyx/tools/gsplat
docker build -t gsplat:v1 .
```

## Usage

```bash
# Basic training (MCMC strategy - recommended)
docker run --gpus all --rm \
    -v /path/to/scene:/data \
    gsplat:v1 \
    --scene /data

# Custom iterations and resolution
docker run --gpus all --rm \
    -v /path/to/scene:/data \
    gsplat:v1 \
    --scene /data \
    --iterations 30000 \
    --resolution 2

# Use default densification strategy
docker run --gpus all --rm \
    -v /path/to/scene:/data \
    gsplat:v1 \
    --scene /data \
    --strategy default
```

## Expected Scene Structure

```
scene/
├── colmap/
│   ├── images/          # Input images
│   ├── masks/           # Optional masks (white=exclude from training)
│   └── sparse/
│       └── 0/           # COLMAP reconstruction
│           ├── cameras.bin
│           ├── images.bin
│           └── points3D.bin
└── output/              # Training output (created)
    ├── ply/
    │   └── point_cloud_30000.ply  # Standard 3DGS format
    ├── ckpts/
    └── stats/
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--scene` | required | Path to scene directory |
| `--output` | scene/output | Output directory |
| `--iterations` | 30000 | Training iterations |
| `--resolution` | 1 | Resolution scale (1=full, 2=half, 4=quarter) |
| `--strategy` | mcmc | Densification: `mcmc` (better) or `default` |
| `--sh-degree` | 3 | Spherical harmonics degree (0-4) |
| `--eval` | false | Hold out test images for evaluation |

## Mask Support

Masks are automatically loaded from `colmap/masks/` if present.

- **White pixels (255)** = EXCLUDED from training (sky, water, dynamic objects)
- **Black pixels (0)** = INCLUDED in training (static scene)

Masks are applied to the loss function - masked pixels don't contribute gradients.
Use `segformer-mask` to generate masks before training.

## Output PLY Format

The output PLY is **standard 3DGS format** with:
- Position (x, y, z)
- Spherical harmonics (f_dc_0/1/2, f_rest_*)
- Opacity
- Scale (scale_0, scale_1, scale_2)
- Rotation quaternion (rot_0, rot_1, rot_2, rot_3)

Compatible with:
- [XV3DGS Unreal Plugin](https://github.com/xverse-engine/XScene-UEPlugin)
- [Supersplat](https://playcanvas.com/supersplat/editor)
- [Unity VR Gaussian Splatting](https://github.com/clarte53/GaussianSplattingVRViewerUnity)
- Web viewers (PlayCanvas, Three.js)

## Push to Registry

```bash
docker tag gsplat:v1 ghcr.io/aenoriss/gsplat:v1
docker push ghcr.io/aenoriss/gsplat:v1
```
