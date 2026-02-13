# MILo - Mesh-In-the-Loop Gaussian Splatting

Produces high-quality triangulated meshes from COLMAP datasets using differentiable mesh extraction during 3DGS training.

**Paper**: SIGGRAPH Asia 2025
**Authors**: Same team as SuGaR
**Output**: PLY mesh with vertex colors (compatible with Unreal, Blender, etc.)

## Docker Build

```bash
cd tools/milo
docker build -t onyx/milo:v1 .
```

## Usage

### Basic Training (Indoor)

```bash
docker run --gpus all -v $(pwd)/scenes:/data onyx/milo:v1 \
    --scene /data/myScene \
    --metric indoor
```

### Outdoor Scene

```bash
docker run --gpus all -v $(pwd)/scenes:/data onyx/milo:v1 \
    --scene /data/myScene \
    --metric outdoor
```

### High Resolution Mesh

```bash
docker run --gpus all -v $(pwd)/scenes:/data onyx/milo:v1 \
    --scene /data/myScene \
    --metric indoor \
    --mesh_config highres \
    --dense
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--scene` | required | Path to scene (with colmap/ subdirectory) |
| `--metric` | `indoor` | Scene type: `indoor` or `outdoor` |
| `--rasterizer` | `radegs` | Rasterizer: `radegs` or `gof` |
| `--mesh_config` | `default` | Resolution: `verylowres`, `lowres`, `default`, `highres`, `veryhighres` |
| `--dense` | false | Use dense Gaussians (recommended for highres+) |
| `--iterations` | 18000 | Training iterations |
| `--output` | scene/output/milo | Output directory |

## Mesh Resolution Configs

| Config | Vertices | File Size | Use Case |
|--------|----------|-----------|----------|
| `verylowres` | ~250k | <20MB | Quick preview |
| `lowres` | ~500k | <50MB | Mobile/web |
| `default` | ~2-5M | ~100MB | Standard quality |
| `highres` | ~9M | ~200MB | High detail |
| `veryhighres` | ~14M | ~300MB | Maximum quality |

## Output

```
scene/output/milo/
├── point_cloud/
│   └── iteration_18000/
│       └── point_cloud.ply    # Gaussian splats (can be viewed)
├── mesh_learnable_sdf.ply     # Final mesh output
└── cameras.json
```

## Integration with Unreal Engine

The output `mesh_learnable_sdf.ply` can be imported directly into Unreal Engine as a static mesh. No special plugin required (unlike Gaussian splats which need XVERSE).

## Comparison: gsplat vs MILo

| Aspect | gsplat | MILo |
|--------|--------|------|
| Output | Gaussian splats | Triangulated mesh |
| Rendering | Requires splat renderer | Standard mesh renderer |
| Physics | Not supported | Full collision/physics |
| Editing | Limited | Full mesh editing (Blender) |
| File size | Larger | Smaller (mesh is compressed) |
| Quality | Best for viewing | Best for integration |
