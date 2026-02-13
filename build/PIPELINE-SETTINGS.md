# Onyx Pipeline — Working Settings

Tested configurations for each pipeline step. Updated as we verify each stage.

## 1. Ingest (onyx-ingest)

**Status:** Verified

```bash
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/build/test/sfm_data:/data \
  onyx-ingest:latest \
  python /workspace/360Extractor/src/main.py \
    --mode 360 \
    --input /data/input_video.mp4 \
    --output /data/images \
    --interval 0.8
```

| Setting | Value | Notes |
|---|---|---|
| Layout | `colmap` (default) | 12 views: 4 yaw × 3 pitch (-35°, 0°, +35°) at 90° FOV |
| Active cameras | All 12 (default) | Use all cameras; sky/ground handled by Segformer masking |
| Interval | Calculate per video | `interval = duration / (target_images / 12)` |
| Target images | 300-500 (outdoor) | 500-1000 for indoor/detailed scenes |
| Format | `jpg` (default) | Quality 95, negligible loss for SfM/3DGS |
| Resolution | `2048` (default) | Per camera view |
| AI mask | Not needed for drone | `--ai-mask` available for ground-level with pedestrians |
| GPU | Not required | Unless using `--ai-mask` or `--ai-skip` |

**Camera index map (colmap layout):**
```
Up row (+35°, yaw staggered by 45°):
 0: FrontRight_Up  1: RightBack_Up  2: BackLeft_Up  3: LeftFront_Up

Mid row (0°):
 4: Front_Mid      5: Right_Mid     6: Back_Mid     7: Left_Mid

Down row (-35°):
 8: Front_Down     9: Right_Down   10: Back_Down   11: Left_Down
```

**Why colmap layout (not cube)?** Matches COLMAP's official `panorama_sfm.py` "overlapping"
preset on the main branch: `num_steps_yaw=4, pitches_deg=(-35, 0, 35), fov=90°`.
The 3 pitch levels create 55° vertical overlap between rows, and the Up row's 45° yaw
stagger ensures seams from one row fall in the center of adjacent rows' views.
Standard cubemap (6 faces, zero overlap) produces sparse feature matches at boundaries.

**Interval calculation example:**
```
48s video, 400 target images, 12 cameras:
interval = 48 / (400 / 12) = 48 / 33.3 = 1.44s → use 1.4s
```

**Output:** `{output_dir}/{video_name}_processed/*.jpg`

---

## 2. InstantSfM (onyx-instantsfm)

**Status:** Verified

```bash
docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp \
  -v $(pwd)/build/test/sfm_data:/data \
  onyx-instantsfm:latest \
  bash -c "ins-feat --data_path /data --single_camera && ins-sfm --data_path /data"
```

| Setting | Value | Notes |
|---|---|---|
| `--single_camera` | Yes | All cube faces share same intrinsics (90° FOV pinhole) |
| `-e HOME=/tmp` | Required | Triton (bae) cache needs writable home dir with `--user` |
| GPU | Required | COLMAP GPU-accelerated SIFT |

**Test results (165 input images, 48s drone clip):**
- 132/165 images registered (80%)
- 20,061 sparse points
- Feature extraction: 107s, SfM: 50s

**Input:** `{data_path}/images/*.jpg`
**Output:** `{data_path}/sparse/0/` (cameras.bin, images.bin, points3D.bin)

---

## 3. MiLo (onyx-milo)

**Status:** Verified

```bash
docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp \
  -v $(pwd)/build/test/sfm_data:/data \
  onyx-milo:latest \
  --scene /data --metric outdoor --extract-mesh
```

| Setting | Value | Notes |
|---|---|---|
| `--scene` | Path to data dir | Must contain `sparse/0/` and `images/` |
| `--metric` | `outdoor` | Use `indoor` for indoor scenes |
| Iterations | 18000 (fixed) | Hardcoded by MiLo's `configs/fast`; not configurable via CLI |
| `--extract-mesh` | Yes | Runs SDF mesh extraction after training |
| `--mesh_config` | `default` | Options: verylowres, lowres, default, highres, veryhighres |
| `--dense` | No | Enable for highres/veryhighres mesh configs |
| `-e HOME=/tmp` | Required | Triton cache needs writable home dir with `--user` |
| GPU | Required | Training + mesh extraction are GPU-intensive |

**Build fix:** tetranerf C++ extension `.so` must be manually copied to site-packages (setup.py has CMake build commented out).

**Benchmark (132 images, 18k iterations, RTX 4090 on vast.ai):**

| Phase | Time |
|---|---|
| Training (18k iterations) | 35 min 24 sec |
| Mesh extraction (SDF) | ~1 min 30 sec |
| **Total** | **36 min 54 sec** |

| Metric | Value |
|---|---|
| Output splat | 33.2 MB point_cloud.ply |
| Output mesh | 66.3 MB mesh_learnable_sdf.ply |
| Checkpoint | 81.3 MB |
| GPU | RTX 4090 24GB |
| Training phases | 0-3k: ~80 it/s (pure 3DGS), 3k-9k: ~14 it/s (mesh reg.), 9k-18k: ~8 it/s (full mesh losses) |

**Input:** `{scene}/images/*.jpg` + `{scene}/sparse/0/` (cameras.bin, images.bin, points3D.bin)
**Output:** `{scene}/output/milo/` (checkpoint .pth + mesh_learnable_sdf.ply)

---

## 3b. gsplat/splatfacto (onyx-gsplat)

**Status:** Verified

```bash
docker run --rm --gpus all --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -v $(pwd)/build/test/sfm_data:/data \
  onyx-gsplat:latest \
  --scene /data
```

| Setting | Value | Notes |
|---|---|---|
| `--scene` | Path to data dir | Must contain `colmap/sparse/0/` and `colmap/images/` |
| `--iterations` | 30000 (default) | Standard splatfacto training |
| `--resolution` | 1 (default) | 1=full, 2=half, 4=quarter |
| `-e HOME=/tmp` | Required | Triton/torch cache needs writable home |
| `TORCHDYNAMO_DISABLE=1` | Set in Dockerfile | Disables torch inductor JIT (~10 min cold start). |
| `TORCH_EXTENSIONS_DIR` | `/opt/torch_extensions` | gsplat CUDA kernels pre-compiled at build time. |
| GPU | Required | Training is GPU-intensive |

**Data structure:** gsplat/nerfstudio expects `{scene}/colmap/sparse/0/` and `{scene}/colmap/images/` (different from MiLo which reads `sparse/0/` and `images/` at root). Copy or symlink from InstantSfM output.

**Build optimizations:**
- Wrapper passes `--downscale-factor` to prevent interactive prompt in headless Docker.
- Evaluation disabled (`--steps-per-eval-* 0`, `--vis tensorboard`) — pure overhead for production.
- Checkpoint saves only at end (`--steps-per-save 30000`).
- gsplat CUDA kernels pre-compiled during `docker build` (eliminates ~10-15 min JIT on first run).
- torch inductor disabled (`TORCHDYNAMO_DISABLE=1`) — avoids architecture-specific JIT compilation.

**Benchmark (132 images, 30k iterations, RTX 4090 on vast.ai):**

| Phase | Time |
|---|---|
| CUDA kernel compilation (old image, no pre-compile) | 17 min 15 sec |
| Actual training (30k iterations) | 28 min 48 sec |
| PLY export | ~2 sec |
| **Total (old image, with JIT)** | **46 min 3 sec** |
| **Total (new image, pre-compiled)** | **~29 min (estimated)** |

| Metric | Value |
|---|---|
| Output | 256 MB splat.ply |
| GPU | RTX 4090 24GB |
| Reference benchmark | gsplat standalone: 35m 49s on A100 (MipNeRF360) |

**Input:** `{scene}/colmap/images/*.jpg` + `{scene}/colmap/sparse/0/`
**Output:** `{scene}/output/splatfacto/` (checkpoint + exported splat.ply)

**Prototyping benchmark (132 images, 7k iterations, RTX 4090 on vast.ai):**

| Phase | Time |
|---|---|
| Training (7k iterations) | 2 min 27 sec |
| PLY export | ~2 sec |
| **Total** | **~2 min 30 sec** |

| Metric | Value |
|---|---|
| Output | 223 MB splat.ply |
| Gaussians | 941,396 |
| GPU | RTX 4090 24GB |

---

## Production vs Prototyping

**Production** — full quality for final delivery:

| Method | Iterations | Time (RTX 4090) | Splat Size | Gaussians | Mesh |
|---|---|---|---|---|---|
| MiLo | 18k (fixed) | ~37 min | 33 MB | ~107k | 66 MB (SDF) |
| gsplat | 30k | ~29 min | 256 MB | ~930k | No |

**Prototyping** — quick artist preview:

| Method | Iterations | Time (RTX 4090) | Splat Size | Gaussians | Notes |
|---|---|---|---|---|---|
| gsplat | 7k | ~2.5 min | 223 MB | ~941k | Fast iteration, good for previewing coverage/quality |
| MiLo | N/A | N/A | N/A | N/A | Not suitable — training schedule milestones require full 18k |

**Notes:**
- MiLo's mesh-in-the-loop training has fixed milestones (simplification at 3k and 8k iterations). Stopping early produces incomplete results.
- MiLo produces 8.7x fewer Gaussians than gsplat (107k vs 930k), resulting in much smaller splats — significant for real-time rendering in game engines.
- gsplat 7k gives a good preview of scene coverage and quality in under 3 minutes. Use for artist iteration before committing to a full production run.

---

## 4. OpenMVS (onyx-openmvs)

**Status:** Verified (TextureMesh OBJ export has segfault bug — use default MVS/PLY export)

```bash
# Step 1: Convert COLMAP to OpenMVS format
docker run --rm --gpus all \
  -v $(pwd)/build/test/sfm_data:/data \
  onyx-openmvs:latest \
  bash -c "cp /data/sparse/0/* /data/sparse/ && \
    InterfaceCOLMAP --input-file /data --image-folder /data/images --output-file /data/scene.mvs"

# Step 2: Densify point cloud
docker run --rm --gpus all \
  -v $(pwd)/build/test/sfm_data:/data \
  onyx-openmvs:latest \
  DensifyPointCloud --input-file /data/scene.mvs --output-file /data/scene_dense.mvs

# Step 3: Reconstruct mesh
docker run --rm --gpus all \
  -v $(pwd)/build/test/sfm_data:/data \
  onyx-openmvs:latest \
  ReconstructMesh --input-file /data/scene_dense.mvs --output-file /data/scene_mesh.mvs

# Step 4: Texture mesh (skip --export-type obj, segfaults on v2.3.0)
docker run --rm --gpus all \
  -v $(pwd)/build/test/sfm_data:/data \
  onyx-openmvs:latest \
  TextureMesh --input-file /data/scene_dense.mvs --mesh-file /data/scene_mesh.ply --output-file /data/scene_textured.mvs
```

| Setting | Value | Notes |
|---|---|---|
| GPU | Required | DensifyPointCloud uses CUDA for depth map computation |
| Path convention | `sparse/cameras.bin` at root | InterfaceCOLMAP appends `sparse/` internally — copy `sparse/0/*` to `sparse/` |
| Image paths | Absolute, resolved at InterfaceCOLMAP time | Stored in .mvs file; ensure images are accessible at same path for TextureMesh |
| `--export-type obj` | **Do NOT use** | Segfaults on v2.3.0 after completing texturing. Use default MVS+PLY export |

**Known issues:**
- InterfaceCOLMAP expects `sparse/cameras.bin` directly under `--input-file` path (not `sparse/0/cameras.bin`). Must copy files up.
- InterfaceCOLMAP stores absolute image paths in .mvs file based on resolved `--image-folder`. Ensure consistent paths across steps.
- TextureMesh OBJ export crashes with segfault after completing texturing. Use default export format.
- TextureMesh atlas generation is CPU-only (~40 min for 1.4M faces on 128-core EPYC). GPU only used for view initialization.

**Benchmark (132 images, RTX 4090 + 128-core EPYC on vast.ai):**

| Step | Time | Output |
|---|---|---|
| InterfaceCOLMAP | ~3s | scene.mvs (1.3 MB) |
| DensifyPointCloud | 53s | scene_dense.ply (142 MB, 2.36M points) |
| ReconstructMesh | 41s | scene_mesh.ply (27 MB, 718k vertices, 1.4M faces) |
| TextureMesh | 42m 3s | scene_textured.mvs + texture atlas (35k patches, 8192px) |
| **Total** | **~44 min** | |

**Input:** `{data}/images/*.jpg` + `{data}/sparse/0/` (cameras.bin, images.bin, points3D.bin)
**Output:** `{data}/scene_textured.mvs` + `{data}/scene_textured.ply` + texture atlas

---

## 5. Segformer (onyx-segformer)

**Status:** Verified (Stage 3 outlier removal skipped — scipy missing in container)

```bash
# Mask generation + Gaussian filtering
docker run --rm --gpus all --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -v $(pwd)/build/test/sfm_data:/data \
  onyx-segformer:latest \
  --input /data/images \
  --output /data/masks \
  --ply /data/output/milo/point_cloud/iteration_18000/point_cloud.ply \
  --cameras /data/sparse/0
```

| Setting | Value | Notes |
|---|---|---|
| `--input` | Path to input images | Same images used for SfM/3DGS |
| `--output` | Output mask directory | White=INCLUDE, Black=EXCLUDE (nerfstudio convention) |
| `--ply` | Optional: Gaussian PLY | If provided, runs Clean-GS filtering after mask generation |
| `--cameras` | COLMAP sparse dir | Required for Gaussian filtering (camera intrinsics/extrinsics) |
| Classes masked | sky, water, sea, person, boat | Default outdoor classes |
| `--skip-color-validation` | Yes (default) | Stage 2 disabled for speed |
| `-e HOME=/tmp` | Required | Model cache needs writable home dir with `--user` |
| GPU | Required | SegFormer inference on CUDA |

**Known issues:**
- scipy not installed in container — Stage 3 (k-NN outlier removal) is skipped
- RuntimeWarning on invalid cast during projection — cosmetic, doesn't affect results

**Benchmark (165 images + 110k Gaussians, RTX 4090 on vast.ai):**

| Phase | Time | Details |
|---|---|---|
| Mask generation | 26s | 165 images at ~6.2 it/s |
| Camera filtering | <1s | 111/132 cameras kept (21 excluded, <5% mask coverage) |
| Stage 1: Whitelist filtering | <1s | 92,787 kept / 17,762 removed (16.1%) |
| Stage 2: Color validation | Skipped | `--skip-color-validation` |
| Stage 3: Outlier removal | Skipped | scipy not available |
| **Total** | **~42s** | |

| Metric | Value |
|---|---|
| Input Gaussians | 110,549 |
| Output Gaussians | 92,787 (83.9% retained) |
| Compression | 16.1% removed |
| Output PLY | 29 MB |

**Input:** `{data}/images/*.jpg` + optional `{data}/sparse/0/` + Gaussian PLY
**Output:** `{data}/masks/*.png` + `{data}/point_cloud_filtered.ply`

---

## Pipeline Flags (for orchestrator)

| Flag | Values | Default | Effect |
|---|---|---|---|
| `--mode` | `360`, `standard` | `360` | 360 = equirect→cubemap, standard = raw frames |
| `--method` | `milo`, `gsplat`, `openmvs` | `milo` | Reconstruction method |
| `--scene` | `indoor`, `outdoor` | `outdoor` | outdoor excludes top camera face |
