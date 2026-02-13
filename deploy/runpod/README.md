# Onyx Pipeline - RunPod Deployment

## Quick Start

```bash
# 1. Download the CLI script
curl -O https://raw.githubusercontent.com/aenoriss/onyx/main/deploy/runpod/onyx
chmod +x onyx

# 2. Pull the image (16.9GB)
./onyx pull

# 3. Start interactive shell
./onyx shell
```

## Commands

| Command | Description |
|---------|-------------|
| `./onyx pull` | Pull Docker image from ghcr.io |
| `./onyx shell` | Start interactive shell with GPU |
| `./onyx extract` | Run 360Extractor on video |
| `./onyx sfm` | Run InstantSfM on frames |
| `./onyx pipeline` | Run full pipeline (extract + sfm) |
| `./onyx status` | Show GPU and container status |
| `./onyx stop` | Stop running containers |

## Examples

### Interactive Shell
```bash
./onyx shell
# Inside container:
ins-feat --data_path /data/scene1/colmap --single_camera
ins-sfm --data_path /data/scene1/colmap --export_txt
```

### Extract Frames from 360 Video
```bash
./onyx extract \
    --input /data/videos/drone.mp4 \
    --output /data/scene1/frames \
    --interval 1.0 \
    --cameras 12 \
    --layout fibonacci
```

### Run SfM on Extracted Frames
```bash
./onyx sfm --scene /data/scene1
```

### Full Pipeline
```bash
./onyx pipeline \
    --input /data/videos/drone.mp4 \
    --scene /data/scene1
```

## Directory Structure

After running the pipeline, your scene will have:

```
/data/scene1/
├── frames/                 # Extracted pinhole images
│   ├── frame_000000_cam_00.png
│   ├── frame_000000_cam_01.png
│   └── ...
└── colmap/
    ├── images/             # Symlinks to frames
    ├── database.db         # COLMAP features
    └── sparse/0/
        ├── cameras.txt     # Intrinsics
        ├── images.txt      # Poses
        └── points3D.txt    # Sparse cloud
```

## Image Contents

- **COLMAP 3.11.0** - Built from source with headless GPU SIFT
- **InstantSfM** - Fast SfM with bae/pypose
- **360Extractor** - Equirectangular to pinhole conversion
- **PyTorch 2.5.1** + CUDA 12.6
- **gsplat 1.5.3** - Gaussian splatting kernels

## Requirements

- NVIDIA GPU with 8+ GB VRAM
- Docker with nvidia-container-toolkit
- RunPod or similar GPU cloud

## Troubleshooting

### Image pull fails
```bash
# Login to ghcr.io (if repo is private)
docker login ghcr.io -u aenoriss
```

### GPU not detected
```bash
# Verify nvidia-docker is working
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

### Out of memory during SfM
- Reduce number of images
- Use a GPU with more VRAM (24GB+ recommended for large scenes)
