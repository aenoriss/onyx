# Onyx

Containerized 3D reconstruction pipeline that turns a video into Gaussian splats, photogrammetry meshes, or both. A single Python script orchestrates Docker containers for frame extraction, structure-from-motion, reconstruction, and semantic masking — with resume, forking, and live progress tracking built in.

## Quick Start

```bash
# Download the orchestrator (or clone the repo)
wget https://github.com/aenoriss/onyx/raw/main/build/run_pipeline.py

# Pull all container images (~25-30 GB, one-time)
python run_pipeline.py --install

# Run a pipeline
python run_pipeline.py \
  --input video.mp4 --output ./workdir \
  --mode gaussian --scene outdoor --quality production
```

## CLI Reference

```
python run_pipeline.py [ACTION] [OPTIONS]

Actions:
  --install                     Pull all container images from ghcr.io
  --input/-i + --output/-o      Start a new pipeline run (requires --mode, --scene, --quality)
  --resume WORKDIR              Resume a failed/interrupted run
  --fork WORKDIR                Run different reconstruction on same data
  --list [DIR]                  List batches in a directory
  --export WORKDIR              Export batch as zip
  --import-batch ZIPFILE        Import batch from zip
  --purge WORKDIR               Delete working directory data
  --purge-all [DIR]             Purge all batches in directory

Options:
  --mode        gaussian | photogrammetry
  --scene       indoor | outdoor
  --quality     proto | production
  --video       normal | 360 | auto (default: auto)
  --from-step   Force re-run from INGEST | SFM | RECONSTRUCTION | SEGFORMER
  --interval    Override frame extraction interval (seconds)
  --local       Use local Docker images instead of ghcr.io
  --dry-run     Print commands without executing
  --skip-ingest / --skip-sfm / --skip-segformer
```

## Containers

| Image | Role | GPU |
|---|---|---|
| `onyx-base` | Shared foundation (CUDA + PyTorch) | — |
| `onyx-ingest` | Video frame extraction (360 + standard) | Optional |
| `onyx-instantsfm` | Structure from Motion | Yes |
| `onyx-milo` | Gaussian splatting + mesh (production) | Yes |
| `onyx-gsplat` | Gaussian splatting (fast proto) | Yes |
| `onyx-openmvs` | Photogrammetry (dense + mesh + texture) | Yes |
| `onyx-segformer` | Semantic masking + Gaussian filtering | Yes |

## Deployment

See [build/DEPLOYMENT.md](build/DEPLOYMENT.md) for architecture details, build commands, scaling path, and platform compatibility.
