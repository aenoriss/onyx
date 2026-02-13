# Onyx Pipeline — Deployment Guide

## Overview

The Onyx pipeline is a **host-side Python script** (`run_pipeline.py`) that pulls and runs individual tool containers from GitHub Container Registry. No special installation beyond Docker and NVIDIA drivers.

```
Host machine (Linux + Docker + NVIDIA drivers)
    |
    |  python run_pipeline.py --install        (one-time: pulls 6 images)
    |  python run_pipeline.py --input ...      (run pipeline)
    |
    +-- Docker daemon
        |
        +-- onyx-ingest:latest         (pulled from ghcr.io, runs as needed)
        +-- onyx-instantsfm:latest
        +-- onyx-milo:latest           (or onyx-gsplat, or onyx-openmvs)
        +-- onyx-segformer:latest
```

## Prerequisites

- **Linux** (Ubuntu 22.04+ recommended)
- **Docker** (27.x+) with the Docker daemon running
- **NVIDIA GPU** + drivers (for SfM, reconstruction, segformer)
- **nvidia-container-toolkit** (enables `--gpus all` in Docker)
- **Python 3** (for the orchestrator script — stdlib only, no pip deps)
- **ffprobe** (for auto-calculating frame extraction interval — part of ffmpeg)

## Quick Start

```bash
# 1. Clone the repo (or just copy run_pipeline.py)
git clone <repo-url>
cd onyx

# 2. Install (pulls all container images — one-time, ~25-30GB)
python build/run_pipeline.py --install

# 3. Run a pipeline
python build/run_pipeline.py \
  --input /path/to/video.mp4 --output ./workdir \
  --mode gaussian --scene outdoor --quality production --video 360
```

## Architecture

### Why This Design

| Concern | Decision |
|---|---|
| **Modularity** | Each tool is its own Docker image. Add/remove/swap tools without rebuilding anything else. |
| **Isolation** | Each tool runs in its own container with its own Python environment. No version conflicts. |
| **Build reliability** | Each image builds independently. One broken tool doesn't block others. |
| **Dependency conflicts** | Impossible — gsplat, nerfstudio, OpenMVS, MiLo each have their own site-packages. |
| **Image size** | Shared base layer (~10GB) + per-tool layers (~3-4GB each). Total unique disk: ~25-30GB. |
| **Deployment** | One `--install`, then run. No host-side pip, no CUDA toolkit, no compilation. |

### Runtime Flow

```
run_pipeline.py (host Python script)
    |
    +-- Step 1: docker run onyx-ingest:latest
    |   Mounts: ./workdir -> /data
    |   Writes: /data/images/*.jpg
    |   Completes, container removed (--rm)
    |
    +-- Step 2: docker run onyx-instantsfm:latest --gpus all
    |   Mounts: ./workdir -> /data
    |   Reads:  /data/images/*.jpg
    |   Writes: /data/sparse/0/*.bin
    |
    +-- Step 3: docker run onyx-milo:latest --gpus all
    |   Mounts: ./workdir -> /data
    |   Reads:  /data/images/ + /data/sparse/0/
    |   Writes: /data/output/milo/
    |
    +-- Step 4: docker run onyx-segformer:latest --gpus all
        Mounts: ./workdir -> /data
        Reads:  /data/images/ + /data/output/milo/*.ply
        Writes: /data/masks/
```

Key points:
- **Sequential** — one container at a time, orchestrated by the host script
- **Shared volume** — all containers mount the same host directory at `/data`
- **Ephemeral containers** — each runs with `--rm`, no state left in Docker
- **Persistent state** — `pipeline_state.json` on the shared volume survives crashes

### Container Inventory

| Image | Role | GPU | Size (approx) |
|---|---|---|---|
| `onyx-base:latest` | Shared foundation (CUDA + PyTorch) — not run directly | — | ~10GB |
| `onyx-ingest:latest` | Video frame extraction (360 + standard) | Optional | ~12GB |
| `onyx-instantsfm:latest` | Structure from Motion (COLMAP + bae) | Yes | ~15GB |
| `onyx-milo:latest` | Gaussian splatting + mesh (production quality) | Yes | ~14GB |
| `onyx-gsplat:latest` | Gaussian splatting (fast prototyping) | Yes | ~13GB |
| `onyx-openmvs:latest` | Photogrammetry (dense + mesh + texture) | Yes | ~13GB |
| `onyx-segformer:latest` | Semantic masking + Gaussian filtering | Yes | ~11GB |

All tool images inherit from `onyx-base`, sharing ~10GB of CUDA/PyTorch layers. Total unique disk usage when all images are pulled: ~25-30GB (not the sum, due to shared layers).

### Shared Filesystem

All containers mount the same host directory. This is the only coupling between steps.

```
./workdir/                          <- host directory
|-- pipeline_state.json             <- orchestrator writes (persistent state)
|-- .progress.json                  <- containers write (live progress)
|-- input_video.mp4                 <- symlink to user's video
|-- images/                         <- INGEST writes, SFM reads
|-- sparse/0/                       <- SFM writes, RECONSTRUCTION reads
|-- output/                         <- RECONSTRUCTION writes
|   |-- milo/                       <- (or splatfacto/ or openmvs/)
|-- masks/                          <- SEGFORMER writes
|-- logs/                           <- orchestrator writes (per-step logs)
```

---

## Commands Reference

### Install

```bash
python run_pipeline.py --install
```

Pulls all 6 tool images from `ghcr.io/aenoriss`. Shows progress per image. Skips already-installed images. Run once per machine.

### New Run

```bash
python run_pipeline.py \
  --input video.mp4 --output ./workdir \
  --mode gaussian --scene outdoor --quality production --video 360
```

All 6 parameters are required for a new run (no defaults — forces explicit choice).

### Resume

```bash
# Resume from where it failed
python run_pipeline.py --resume ./workdir

# Force re-run from a specific step
python run_pipeline.py --resume ./workdir --from-step RECONSTRUCTION
```

### Fork

```bash
# Run different reconstruction on same images (reuses INGEST + SFM)
python run_pipeline.py --fork ./workdir --mode photogrammetry --quality proto
```

### Data Management

```bash
# List batches
python run_pipeline.py --list ./

# Export batch as zip
python run_pipeline.py --export ./workdir -o ./batch.zip

# Import batch from zip
python run_pipeline.py --import-batch ./batch.zip -o ./imported

# Purge single batch
python run_pipeline.py --purge ./workdir

# Purge all batches in directory
python run_pipeline.py --purge-all ./
```

---

## Build & Push

### docker-compose.yml (Build Manifest)

Docker Compose is used as a **build tool** only (not runtime orchestration):

```yaml
services:
  base:
    build:
      context: .
      dockerfile: build/Dockerfile.onyx-base
    image: ghcr.io/aenoriss/onyx-base:latest

  ingest:
    build:
      context: .
      dockerfile: build/Dockerfile.onyx-ingest
    image: ghcr.io/aenoriss/onyx-ingest:latest
    depends_on: [base]

  instantsfm:
    build:
      context: .
      dockerfile: build/Dockerfile.onyx-instantsfm
    image: ghcr.io/aenoriss/onyx-instantsfm:latest
    depends_on: [base]

  milo:
    build:
      context: .
      dockerfile: build/Dockerfile.onyx-milo
    image: ghcr.io/aenoriss/onyx-milo:latest
    depends_on: [base]

  gsplat:
    build:
      context: .
      dockerfile: build/Dockerfile.onyx-gsplat
    image: ghcr.io/aenoriss/onyx-gsplat:latest
    depends_on: [base]

  openmvs:
    build:
      context: .
      dockerfile: build/Dockerfile.onyx-openmvs
    image: ghcr.io/aenoriss/onyx-openmvs:latest
    depends_on: [base]

  segformer:
    build:
      context: .
      dockerfile: build/Dockerfile.onyx-segformer
    image: ghcr.io/aenoriss/onyx-segformer:latest
    depends_on: [base]
```

### Build Commands

```bash
# Build all images (base first, then parallel)
docker compose build

# Push all to ghcr.io
docker compose push

# Build + push just one tool (after updating its code)
docker compose build milo && docker compose push milo
```

---

## Scaling Path

### Phase 1: Single Machine (Current)

One GPU server, `run_pipeline.py` orchestrates Docker containers locally. Works on any Linux machine with Docker + NVIDIA drivers — local workstation, cloud VPS, RunPod GPU Pod, Vast.ai.

### Phase 2: Multiple GPU Servers (10-20 concurrent users)

2-3 GPU servers, each with Docker + pre-pulled images. A Redis + Celery job queue distributes work. `run_pipeline.py` runs unmodified on whichever server picks the job.

```
Web API (Railway/Vercel, no GPU)
    |
    v
Redis job queue
    |-- GPU Server 1 (runs run_pipeline.py per job)
    |-- GPU Server 2
    +-- GPU Server 3
```

### Phase 3: Kubernetes (100+ concurrent users)

Argo Workflows on a managed Kubernetes cluster (AWS EKS, GCP GKE). Each pipeline step becomes a Kubernetes Job. Containers run unmodified — Argo replaces `run_pipeline.py` as the orchestrator.

**Key principle:** The containers never change between phases. Only the orchestration layer swaps out.

---

## Platform Compatibility

| Platform | Compatible? | Notes |
|---|---|---|
| **Local workstation** | Yes | Full Docker + GPU access |
| **RunPod GPU Pods** | Yes | Full VM with Docker. Script runs unmodified. |
| **Vast.ai** | Yes | Full VM with Docker. Script runs unmodified. |
| **AWS EC2 (g5/p4)** | Yes | Full VM with Docker. |
| **Hetzner GPU** | Yes | Full VM with Docker. |
| **Railway / Vercel / Heroku** | No | PaaS — no Docker daemon access, no GPUs |
| **RunPod Serverless** | No | Runs one container per request, no Docker-in-Docker |

---

## Progress Tracking

### Two-File System

| File | Owner | Purpose | Lifetime |
|------|-------|---------|----------|
| `pipeline_state.json` | Orchestrator (host) | Pipeline-level state, resume data | Persistent |
| `.progress.json` | Container (wrapper) | Live substep progress | Ephemeral |

### Shared Module

`tools/shared/pipeline_progress.py` is installed in `onyx-base` at `/workspace/pipeline_progress.py`. All containers inherit it. Provides:
- `progress()` — atomic JSON write for substep status
- `run_with_progress()` — run subprocess with stdout pattern matching for progress extraction

### Per-Container Substeps

| Container | Substeps | Measurable % |
|-----------|----------|-------------|
| onyx-ingest | 1: extracting_frames | Yes |
| onyx-instantsfm | 1: feature_extraction, 2: sfm | Partial |
| onyx-milo | 1: training, 2: mesh_extraction | Yes / No |
| onyx-gsplat | 1: training, 2: ply_export | Yes / No |
| onyx-openmvs | 1-4: interface, densify, mesh, texture | No (indeterminate) |
| onyx-segformer | 1: mask_generation, 2: gaussian_filtering | Yes / No |
