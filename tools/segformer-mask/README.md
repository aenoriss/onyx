# SegFormer Masking Container

Generates binary masks for sky, water, and dynamic objects to exclude from Gaussian Splatting training using semantic segmentation.

Uses [SegFormer-b5](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) pretrained on ADE20K (150 classes).

## Build

```bash
cd /home/aenoris/projects/onyx/tools/segformer-mask
docker build -t segformer-mask:v1 .
```

## Usage

```bash
# Basic usage (masks sky, water, sea, person, boat by default)
docker run --gpus all --rm \
    -v /path/to/scene:/data \
    segformer-mask:v1 \
    --input /data/colmap/images \
    --output /data/colmap/masks

# Custom classes
docker run --gpus all --rm \
    -v /path/to/scene:/data \
    segformer-mask:v1 \
    --input /data/colmap/images \
    --output /data/colmap/masks \
    --classes "sky,water,sea,river,lake"
```

## Available Classes

Common classes from ADE20K:
- `sky` (2), `water` (21), `sea` (26), `river` (60), `lake` (128)
- `person` (12), `car` (20), `bus` (80), `truck` (83)
- `boat` (76), `ship` (103), `airplane` (90)

## Output

Creates binary PNG masks:
- **White (255)** = exclude from training (sky, water, etc.)
- **Black (0)** = include in training

```
scene/colmap/masks/
├── frame_000000_cam_00.png
├── frame_000000_cam_01.png
└── ...
```

## Why SegFormer over Grounded-SAM2?

- **Grounding DINO** is an object detector - struggles with "environment" regions like sky/water
- **SegFormer** is a semantic segmenter - classifies every pixel, perfect for sky/water
- Simpler, faster, more accurate for this use case

## Push to Registry

```bash
docker tag segformer-mask:v1 ghcr.io/aenoriss/segformer-mask:v1
docker push ghcr.io/aenoriss/segformer-mask:v1
```
