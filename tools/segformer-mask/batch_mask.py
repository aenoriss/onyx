#!/usr/bin/env python3
"""
Batch mask generator + optional Gaussian splat filtering for Onyx Pipeline.

Generates binary masks using SegFormer semantic segmentation.
When --ply is provided, also filters the Gaussian splat using Clean-GS.

Mask convention (nerfstudio compatible):
    - White (255) = INCLUDE in training (static scene)
    - Black (0) = EXCLUDE from training (sky, water, dynamic objects)

Usage:
    # Masks only
    python batch_mask.py --input /path/to/images --output /path/to/masks

    # Masks + filter Gaussian splat
    python batch_mask.py --input /path/to/images --output /path/to/masks \
        --ply /path/to/point_cloud.ply --cameras /path/to/cameras.json

    # With checkpoint filtering (for resumable training)
    python batch_mask.py --input /path/to/images --output /path/to/masks \
        --ply /path/to/point_cloud.ply --cameras /path/to/cameras.json \
        --checkpoint /path/to/chkpnt.pth
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from pipeline_progress import progress, run_with_progress

# ADE20K class indices for objects we want to mask out
# Full list: https://huggingface.co/datasets/huggingface/label-files/blob/main/ade20k-id2label.json
ADE20K_CLASSES = {
    "sky": 2,
    "water": 21,
    "sea": 26,
    "river": 60,
    "lake": 128,
    "boat": 76,
    "ship": 103,
    "person": 12,
    "car": 20,
    "bus": 80,
    "truck": 83,
    "airplane": 90,
    "bird": 3,  # Note: ADE20K class 3 is actually "floor" - bird not in ADE20K
}

# Default classes to mask
DEFAULT_CLASSES = "sky,water,sea,person,boat"

# Model checkpoint
MODEL_CHECKPOINT = "nvidia/segformer-b5-finetuned-ade-640-640"

# Clean-GS filter script location (inside container)
FILTER_SCRIPT = "/workspace/gaussian_mask_filter.py"


def load_model(device, checkpoint):
    """Load SegFormer model and processor."""
    print(f"Loading SegFormer ({checkpoint})...")
    processor = SegformerImageProcessor.from_pretrained(checkpoint)
    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint)
    model.to(device)
    model.eval()
    return processor, model


def process_image(image_path, processor, model, class_indices, device):
    """
    Process a single image and return a binary mask.
    Nerfstudio convention:
        White (255) = areas to INCLUDE in training (static scene)
        Black (0) = areas to EXCLUDE from training (sky, water, etc.)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, num_classes, H/4, W/4)

    # Upsample to original size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(original_size[1], original_size[0]),  # (H, W)
        mode="bilinear",
        align_corners=False
    )

    # Get predicted class per pixel
    predicted_classes = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

    # Create binary mask (nerfstudio convention):
    # Start with white (255 = include), set black (0 = exclude) where target classes detected
    mask = np.full(predicted_classes.shape, 255, dtype=np.uint8)
    for class_idx in class_indices:
        mask[predicted_classes == class_idx] = 0

    return mask


def main():
    parser = argparse.ArgumentParser(
        description="SegFormer mask generation + optional Gaussian splat filtering"
    )
    # Mask generation args
    parser.add_argument("--input", "-i", required=True, help="Input directory containing images")
    parser.add_argument("--output", "-o", required=True, help="Output directory for masks")
    parser.add_argument("--classes", "-c", default=DEFAULT_CLASSES,
                        help=f"Comma-separated class names to mask (default: {DEFAULT_CLASSES})")
    parser.add_argument("--extensions", default="png,jpg,jpeg",
                        help="Comma-separated image extensions to process")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--model", default=MODEL_CHECKPOINT,
                        help=f"SegFormer model checkpoint (default: {MODEL_CHECKPOINT})")
    # PLY filtering args (optional — triggers Clean-GS stage)
    parser.add_argument("--ply", default=None,
                        help="Path to Gaussian splat PLY (enables Clean-GS filtering)")
    parser.add_argument("--cameras", default=None,
                        help="Path to cameras.json (required with --ply)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to training checkpoint (optional, filtered alongside PLY)")
    parser.add_argument("--filter-views", type=int, default=50,
                        help="Number of views for filtering (default: 50)")
    parser.add_argument("--filter-min-views", type=int, default=2,
                        help="Min views a Gaussian must appear in (default: 2)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    extensions = tuple(f".{ext.strip().lower()}" for ext in args.extensions.split(","))
    total_steps = 2 if args.ply else 1

    # Validate PLY filtering args
    if args.ply and not args.cameras:
        print("Error: --cameras is required when --ply is provided")
        sys.exit(1)

    # Parse class names to indices
    class_names = [c.strip().lower() for c in args.classes.split(",")]
    class_indices = []
    for name in class_names:
        if name in ADE20K_CLASSES:
            class_indices.append(ADE20K_CLASSES[name])
        else:
            print(f"Warning: Unknown class '{name}', skipping. Available: {list(ADE20K_CLASSES.keys())}")

    if not class_indices:
        print("Error: No valid classes specified")
        sys.exit(1)

    # Validate input
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    images = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in extensions])

    if not images:
        print(f"Error: No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images to process")
    print(f"Classes to mask: {class_names} -> indices {class_indices}")
    print(f"Output: {output_dir}")
    if args.ply:
        print(f"PLY filtering: {args.ply}")
    print()

    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Using device: {device}")

    # Load model
    processor, model = load_model(device, args.model)

    # ── Stage 1: Generate masks ───────────────────────────────
    progress("mask_generation", "running", 0, step=1, total_steps=total_steps)
    print("\nProcessing images...")

    for idx, image_path in enumerate(tqdm(images, desc="Generating masks")):
        try:
            mask = process_image(image_path, processor, model, class_indices, device)
            output_path = output_dir / f"{image_path.stem}.png"
            cv2.imwrite(str(output_path), mask)

            # Update progress per image
            pct = round((idx + 1) / len(images) * 100)
            progress("mask_generation", "running", pct,
                     detail=f"Image {idx + 1}/{len(images)}",
                     step=1, total_steps=total_steps)

        except Exception as e:
            print(f"\nError processing {image_path.name}: {e}")
            continue

    masks_created = len(list(output_dir.glob("*.png")))
    print(f"\nMasks: {masks_created}/{len(images)} created in {output_dir}")
    progress("mask_generation", "completed", 100, step=1, total_steps=total_steps)

    # ── Stage 2: Filter PLY (if --ply provided) ──────────────
    if args.ply:
        filtered_ply = Path(args.ply).parent / "point_cloud_filtered.ply"

        filter_cmd = [
            "python", "-u", FILTER_SCRIPT,
            "--ply", str(args.ply),
            "--cameras", str(args.cameras),
            "--masks", str(output_dir),
            "--output", str(filtered_ply),
            "--skip-color-validation",
            "--num-views", str(args.filter_views),
            "--min-views", str(args.filter_min_views),
        ]

        if args.checkpoint:
            filtered_ckpt = Path(args.checkpoint).parent / (
                Path(args.checkpoint).stem + "_filtered.pth"
            )
            filter_cmd.extend([
                "--checkpoint", str(args.checkpoint),
                "--checkpoint-output", str(filtered_ckpt),
            ])

        run_with_progress(filter_cmd, "gaussian_filtering",
                          step=2, total_steps=total_steps)

    progress("done", "completed", 100, step=total_steps, total_steps=total_steps)


if __name__ == "__main__":
    main()
