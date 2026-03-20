#!/usr/bin/env python3
"""Onyx InstantSfM wrapper — feature extraction + SfM with progress.

Orchestrates two sequential tools (ins-feat -> ins-sfm) with per-stage
progress tracking. Supports --colmap-mapper flag to use COLMAP's incremental
mapper instead of InstantSfM's global mapper (better for cubemap/multi-camera
data where global SfM fragments the view graph).

Usage:
    python wrapper.py --data_path /data
    python wrapper.py --data_path /data --colmap-mapper
"""

import argparse
import os
import sys

from pipeline_progress import progress, run_with_progress


def main():
    parser = argparse.ArgumentParser(description="Onyx InstantSfM wrapper")
    parser.add_argument("--data_path", required=True,
                        help="Path to data directory (images/ must exist)")
    parser.add_argument("--colmap-mapper", action="store_true", default=True,
                        help="Use COLMAP incremental mapper (default). "
                             "Disable with --no-colmap-mapper for InstantSfM global mapper.")
    parser.add_argument("--no-colmap-mapper", dest="colmap_mapper", action="store_false",
                        help="Use InstantSfM global mapper instead of COLMAP")
    parser.add_argument("--video-type", choices=["normal", "360"], default="normal",
                        help="Video type: normal uses automatic intrinsics, 360 uses fixed 90° FOV")
    args = parser.parse_args()

    total_steps = 2

    # ── Stage 1: Feature extraction + matching (always COLMAP) ──
    feat_patterns = {
        r"(\d+)\s*/\s*(\d+)\s*images": lambda m: (
            round(int(m.group(1)) / int(m.group(2)) * 100),
            f"Image {m.group(1)}/{m.group(2)}",
        ),
    }

    run_with_progress(
        ["ins-feat", "--data_path", args.data_path, "--single_camera", "--video-type", args.video_type],
        stage="feature_extraction",
        step=1, total_steps=total_steps,
        patterns=feat_patterns,
    )

    if args.colmap_mapper:
        # ── Stage 2: COLMAP incremental mapper ──────────────────
        # Reuses features + matches from ins-feat (database.db)
        # Much more robust for cubemap/multi-camera data
        db_path = os.path.join(args.data_path, "database.db")
        image_path = os.path.join(args.data_path, "images")
        sparse_path = os.path.join(args.data_path, "sparse")
        os.makedirs(sparse_path, exist_ok=True)

        mapper_patterns = {
            r"Registering image #(\d+) \((\d+)\)": lambda m: (
                None,
                f"registered {m.group(2)} images",
            ),
            r"(\d+) / (\d+) images": lambda m: (
                round(int(m.group(1)) / int(m.group(2)) * 100),
                f"{m.group(1)}/{m.group(2)} images",
            ),
        }

        run_with_progress(
            [
                "colmap", "mapper",
                "--database_path", db_path,
                "--image_path", image_path,
                "--output_path", sparse_path,
                "--Mapper.ba_global_max_num_iterations", "30",
                "--Mapper.ba_global_max_refinements", "3",
            ],
            stage="sfm",
            step=2, total_steps=total_steps,
            patterns=mapper_patterns,
        )

        # COLMAP mapper outputs to sparse/0/, sparse/1/, etc.
        # Keep only the largest reconstruction (most images)
        best_model = None
        best_count = 0
        for model_dir in sorted(os.listdir(sparse_path)):
            model_path = os.path.join(sparse_path, model_dir)
            if not os.path.isdir(model_path):
                continue
            images_bin = os.path.join(model_path, "images.bin")
            if os.path.exists(images_bin):
                size = os.path.getsize(images_bin)
                if size > best_count:
                    best_count = size
                    best_model = model_dir

        if best_model and best_model != "0":
            # Move best model to sparse/0/
            import shutil
            best_path = os.path.join(sparse_path, best_model)
            target = os.path.join(sparse_path, "0")
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.move(best_path, target)
            print(f"[COLMAP] Kept model {best_model} as sparse/0/")

        # Clean up extra models
        for model_dir in os.listdir(sparse_path):
            model_path = os.path.join(sparse_path, model_dir)
            if os.path.isdir(model_path) and model_dir != "0":
                import shutil
                shutil.rmtree(model_path)

        # For normal video, undistort SIMPLE_RADIAL → PINHOLE so downstream
        # tools (OpenMVS InterfaceCOLMAP) can read the cameras directly.
        # 360 uses hardcoded PINHOLE already — no undistortion needed.
        if args.video_type == "normal":
            undistorted_path = os.path.join(args.data_path, "undistorted")
            if not os.path.exists(undistorted_path):
                print("[SFM] Undistorting cameras (SIMPLE_RADIAL → PINHOLE)...")
                run_with_progress(
                    [
                        "colmap", "image_undistorter",
                        "--image_path", image_path,
                        "--input_path", os.path.join(sparse_path, "0"),
                        "--output_path", undistorted_path,
                        "--output_type", "COLMAP",
                    ],
                    stage="undistort",
                    step=2, total_steps=total_steps,
                )
                print(f"[SFM] Undistorted cameras written to undistorted/")

    else:
        # ── Stage 2: InstantSfM global mapper ──────────────────
        sfm_patterns = {
            r"Running\s+(\w+)": lambda m: (
                None,
                m.group(1),
            ),
        }

        run_with_progress(
            ["ins-sfm", "--data_path", args.data_path],
            stage="sfm",
            step=2, total_steps=total_steps,
            patterns=sfm_patterns,
        )

    progress("done", "completed", 100, step=2, total_steps=total_steps)


if __name__ == "__main__":
    main()
