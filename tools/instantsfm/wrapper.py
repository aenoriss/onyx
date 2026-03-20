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
    parser.add_argument("--single-camera", action="store_true", default=False,
                        help="Force single shared camera model (default for normal video from one physical camera).")
    parser.add_argument("--cameras", nargs='+', default=None,
                        help="Subfolder names for multi-camera input (e.g. --cameras wide ultrawide). "
                             "Images must be pre-sorted into images/<name>/ subfolders. "
                             "Enables per-folder intrinsics — much faster than per-image.")
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
        ["ins-feat", "--data_path", args.data_path, "--video-type", args.video_type]
        + (["--cameras"] + args.cameras if args.cameras else ["--single_camera"]),
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

            # Promote undistorted data as main paths so all downstream tools
            # (MILo, OpenMVS) get PINHOLE cameras and undistorted images:
            #   sparse/0/  <- undistorted/sparse/  (PINHOLE replaces SIMPLE_RADIAL)
            #   images/    <- flat dir of all undistorted images (MILo needs flat)
            #   images.bin <- patched to strip subfolder prefixes (e.g. cam1/)
            import shutil, struct, os as _os

            # 1. Copy undistorted PINHOLE sparse → sparse/0/
            sparse0 = _os.path.join(sparse_path, "0")
            undist_sparse = _os.path.join(undistorted_path, "sparse")
            for f in _os.listdir(undist_sparse):
                shutil.copy2(_os.path.join(undist_sparse, f), _os.path.join(sparse0, f))
            print("[SFM] sparse/0/ updated with PINHOLE cameras")

            # 2. Flatten undistorted/images/**  →  images/ (prefixed by subfolder)
            #    MILo uses os.path.basename(image_name) so needs flat layout.
            #    Prefix each file with its camera subfolder name to avoid collisions
            #    when multiple cameras share identical filenames (e.g. cam1 + cam2).
            #    e.g. undistorted/images/cam1/frame.jpg → images/cam1_frame.jpg
            undist_images = _os.path.join(undistorted_path, "images")
            old_images = _os.path.join(args.data_path, "images")
            if not _os.path.islink(old_images) and _os.path.isdir(old_images):
                _os.rename(old_images, old_images + "_cam_orig")
            elif _os.path.islink(old_images):
                _os.unlink(old_images)
            _os.makedirs(old_images, exist_ok=True)
            # Build mapping: original relative path → flat destination name
            path_to_flat = {}
            for root, _dirs, files in _os.walk(undist_images):
                rel_dir = _os.path.relpath(root, undist_images)
                for fname in files:
                    if rel_dir == ".":
                        flat_name = fname
                    else:
                        # Use top-level subfolder as prefix: cam1/a/b/f.jpg → cam1_f.jpg
                        top = rel_dir.split(_os.sep)[0]
                        flat_name = top + "_" + fname
                    src = _os.path.join(root, fname)
                    dst = _os.path.join(old_images, flat_name)
                    path_to_flat[_os.path.join(rel_dir, fname) if rel_dir != "." else fname] = flat_name
                    shutil.copy2(src, dst)
            flat_count = len(_os.listdir(old_images))
            print(f"[SFM] images/ flattened — {flat_count} undistorted images (copied)")

            # 3. Patch sparse/0/images.bin — rewrite names to match flat filenames
            #    Original names are like "cam1/frame.jpg"; map to "cam1_frame.jpg"
            images_bin = _os.path.join(sparse0, "images.bin")
            with open(images_bin, "rb") as f:
                data = f.read()
            out = bytearray()
            i = 0
            n_imgs = struct.unpack_from("<Q", data, i)[0]; i += 8
            out += struct.pack("<Q", n_imgs)
            for _ in range(n_imgs):
                header = data[i:i+64]; i += 64  # id + qvec + tvec + camera_id
                out += header
                name_start = i
                while data[i:i+1] != b'\x00':
                    i += 1
                orig_name = data[name_start:i].decode()
                i += 1  # skip null
                # Map original "cam1/frame.jpg" → "cam1_frame.jpg"
                parts = orig_name.replace("\\", "/").split("/")
                if len(parts) > 1:
                    flat_name = parts[0] + "_" + parts[-1]
                else:
                    flat_name = orig_name
                out += flat_name.encode() + b'\x00'
                n2d = struct.unpack_from("<Q", data, i)[0]; i += 8
                out += struct.pack("<Q", n2d) + data[i:i + n2d * 24]
                i += n2d * 24
            with open(images_bin, "wb") as f:
                f.write(out)
            print(f"[SFM] images.bin patched — prefixed subfolder names ({n_imgs} images)")

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
