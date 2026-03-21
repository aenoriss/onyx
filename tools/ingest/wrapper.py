#!/usr/bin/env python3
"""Onyx ingest wrapper — video frame extraction with auto-detection.

Auto-detects 360 equirectangular video from metadata and aspect ratio.
Computes optimal frame extraction interval from target image count.

The orchestrator passes high-level intent (target images, scene type).
This wrapper handles all video analysis and extraction details.

Usage:
    # Auto-detect everything (orchestrator mode)
    python wrapper.py --input /data/input_video.mp4 --output /data/images \
        --target-images 300

    # Force 360 mode
    python wrapper.py --input /data/input_video.mp4 --output /data/images \
        --target-images 300 --video-type 360

    # Override interval directly
    python wrapper.py --input /data/input_video.mp4 --output /data/images \
        --video-type 360 --interval 1.4
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

from pipeline_progress import progress, run_with_progress

EVAL_MANIFEST = ".eval_frames.json"

# Approximate smartphone sensor widths for focal length conversion
SENSOR_WIDTHS_MM = {
    # 1x main: ~26mm equiv → sensor ~6.86mm
    # 0.5x ultrawide: ~13mm equiv → sensor ~4.25mm
    # 2x telephoto: ~52mm equiv → sensor ~4.80mm
    # 3x telephoto: ~77mm equiv → sensor ~3.50mm
    "default": 6.17,  # generic smartphone 1/2.55" sensor
}


def focal_mm_to_px(focal_mm, image_width_px, sensor_width_mm=6.17):
    """Convert focal length from mm to pixels given sensor width."""
    return focal_mm * image_width_px / sensor_width_mm


def extract_video_focal_metadata(video_path, image_width):
    """Extract focal length in mm from video metadata using ffprobe.

    Looks for Apple QuickTime camera focal length tag in format and stream tags.
    Returns focal_mm float or None.
    """
    for show_entries in ["format_tags", "stream_tags"]:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", show_entries,
                 "-of", "json", video_path],
                capture_output=True, text=True, timeout=10,
            )
            data = json.loads(result.stdout)
            # Tags may be nested under "format" or "streams"
            if show_entries == "format_tags":
                tags = data.get("format", {}).get("tags", {})
            else:
                streams = data.get("streams", [])
                tags = streams[0].get("tags", {}) if streams else {}

            for key in ["com.apple.quicktime.camera.focal_length",
                        "focal_length", "FocalLength"]:
                if key in tags:
                    val = tags[key]
                    if "/" in str(val):
                        num, den = str(val).split("/")
                        return float(num) / float(den)
                    return float(val)
        except Exception:
            continue
    return None
EXTRACTOR_SCRIPT = "/workspace/360Extractor/src/main.py"
CAMERAS_PER_FRAME_360 = 16
PERSON_CLASS_ID = 0  # COCO class 0 = person


def _filter_person_tiles(image_dir, confidence=0.85, min_area_frac=0.01,
                         skip_files=None):
    """Remove tiles where YOLO detects any person.

    Discards any image with a detected person — masks alone don't fully
    prevent artifacts from bilateral grid / camera optimizer confusion.

    skip_files: set of filenames to skip (e.g. eval frames).
    """
    from ultralytics import YOLO

    skip = skip_files or set()
    model = YOLO("yolov8m-seg.pt")

    files = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f not in skip
    ])

    print(f"[PERSON-FILTER] Scanning {len(files)} tiles for persons...")
    deleted = 0
    rejected_dir = os.path.join(image_dir, "..", "rejected_persons")
    os.makedirs(rejected_dir, exist_ok=True)

    for i, filepath in enumerate(files):
        results = model(filepath, verbose=False)
        if not results or not results[0].boxes:
            continue

        boxes = results[0].boxes
        img_h, img_w = results[0].orig_shape
        img_area = img_h * img_w

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id != PERSON_CLASS_ID or conf < confidence:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_area = (x2 - x1) * (y2 - y1)
            if box_area / img_area >= min_area_frac:
                fname = os.path.basename(filepath)
                shutil.move(filepath, os.path.join(rejected_dir, fname))
                print(f"[PERSON-FILTER] {fname}: conf={conf:.2f} area={box_area/img_area*100:.1f}%")
                deleted += 1
                break

        if (i + 1) % 100 == 0:
            print(f"[PERSON-FILTER] {i+1}/{len(files)} scanned, {deleted} removed")

    print(f"[PERSON-FILTER] Done: removed {deleted}/{len(files)} tiles with persons")


def _select_eval_frames(scores, to_delete, output_dir, target_eval_pct=0.12):
    """Select eval frames from discarded set for DiFix progressive shift.

    Targets ~12% eval holdout (matching NVIDIA's test_every=8 = 12.5%).
    Picks evenly-distributed discarded frames from gaps between training
    frames. All tiles of a frame are kept together (360-aware).

    Saves manifest to {output_dir}/../.eval_frames.json for train.py to read.
    Returns list of eval frame file paths (to exclude from deletion).
    """
    delete_set = set(to_delete)
    all_paths = set(s.path for s in scores)
    kept_paths = all_paths - delete_set

    # Group all tiles by frame
    frames = {}
    for s in scores:
        frames.setdefault(s.frame_idx, []).append(s)

    kept_frame_idxs = sorted(set(
        s.frame_idx for s in scores if s.path in kept_paths
    ))
    discarded_frame_idxs = sorted(set(
        s.frame_idx for s in scores if s.path in delete_set
    ))

    if len(kept_frame_idxs) < 2 or not discarded_frame_idxs:
        print("[EVAL] Not enough frames for eval selection")
        return []

    # Target ~12% eval frames, evenly distributed across gaps
    n_gaps = len(kept_frame_idxs) - 1
    n_target_eval = max(3, round(len(kept_frame_idxs) * target_eval_pct))
    eval_every_n = max(1, n_gaps // n_target_eval) if n_target_eval > 0 else n_gaps

    eval_frame_idxs = []
    for i in range(n_gaps):
        # Pick evenly spaced gaps
        if i % eval_every_n != eval_every_n // 2:
            continue
        if len(eval_frame_idxs) >= n_target_eval:
            break

        # Pick discarded frame closest to a training frame (not midpoint).
        # Matches NVIDIA's test_every=8 pattern: eval cameras are near
        # training cameras, minimizing parallax for DiFix ref images.
        anchor = kept_frame_idxs[i]
        best_idx = None
        best_dist = float("inf")
        for d_idx in discarded_frame_idxs:
            if kept_frame_idxs[i] < d_idx < kept_frame_idxs[i + 1]:
                dist = abs(d_idx - anchor)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = d_idx
        if best_idx is not None:
            eval_frame_idxs.append(best_idx)

    if not eval_frame_idxs:
        print("[EVAL] No suitable eval frames found in gaps")
        return []

    # Collect all tile paths for eval frames
    eval_filenames = []
    eval_paths = []
    for fidx in eval_frame_idxs:
        for tile in frames[fidx]:
            eval_filenames.append(tile.filename)
            eval_paths.append(tile.path)

    # Write manifest
    manifest_path = os.path.join(os.path.dirname(output_dir), EVAL_MANIFEST)
    manifest = {
        "eval_filenames": eval_filenames,
        "eval_frame_indices": eval_frame_idxs,
        "n_eval_frames": len(eval_frame_idxs),
        "n_eval_tiles": len(eval_filenames),
        "n_train_frames": len(kept_frame_idxs),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[EVAL] Selected {len(eval_frame_idxs)} eval frames "
          f"({len(eval_filenames)} tiles) from gaps between training frames")
    print(f"[EVAL] Manifest → {manifest_path}")

    return eval_paths


def flatten_processed_subdir(output_dir):
    """Move images from {output_dir}/*_processed/ into {output_dir}/ directly.

    The 360Extractor always creates a {video_name}_processed/ subfolder instead of
    writing directly to the requested output directory. InstantSfM (ins-feat) expects
    images directly in the images/ directory, so we flatten the structure after extraction.
    """
    try:
        entries = os.listdir(output_dir)
    except OSError:
        return

    processed_dirs = [
        e for e in entries
        if e.endswith("_processed") and os.path.isdir(os.path.join(output_dir, e))
    ]
    if not processed_dirs:
        return

    for subdir_name in processed_dirs:
        src = os.path.join(output_dir, subdir_name)
        moved = 0
        for fname in os.listdir(src):
            shutil.move(os.path.join(src, fname), os.path.join(output_dir, fname))
            moved += 1
        try:
            os.rmdir(src)
        except OSError:
            pass
        print(f"[INFO] Flattened {subdir_name}/ → images/ ({moved} files)")


def get_video_duration(path):
    """Get video duration in seconds using ffprobe.

    Tries format-level duration first (fast), then stream-level fallback.
    Some cameras (DJI, GoPro) store duration only in stream tags, not the
    format header, causing format=duration to return an empty string.
    """
    probe_attempts = [
        # Standard: duration in format container header
        ["-show_entries", "format=duration", "-of", "csv=p=0"],
        # Fallback: duration on the first video stream
        ["-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "csv=p=0"],
    ]
    for probe_args in probe_attempts:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet"] + probe_args + [path],
                capture_output=True, text=True, timeout=10,
            )
            val = result.stdout.strip().split("\n")[0].strip()
            if val and val not in ("N/A", ""):
                return float(val)
        except Exception as e:
            print(f"[WARN] ffprobe duration failed: {e}")
    return None


def get_video_dimensions(path):
    """Get video width and height using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "json", path],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception:
        return None, None


def detect_360(path):
    """Detect if video is 360 equirectangular.

    Checks:
    1. Spherical metadata (Google Spatial Media spec) — definitive
    2. Aspect ratio = 2:1 — strong heuristic (equirectangular is always 2:1)

    Returns: True (360), False (standard), or None (uncertain)
    """
    # Check for spherical metadata
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries",
             "stream_side_data=projection", "-of", "json", path],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout.lower()
        if "equirectangular" in output or "spherical" in output:
            return True
    except Exception:
        pass

    # Also check format tags for spherical metadata
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries",
             "format_tags", "-of", "json", path],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout.lower()
        if "spherical" in output or "equirectangular" in output:
            return True
    except Exception:
        pass

    # Aspect ratio heuristic: 2:1 = equirectangular
    # Note: some 360 cameras output 16:9 (1.778) equirectangular — don't False-match those.
    # Only return False for clearly-portrait or square videos (< 1.5).
    w, h = get_video_dimensions(path)
    if w and h:
        ratio = w / h
        if abs(ratio - 2.0) < 0.05:  # within 2.5% of 2:1
            return True
        if ratio < 1.5:  # portrait or square — definitely not equirectangular
            return False

    return None


def compute_interval(duration, target_images, is_360):
    """Compute frame extraction interval from target image count."""
    if is_360:
        source_frames = max(1, target_images // CAMERAS_PER_FRAME_360)
    else:
        source_frames = target_images
    interval = duration / source_frames
    return round(interval, 2)


def main():
    parser = argparse.ArgumentParser(description="Onyx ingest wrapper")
    parser.add_argument("--input", "-i", required=True, help="Input video file")
    parser.add_argument("--output", "-o", required=True, help="Output image directory")
    parser.add_argument("--target-images", type=int, default=None,
                        help="Target number of output images (auto-calculates interval)")
    parser.add_argument("--video-type", default="auto", choices=["360", "normal", "auto"],
                        help="Video type: 360, normal, or auto-detect (default: auto)")
    parser.add_argument("--interval", type=float, default=None,
                        help="Override: frame extraction interval in seconds")
    parser.add_argument("--layout", default="colmap",
                        choices=["ring", "cube", "fibonacci", "colmap"],
                        help="Camera layout for 360 mode (default: colmap)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Output tile resolution (default: auto from source, no upsampling)")
    parser.add_argument("--format", default="jpg", choices=["jpg", "png"],
                        help="Output image format (default: jpg)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality (default: 95)")
    parser.add_argument("--holdout", action="store_true",
                        help="Select eval holdout frames from discards for quality metrics")
    args = parser.parse_args()

    progress("detecting_video", "running", 0, step=1, total_steps=1)

    # ── Detect video type ──────────────────────────────────────
    if args.video_type == "auto":
        detected = detect_360(args.input)
        if detected is True:
            is_360 = True
            print("[AUTO-DETECT] 360 equirectangular video detected")
        elif detected is False:
            is_360 = False
            print("[AUTO-DETECT] Standard video detected")
        else:
            is_360 = False
            print("[AUTO-DETECT] Could not determine video type, assuming standard")
    else:
        is_360 = (args.video_type == "360")
        print(f"[VIDEO TYPE] {args.video_type} (user specified)")

    # Write detected type for orchestrator to read back
    detection_file = os.path.join(os.path.dirname(args.output), ".video_type")
    try:
        with open(detection_file, "w", encoding="utf-8") as f:
            f.write("360" if is_360 else "normal")
    except Exception:
        pass

    # ── Compute interval ───────────────────────────────────────
    smart_select = False
    if args.interval:
        interval = args.interval
        print(f"[INTERVAL] {interval}s (user override)")
    elif args.target_images:
        duration = get_video_duration(args.input)
        if duration is None:
            print("[ERROR] Could not determine video duration")
            sys.exit(1)

        # Smart selection: over-extract at 1.6fps (1 frame every 0.625s)
        # Denser sampling gives more candidates for smart selection.
        # Matches the effective extraction rate of a 1.6x slowed video at 1fps:
        # slowed video stretches time → 1fps captures every 0.625s of real footage.
        interval = 0.625  # every 0.625s — replicates 1.6x slow video effect
        cameras = CAMERAS_PER_FRAME_360 if is_360 else 1
        target_frames = max(1, args.target_images // cameras)
        oversampled_frames = int(duration / interval)
        print(f"[SMART] Over-extracting at 1fps ({oversampled_frames} frames), "
              f"will select best {target_frames} frames "
              f"(target {args.target_images} images)")
        smart_select = True
    else:
        interval = 1.0
        print(f"[INTERVAL] {interval}s (default)")

    mode = "360" if is_360 else "standard"

    # ── Auto-calculate tile resolution to avoid upsampling ────
    # For 90° FOV tiles from equirectangular, optimal = src_width / 4.
    # Upsampling beyond source density creates fake blur via interpolation.
    resolution = args.resolution
    if resolution is None:
        src_w, src_h = get_video_dimensions(args.input)
        if is_360:
            if src_w:
                # 90° FOV = 1/4 of 360° → max useful pixels = src_w / 4
                native_res = src_w // 4
                # Round down to nearest multiple of 64 for codec friendliness
                native_res = (native_res // 64) * 64
                resolution = native_res
                print(f"[AUTO-RES] Source {src_w}px → tile {resolution}px "
                      f"(native for 90° FOV, no upsampling)")
            else:
                resolution = 2048
                print(f"[AUTO-RES] Could not detect source resolution, using {resolution}px")
        else:
            if src_w:
                # Standard mode: use native source width, no downscale
                resolution = (src_w // 64) * 64
                print(f"[AUTO-RES] Source {src_w}x{src_h}px → extracting at {resolution}px wide (native)")
            else:
                resolution = 0  # 0 = no resize in extractor
                print(f"[AUTO-RES] Could not detect source resolution, extracting at native size")

    print(f"\n{'='*60}")
    print(f"Ingest: {mode} extraction")
    print(f"  Input:      {args.input}")
    print(f"  Output:     {args.output}")
    print(f"  Interval:   {interval}s")
    print(f"  Resolution: {resolution}px")
    print(f"  Format:     {args.format}")
    if is_360:
        print(f"  Layout:     {args.layout}")
    print(f"{'='*60}\n")

    # ── Run extraction ─────────────────────────────────────────
    progress("extracting_frames", "running", 0, step=1, total_steps=1)

    cmd = [
        "python", "-u", EXTRACTOR_SCRIPT,
        "--input", args.input,
        "--output", args.output,
        "--mode", mode,
        "--interval", str(interval),
        "--resolution", str(resolution),
        "--format", args.format,
        "--quality", str(args.quality),
    ]

    if is_360:
        cmd.extend(["--layout", args.layout])

    patterns = {
        r"\[(\d+)%\]": lambda m: (
            int(m.group(1)),
            f"Frame extraction {m.group(1)}%",
        ),
        r"(\d+)/(\d+)%": lambda m: (
            int(m.group(1)),
            f"Frame extraction {m.group(1)}%",
        ),
    }

    run_with_progress(cmd, "extracting_frames",
                      step=1, total_steps=1, patterns=patterns)

    # The 360Extractor creates a {video_name}_processed/ subfolder inside the output
    # directory rather than writing images directly to it. Flatten it so that
    # InstantSfM (ins-feat) can find the images with a non-recursive os.listdir.
    flatten_processed_subdir(args.output)

    # ── Smart frame/tile selection ────────────────────────────
    if smart_select:
        from frame_selector import score_tiles, select_and_prune

        progress("scoring_tiles", "running", 0, step=1, total_steps=1)
        print("[SMART] Scoring tiles with SIFT features...")
        scores = score_tiles(args.output)
        # max_gap must satisfy two constraints:
        # 1. Time-based: ~8 seconds of real camera movement between force-keeps
        # 2. Budget-based: gap must be large enough to allow reaching target frame count
        # Use the larger to ensure budget is always achievable.
        time_gap = max(8, round(8.0 / interval))
        target_frames_est = max(1, args.target_images // cameras)
        budget_gap = max(8, oversampled_frames // target_frames_est)
        adaptive_max_gap = max(time_gap, budget_gap)
        to_delete = select_and_prune(scores, target_count=args.target_images,
                                     max_gap=adaptive_max_gap)

        # ── Save eval frames (only with --holdout) ─────────────
        eval_paths = []
        if args.holdout:
            eval_paths = _select_eval_frames(scores, to_delete, args.output)

        # Delete discarded frames (except eval frames)
        eval_set = set(eval_paths)
        n_deleted = 0
        for path in to_delete:
            if path not in eval_set:
                os.remove(path)
                n_deleted += 1

        remaining = len(scores) - n_deleted
        print(f"[SMART] Scored {len(scores)} tiles, kept {remaining} "
              f"(train + {len(eval_paths)} eval), deleted {n_deleted}")

    # ── Post-selection person filtering (YOLO nano) ────────────
    # Run on ALL surviving tiles including eval. Eval frames with persons
    # would artificially reduce quality metrics (model never sees persons,
    # but GT image has them → PSNR/SSIM tanks). Remove affected eval
    # frames from manifest after filtering.
    if is_360:
        progress("person_filter", "running", 0, step=1, total_steps=1)
        _filter_person_tiles(args.output)

        # Update eval manifest: remove eval frames deleted by person filter
        if smart_select and eval_paths:
            _surviving = [p for p in eval_paths if os.path.exists(p)]
            _removed = len(eval_paths) - len(_surviving)
            if _removed > 0:
                eval_paths = _surviving
                # Rewrite manifest with surviving eval frames only
                manifest_path = os.path.join(
                    os.path.dirname(args.output), EVAL_MANIFEST)
                if os.path.exists(manifest_path):
                    _surviving_names = [os.path.basename(p) for p in _surviving]
                    manifest = {
                        "eval_filenames": _surviving_names,
                        "n_eval_tiles": len(_surviving_names),
                    }
                    with open(manifest_path, "w") as mf:
                        json.dump(manifest, mf, indent=2)
                print(f"[EVAL] Removed {_removed} eval tiles with persons, "
                      f"{len(_surviving)} remaining")

    # ── Extract camera focal length metadata ────────────────
    focal_mm = extract_video_focal_metadata(args.input, resolution)
    if focal_mm is not None:
        sensor_w = SENSOR_WIDTHS_MM["default"]
        focal_px = focal_mm_to_px(focal_mm, resolution, sensor_w)
        metadata = {
            "cameras": {
                "default": {
                    "focal_mm": focal_mm,
                    "sensor_width_mm": sensor_w,
                    "focal_px": focal_px,
                    "image_width_px": resolution,
                    "source": "ffprobe"
                }
            }
        }
        meta_path = os.path.join(args.output, ".camera_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[METADATA] Saved focal length: {focal_mm}mm → {focal_px:.1f}px")

    progress("done", "completed", 100, step=1, total_steps=1)


if __name__ == "__main__":
    main()
