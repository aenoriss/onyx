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

EXTRACTOR_SCRIPT = "/workspace/360Extractor/src/main.py"
CAMERAS_PER_FRAME_360 = 12


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
    w, h = get_video_dimensions(path)
    if w and h:
        ratio = w / h
        if abs(ratio - 2.0) < 0.05:  # within 2.5% of 2:1
            return True
        if ratio < 1.8:  # clearly not 2:1
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
    parser.add_argument("--resolution", type=int, default=2048,
                        help="Output image resolution (default: 2048)")
    parser.add_argument("--format", default="jpg", choices=["jpg", "png"],
                        help="Output image format (default: jpg)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality (default: 95)")
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
    if args.interval:
        interval = args.interval
        print(f"[INTERVAL] {interval}s (user override)")
    elif args.target_images:
        duration = get_video_duration(args.input)
        if duration is None:
            print("[ERROR] Could not determine video duration")
            sys.exit(1)
        interval = compute_interval(duration, args.target_images, is_360)
        cameras = CAMERAS_PER_FRAME_360 if is_360 else 1
        source_frames = max(1, args.target_images // cameras)
        print(f"[INTERVAL] {interval}s (from {duration:.1f}s video, "
              f"target {args.target_images} images, "
              f"{source_frames} source frames × {cameras} cameras)")
    else:
        interval = 1.0
        print(f"[INTERVAL] {interval}s (default)")

    mode = "360" if is_360 else "standard"

    print(f"\n{'='*60}")
    print(f"Ingest: {mode} extraction")
    print(f"  Input:    {args.input}")
    print(f"  Output:   {args.output}")
    print(f"  Interval: {interval}s")
    if is_360:
        print(f"  Layout:   {args.layout}")
    print(f"{'='*60}\n")

    # ── Run extraction ─────────────────────────────────────────
    progress("extracting_frames", "running", 0, step=1, total_steps=1)

    cmd = [
        "python", "-u", EXTRACTOR_SCRIPT,
        "--input", args.input,
        "--output", args.output,
        "--mode", mode,
        "--interval", str(interval),
        "--resolution", str(args.resolution),
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

    progress("done", "completed", 100, step=1, total_steps=1)


if __name__ == "__main__":
    main()
