"""Score and select best tiles from over-extracted candidate set.

For 360/multi-camera data, uses a fixed reference camera to measure visual
overlap between consecutive frames — comparing the SAME viewpoint across time.
This correctly measures camera movement rather than comparing unrelated
camera directions (which always appear different regardless of movement).

For standard video (single camera per frame), falls back to direct comparison.

Keeps frames where camera has moved enough (overlap drops below threshold),
ensuring spatial coverage rather than uniform time sampling.
"""

import cv2
import os
import re
import sys
import time
from dataclasses import dataclass

import numpy as np

SCORE_RESOLUTION = 512  # downscale for fast SIFT scoring
OVERLAP_RESOLUTION = 256  # downscale for overlap matching (speed)

# Pattern: {video}_frame{XXXXXX}_{camera}.jpg
TILE_PATTERN = re.compile(r".*_frame(\d+)_(.+)\.(jpg|png)$", re.IGNORECASE)
# Generic image extensions for non-tile (photo folder) input
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

# FLANN matcher params for SIFT
FLANN_INDEX_KDTREE = 1
FLANN_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
FLANN_SEARCH = dict(checks=50)
LOWE_RATIO = 0.7  # Lowe's ratio test threshold


@dataclass
class TileScore:
    path: str           # full path to saved image
    filename: str
    frame_idx: int      # parsed from filename
    camera: str         # parsed from filename (e.g. "Front", "BackLeft_Up")
    sift_count: int     # SIFT keypoint count
    sharpness: float    # Laplacian variance


def score_tiles(image_dir: str) -> list[TileScore]:
    """Score all tiles with SIFT keypoint count + sharpness."""
    sift = cv2.SIFT_create()
    scores = []

    files = sorted(os.listdir(image_dir))
    tile_files = [f for f in files if TILE_PATTERN.match(f)]
    total = len(tile_files)
    print(f"[SCORE] Scoring {total} tiles at {SCORE_RESOLUTION}px...")
    t0 = time.time()

    for i, fname in enumerate(tile_files):
        match = TILE_PATTERN.match(fname)
        frame_idx = int(match.group(1))
        camera = match.group(2)
        filepath = os.path.join(image_dir, fname)

        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Downscale for speed
        h, w = img.shape[:2]
        if max(h, w) > SCORE_RESOLUTION:
            scale = SCORE_RESOLUTION / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        keypoints = sift.detect(img, None)
        sift_count = len(keypoints)
        sharpness = cv2.Laplacian(img, cv2.CV_64F).var()

        scores.append(TileScore(
            path=filepath,
            filename=fname,
            frame_idx=frame_idx,
            camera=camera,
            sift_count=sift_count,
            sharpness=sharpness,
        ))

        if (i + 1) % 200 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"[SCORE] {i+1}/{total} ({rate:.0f} tiles/s)")

    elapsed = time.time() - t0
    n_frames = len(set(s.frame_idx for s in scores))
    n_cameras = len(set(s.camera for s in scores))
    print(f"[SCORE] Done in {elapsed:.1f}s — {len(scores)} tiles, "
          f"{n_frames} frames, {n_cameras} cameras")
    if scores:
        sift_counts = [s.sift_count for s in scores]
        print(f"[SCORE] SIFT: {min(sift_counts)}-{max(sift_counts)} "
              f"(mean {sum(sift_counts)/len(sift_counts):.0f})")

    return scores


def score_images(image_dir: str, cameras: list[str] | None = None) -> list[TileScore]:
    """Score generic images (photo folder input) with SIFT + sharpness.

    Unlike score_tiles() which expects video-extracted tile filenames,
    this handles arbitrary image filenames. Each image becomes its own
    "frame" (frame_idx assigned by sort order). For multi-camera input,
    images in each camera subfolder share the same frame_idx by sort position.

    Args:
        image_dir: Directory containing images (or camera subfolders).
        cameras: List of camera subfolder names (e.g. ["ultra", "wide"]).
                 If None, treats all images in image_dir as single-camera.
    """
    sift = cv2.SIFT_create()
    scores = []

    if cameras:
        # Multi-camera: pair images across subfolders by sort position
        cam_files: dict[str, list[str]] = {}
        for cam in cameras:
            cam_dir = os.path.join(image_dir, cam)
            if not os.path.isdir(cam_dir):
                print(f"[SCORE] Warning: camera folder {cam} not found, skipping")
                continue
            cam_files[cam] = sorted([
                f for f in os.listdir(cam_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            ])

        # Build list: (frame_idx, camera, filepath)
        file_list = []
        for cam, files in cam_files.items():
            for idx, fname in enumerate(files):
                filepath = os.path.join(image_dir, cam, fname)
                file_list.append((idx, cam, filepath, fname))
    else:
        # Single camera: each image is its own frame
        files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f))
            and os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ])
        file_list = [(idx, "default", os.path.join(image_dir, f), f)
                     for idx, f in enumerate(files)]

    total = len(file_list)
    if total == 0:
        print("[SCORE] No images found to score")
        return []

    print(f"[SCORE] Scoring {total} images at {SCORE_RESOLUTION}px...")
    t0 = time.time()

    for i, (frame_idx, camera, filepath, fname) in enumerate(file_list):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape[:2]
        if max(h, w) > SCORE_RESOLUTION:
            scale = SCORE_RESOLUTION / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        keypoints = sift.detect(img, None)
        sift_count = len(keypoints)
        sharpness = cv2.Laplacian(img, cv2.CV_64F).var()

        scores.append(TileScore(
            path=filepath,
            filename=fname,
            frame_idx=frame_idx,
            camera=camera,
            sift_count=sift_count,
            sharpness=sharpness,
        ))

        if (i + 1) % 200 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"[SCORE] {i+1}/{total} ({rate:.0f} images/s)")

    elapsed = time.time() - t0
    n_frames = len(set(s.frame_idx for s in scores))
    n_cameras = len(set(s.camera for s in scores))
    print(f"[SCORE] Done in {elapsed:.1f}s — {len(scores)} images, "
          f"{n_frames} frames, {n_cameras} cameras")
    if scores:
        sift_counts = [s.sift_count for s in scores]
        print(f"[SCORE] SIFT: {min(sift_counts)}-{max(sift_counts)} "
              f"(mean {sum(sift_counts)/len(sift_counts):.0f})")

    return scores


def _downscale_gray(img, max_res):
    """Downscale a grayscale image if larger than max_res."""
    h, w = img.shape[:2]
    if max(h, w) > max_res:
        scale = max_res / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def _compute_overlap(img_path_a, img_path_b, sift, flann):
    """Compute visual overlap ratio between two images using SIFT matching.

    Returns overlap score in [0, 1]:
      1.0 = identical views (all features match)
      0.0 = completely different views (no matches)
    """
    img_a = cv2.imread(img_path_a, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(img_path_b, cv2.IMREAD_GRAYSCALE)
    if img_a is None or img_b is None:
        return 0.0

    img_a = _downscale_gray(img_a, OVERLAP_RESOLUTION)
    img_b = _downscale_gray(img_b, OVERLAP_RESOLUTION)

    kp_a, desc_a = sift.detectAndCompute(img_a, None)
    kp_b, desc_b = sift.detectAndCompute(img_b, None)

    if desc_a is None or desc_b is None or len(kp_a) < 5 or len(kp_b) < 5:
        return 0.0

    matches = flann.knnMatch(desc_a, desc_b, k=2)

    # Lowe's ratio test
    good = 0
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < LOWE_RATIO * n.distance:
                good += 1

    # Overlap = fraction of features that matched
    max_features = min(len(kp_a), len(kp_b))
    if max_features == 0:
        return 0.0

    return good / max_features


def _pick_reference_camera(scores):
    """Pick the best camera to use as overlap reference for 360/multi-camera data.

    For 360 video, each frame produces N tiles (one per camera direction).
    Overlap must be measured on the SAME camera across consecutive frames —
    otherwise comparing e.g. Front vs Back always shows low overlap regardless
    of actual camera movement.

    Selection: camera with highest median SIFT count across all frames.
    High SIFT = feature-rich scenes (not sky/ground) = best motion signal.

    Returns None if single-camera data (no reference needed).
    """
    camera_sifts: dict[str, list[int]] = {}
    for s in scores:
        camera_sifts.setdefault(s.camera, []).append(s.sift_count)

    if len(camera_sifts) <= 1:
        return None  # single camera, no reference needed

    # Pick camera with highest median SIFT (robust to outlier frames)
    def median_sift(cam):
        vals = sorted(camera_sifts[cam])
        return vals[len(vals) // 2]

    best_cam = max(camera_sifts, key=median_sift)
    print(f"[SELECT] 360 mode: {len(camera_sifts)} cameras detected")
    print(f"[SELECT] Reference camera: {best_cam} "
          f"(median {median_sift(best_cam)} SIFT, "
          f"{len(camera_sifts[best_cam])} frames)")
    return best_cam


def select_and_prune(
    scores: list[TileScore],
    target_count: int,
    min_sift: int = 40,
    min_frames_keep: int = 10,
    max_overlap: float = 0.45,
    max_gap: int = 8,
) -> list[str]:
    """Select best tiles using visual overlap and return list of paths TO DELETE.

    For 360/multi-camera data: automatically detects multiple cameras per frame,
    picks a consistent reference camera, and compares that same camera across
    consecutive frames to measure actual camera movement.

    For standard video: compares consecutive frames directly.

    Industry standard: COLMAP recommends 60-80% visual overlap between
    consecutive views. Our SIFT match ratio of 0.45 corresponds to roughly
    60-70% visual overlap (SIFT features cluster in textured regions,
    underestimating true overlap).

    Args:
        target_count: Target number of tiles to keep (set by orchestrator)
        max_overlap: Maximum overlap ratio to consider frames redundant (0.45 = 45%)
        max_gap: Force-keep after this many skipped frames (prevents coverage gaps)
    """
    if not scores:
        return []

    n_featureless = sum(1 for s in scores if s.sift_count < min_sift)

    # Filter blurry tiles (Laplacian variance threshold)
    # Sharp images: >100, moderate: 50-100, blurry: <50
    # Use relative threshold: bottom 10% of sharpness distribution
    if scores:
        sharpness_vals = sorted(s.sharpness for s in scores)
        sharpness_p10 = sharpness_vals[len(sharpness_vals) // 10]
        min_sharpness = max(sharpness_p10, 30.0)  # at least 30, or p10
        n_blurry = sum(1 for s in scores if s.sharpness < min_sharpness)
    else:
        min_sharpness = 30.0
        n_blurry = 0

    keep_budget = target_count
    print(f"[SELECT] {len(scores)} total, {n_featureless} featureless "
          f"(< {min_sift} SIFT), {n_blurry} blurry (sharpness < {min_sharpness:.0f})")
    print(f"[SELECT] Budget: {keep_budget} tiles (target {target_count})")

    # Group ALL tiles by frame
    frames: dict[int, list[TileScore]] = {}
    for s in scores:
        frames.setdefault(s.frame_idx, []).append(s)

    sorted_frame_idxs = sorted(frames.keys())
    n_frames_total = len(sorted_frame_idxs)

    if n_frames_total == 0:
        return []

    # ── Pick reference camera for overlap comparison ──────────
    # For 360 data: use a FIXED camera across all frames so we compare
    # the same viewpoint over time (measures actual movement).
    # For standard video: use the single tile per frame (ref_camera=None).
    ref_camera = _pick_reference_camera(scores)
    is_multicam = ref_camera is not None

    # Build per-frame reference tile lookup
    # For 360: the reference camera's tile from each frame
    # For standard: the tile with highest SIFT count (only 1 per frame anyway)
    ref_tiles: dict[int, TileScore] = {}
    if is_multicam:
        # Index: frame_idx → {camera: TileScore}
        frame_cam_map: dict[int, dict[str, TileScore]] = {}
        for s in scores:
            frame_cam_map.setdefault(s.frame_idx, {})[s.camera] = s

        # For each frame, use the reference camera's tile.
        # Fallback: if reference camera missing from a frame, use the camera
        # with the most SIFT features (best available proxy).
        n_fallback = 0
        for fidx in sorted_frame_idxs:
            cam_map = frame_cam_map[fidx]
            if ref_camera in cam_map:
                ref_tiles[fidx] = cam_map[ref_camera]
            else:
                ref_tiles[fidx] = max(cam_map.values(), key=lambda t: t.sift_count)
                n_fallback += 1
        if n_fallback:
            print(f"[SELECT] Warning: reference camera missing from "
                  f"{n_fallback} frames, used fallback")
    else:
        for fidx, tiles in frames.items():
            ref_tiles[fidx] = max(tiles, key=lambda t: t.sift_count)

    # Estimate target frames for overlap pass (rough, using raw tile count).
    # Accurate count computed after per-tile filtering on kept frames only.
    avg_tiles_per_frame = len(scores) / max(n_frames_total, 1)
    target_frames = max(min_frames_keep, int(keep_budget / max(avg_tiles_per_frame, 1)))

    if target_frames >= n_frames_total:
        print(f"[SELECT] Keeping all {len(scores)} tiles "
              f"({n_frames_total} frames, budget {keep_budget})")
        return []

    # ── Overlap-based keyframe selection ──────────────────────
    print(f"[SELECT] Computing visual overlap between {n_frames_total} frames "
          f"(target {target_frames} frames)...")
    t0 = time.time()

    sift = cv2.SIFT_create(nfeatures=200)  # fewer features = faster matching
    flann = cv2.FlannBasedMatcher(FLANN_PARAMS, FLANN_SEARCH)

    # Greedy forward pass: keep frame if it differs enough from last kept frame
    kept_frame_idxs = [sorted_frame_idxs[0]]  # always keep first
    last_kept_ref = ref_tiles[sorted_frame_idxs[0]].path
    frames_since_kept = 0

    for i in range(1, n_frames_total):
        fidx = sorted_frame_idxs[i]
        current_ref = ref_tiles[fidx].path
        frames_since_kept += 1

        # Force-keep if too many frames skipped (coverage guarantee)
        if frames_since_kept >= max_gap:
            kept_frame_idxs.append(fidx)
            last_kept_ref = current_ref
            frames_since_kept = 0
            continue

        overlap = _compute_overlap(last_kept_ref, current_ref, sift, flann)

        if overlap < max_overlap:
            # Camera moved enough — keep this frame
            kept_frame_idxs.append(fidx)
            last_kept_ref = current_ref
            frames_since_kept = 0

    elapsed = time.time() - t0
    print(f"[SELECT] Overlap pass: {len(kept_frame_idxs)}/{n_frames_total} frames "
          f"kept in {elapsed:.1f}s")

    # ── Per-tile filter on KEPT frames only ──────────────────
    # Filter featureless/blurry tiles only from frames selected by overlap.
    # More efficient than filtering all frames (only ~half survive overlap).
    n_featureless_dropped = 0
    n_blurry_dropped = 0
    for fidx in list(kept_frame_idxs):
        surviving = []
        for t in frames[fidx]:
            if t.sift_count < min_sift:
                n_featureless_dropped += 1
            elif t.sharpness < min_sharpness:
                n_blurry_dropped += 1
            else:
                surviving.append(t)
        if surviving:
            frames[fidx] = surviving
        else:
            # Frame has zero surviving tiles — remove from kept
            kept_frame_idxs.remove(fidx)

    if n_featureless_dropped:
        print(f"[SELECT] Filtered {n_featureless_dropped} featureless tiles "
              f"(< {min_sift} SIFT) from kept frames")
    if n_blurry_dropped:
        print(f"[SELECT] Filtered {n_blurry_dropped} blurry tiles "
              f"(sharpness < {min_sharpness:.0f}) from kept frames")

    # ── Motion blur detection on KEPT frames ────────────────
    # Kept frames are spaced apart (10+ seconds), so a blurry frame
    # stands out against sharp neighbors. Checks reference camera tile
    # sharpness across kept frames with a 5-frame sliding window.
    # Rejects frames below 60% of local median → camera jolt.
    _window = 5
    _blur_threshold = 0.6
    _blur_rejected = 0
    if len(kept_frame_idxs) > _window:
        _kept_sharpness = []
        for fidx in kept_frame_idxs:
            # Use reference camera tile, or best surviving tile if ref was filtered
            if fidx in ref_tiles:
                _kept_sharpness.append(ref_tiles[fidx].sharpness)
            elif frames.get(fidx):
                _kept_sharpness.append(max(t.sharpness for t in frames[fidx]))
            else:
                _kept_sharpness.append(0)

        _to_remove = []
        for i in range(len(kept_frame_idxs)):
            lo = max(0, i - _window // 2)
            hi = min(len(kept_frame_idxs), i + _window // 2 + 1)
            if hi - lo < 3:
                continue
            neighbors = sorted(_kept_sharpness[lo:hi])
            local_median = neighbors[len(neighbors) // 2]
            if local_median > 0 and _kept_sharpness[i] < local_median * _blur_threshold:
                _to_remove.append(kept_frame_idxs[i])

        for fidx in _to_remove:
            kept_frame_idxs.remove(fidx)
            _blur_rejected += 1

        if _blur_rejected:
            print(f"[SELECT] Motion blur: rejected {_blur_rejected} blurry kept frames")

    # Recompute accurate tile counts and target after filtering
    kept_tiles_count = sum(len(frames[f]) for f in kept_frame_idxs)
    avg_tiles_per_frame = kept_tiles_count / max(len(kept_frame_idxs), 1)
    target_frames = max(min_frames_keep, int(keep_budget / max(avg_tiles_per_frame, 1)))
    print(f"[SELECT] {kept_tiles_count} tiles across {len(kept_frame_idxs)} frames "
          f"(avg {avg_tiles_per_frame:.1f}/frame, target {target_frames} frames)")

    # ── Budget trim/fill using accurate tile counts ────────

    if kept_tiles_count > keep_budget * 1.3 and len(kept_frame_idxs) > target_frames:
        # Too many frames — score and drop worst frames until within budget
        # Score each kept frame by composite quality
        max_sharpness = max(s.sharpness for s in scores) or 1.0
        frame_quality = {}
        for fidx in kept_frame_idxs:
            tiles = frames[fidx]
            composite = sum(
                t.sift_count * 0.7 + (t.sharpness / max_sharpness) * 1000 * 0.3
                for t in tiles
            ) / len(tiles)
            frame_quality[fidx] = composite

        # Sort by quality × spacing, remove from dense clusters first.
        # Protects isolated frames even if lower quality, drops from
        # clusters where neighbors can compensate. Never removes first/last.
        def _drop_score(fidx):
            """Lower score = more likely to drop."""
            idx_in_kept = kept_frame_idxs.index(fidx)
            gap_before = fidx - kept_frame_idxs[idx_in_kept - 1] if idx_in_kept > 0 else max_gap
            gap_after = kept_frame_idxs[idx_in_kept + 1] - fidx if idx_in_kept < len(kept_frame_idxs) - 1 else max_gap
            min_gap_val = min(gap_before, gap_after)
            spacing = min(min_gap_val / max_gap, 1.0)  # 0=dense cluster, 1=isolated
            quality = frame_quality[fidx] / (max(frame_quality.values()) or 1.0)
            return quality * 0.6 + spacing * 0.4

        removable = sorted(kept_frame_idxs[1:-1], key=_drop_score)
        while kept_tiles_count > keep_budget and removable:
            drop_fidx = removable.pop(0)
            # Check if dropping this frame would create a coverage gap
            idx_in_kept = kept_frame_idxs.index(drop_fidx)
            prev_fidx = kept_frame_idxs[idx_in_kept - 1]
            next_fidx = kept_frame_idxs[idx_in_kept + 1]
            prev_pos = sorted_frame_idxs.index(prev_fidx)
            next_pos = sorted_frame_idxs.index(next_fidx)
            if next_pos - prev_pos > max_gap:
                continue  # skip — would leave a coverage hole
            kept_frame_idxs.remove(drop_fidx)
            kept_tiles_count -= len(frames[drop_fidx])

        print(f"[SELECT] After budget trim: {len(kept_frame_idxs)} frames, "
              f"~{kept_tiles_count} tiles")

    elif kept_tiles_count < keep_budget * 0.7 and len(kept_frame_idxs) < target_frames:
        # Too few frames — fill gaps with best unkept frames
        kept_set = set(kept_frame_idxs)
        unkept = [f for f in sorted_frame_idxs if f not in kept_set]
        max_sharpness = max(s.sharpness for s in scores) or 1.0

        # Score unkept frames by quality × gap-filling value.
        # Prefer adding frames that fill the widest gaps between kept frames.
        def _fill_score(fidx):
            """Higher = better candidate to fill."""
            tiles = frames[fidx]
            quality = sum(
                t.sift_count * 0.7 + (t.sharpness / max_sharpness) * 1000 * 0.3
                for t in tiles
            ) / len(tiles)
            # Find which gap this frame would fill
            import bisect
            pos = bisect.bisect_left(kept_frame_idxs, fidx)
            if pos == 0:
                gap = kept_frame_idxs[0] - fidx
            elif pos >= len(kept_frame_idxs):
                gap = fidx - kept_frame_idxs[-1]
            else:
                gap = kept_frame_idxs[pos] - kept_frame_idxs[pos - 1]
            gap_score = min(gap / max_gap, 1.0)  # larger gap = higher priority
            max_q = max((sum(t.sift_count for t in frames[f]) / len(frames[f])
                        for f in unkept), default=1.0)
            return (quality / (max_q or 1.0)) * 0.4 + gap_score * 0.6

        unkept_scored = sorted(unkept, key=_fill_score, reverse=True)

        for fidx in unkept_scored:
            if kept_tiles_count >= keep_budget:
                break
            kept_frame_idxs.append(fidx)
            kept_frame_idxs.sort()  # keep sorted for gap calculation
            kept_tiles_count += len(frames[fidx])

        kept_frame_idxs.sort()
        print(f"[SELECT] After gap fill: {len(kept_frame_idxs)} frames, "
              f"~{kept_tiles_count} tiles")

    # Collect all tiles from kept frames (already filtered)
    kept_set = set(kept_frame_idxs)
    kept_tiles = set()
    for fidx in kept_set:
        for t in frames[fidx]:
            kept_tiles.add(t.path)

    # Build delete list
    to_delete = []
    for s in scores:
        if s.path not in kept_tiles:
            to_delete.append(s.path)

    print(f"[SELECT] Keeping {len(kept_tiles)} tiles from "
          f"{len(kept_set)}/{n_frames_total} frames, "
          f"deleting {len(to_delete)}")

    return to_delete


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score and select tiles")
    parser.add_argument("--dir", required=True, help="Image directory")
    parser.add_argument("--target", type=int, default=300, help="Target tile count")
    parser.add_argument("--min-sift", type=int, default=80, help="Min SIFT features (at 512px scoring res)")
    parser.add_argument("--max-overlap", type=float, default=0.45,
                        help="Max visual overlap to consider frames redundant (0-1)")
    parser.add_argument("--dry-run", action="store_true", help="Don't delete, just report")
    args = parser.parse_args()

    scores = score_tiles(args.dir)
    if not scores:
        print("No tiles found matching expected filename pattern")
        sys.exit(0)

    # Per-camera stats
    cameras: dict[str, list[int]] = {}
    for s in scores:
        cameras.setdefault(s.camera, []).append(s.sift_count)
    print("\n  Per-camera SIFT stats:")
    for cam in sorted(cameras.keys()):
        vals = cameras[cam]
        print(f"    {cam:20s}: mean={sum(vals)/len(vals):6.0f}  "
              f"min={min(vals):4d}  max={max(vals):5d}  n={len(vals)}")

    to_delete = select_and_prune(
        scores, target_count=args.target,
        min_sift=args.min_sift,
        max_overlap=args.max_overlap,
    )
    remaining = len(scores) - len(to_delete)
    print(f"\nResult: keep {remaining}, delete {len(to_delete)} "
          f"(target was {args.target})")

    if not args.dry_run and to_delete:
        for path in to_delete:
            os.remove(path)
        print(f"Deleted {len(to_delete)} files")
    elif args.dry_run and to_delete:
        print("(dry run, no files deleted)")
