"""Score and select best tiles from over-extracted candidate set.

Uses SIFT feature matching to measure visual overlap between consecutive
frames. Keeps frames where camera has moved enough (overlap drops below
threshold), ensuring spatial coverage rather than uniform time sampling.
"""

import cv2
import os
import re
import sys
import time
from dataclasses import dataclass, field

import numpy as np

SCORE_RESOLUTION = 512  # downscale for fast SIFT scoring
OVERLAP_RESOLUTION = 256  # downscale for overlap matching (speed)

# Pattern: {video}_frame{XXXXXX}_{camera}.jpg
TILE_PATTERN = re.compile(r".*_frame(\d+)_(.+)\.(jpg|png)$", re.IGNORECASE)

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

    # Downscale for speed
    for img in [img_a, img_b]:
        h, w = img.shape[:2]
        if max(h, w) > OVERLAP_RESOLUTION:
            scale = OVERLAP_RESOLUTION / max(h, w)
            img_a_r = cv2.resize(img_a, (int(img_a.shape[1] * scale), int(img_a.shape[0] * scale)))
            img_b_r = cv2.resize(img_b, (int(img_b.shape[1] * scale), int(img_b.shape[0] * scale)))
            break
    else:
        img_a_r, img_b_r = img_a, img_b

    kp_a, desc_a = sift.detectAndCompute(img_a_r, None)
    kp_b, desc_b = sift.detectAndCompute(img_b_r, None)

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


def select_and_prune(
    scores: list[TileScore],
    target_count: int,
    min_sift: int = 50,
    min_frames_keep: int = 10,
    max_overlap: float = 0.45,
    max_gap: int = 8,
) -> list[str]:
    """Select best tiles using visual overlap and return list of paths TO DELETE.

    Uses SIFT feature matching between consecutive frames to measure visual
    overlap. Keeps frames where the camera has moved enough (overlap < max_overlap).
    Guarantees coverage by force-keeping if max_gap consecutive frames skipped.

    Args:
        max_overlap: Maximum overlap ratio to consider frames redundant (0.45 = 45%)
        max_gap: Force-keep after this many skipped frames (prevents coverage gaps)
    """
    if not scores:
        return []

    n_featureless = sum(1 for s in scores if s.sift_count < min_sift)

    # Overshoot: keep 1.6x target to compensate for InstantSfM dropping
    # unregistered images during SfM (typically 10-40% fail to register)
    keep_budget = int(target_count * 1.6)
    print(f"[SELECT] {len(scores)} total, {n_featureless} featureless "
          f"(< {min_sift} SIFT)")
    print(f"[SELECT] Budget: {keep_budget} tiles (target {target_count}, 1.6x overshoot)")

    # Group ALL tiles by frame
    frames: dict[int, list[TileScore]] = {}
    for s in scores:
        frames.setdefault(s.frame_idx, []).append(s)

    sorted_frame_idxs = sorted(frames.keys())
    n_frames_total = len(sorted_frame_idxs)

    if n_frames_total == 0:
        return []

    # Pick representative tile per frame (highest SIFT count) for overlap comparison
    rep_tiles: dict[int, TileScore] = {}
    for fidx, tiles in frames.items():
        rep_tiles[fidx] = max(tiles, key=lambda t: t.sift_count)

    avg_tiles_per_frame = len(scores) / n_frames_total
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
    last_kept_rep = rep_tiles[sorted_frame_idxs[0]].path
    frames_since_kept = 0

    for i in range(1, n_frames_total):
        fidx = sorted_frame_idxs[i]
        current_rep = rep_tiles[fidx].path
        frames_since_kept += 1

        # Force-keep if too many frames skipped (coverage guarantee)
        if frames_since_kept >= max_gap:
            kept_frame_idxs.append(fidx)
            last_kept_rep = current_rep
            frames_since_kept = 0
            continue

        overlap = _compute_overlap(last_kept_rep, current_rep, sift, flann)

        if overlap < max_overlap:
            # Camera moved enough — keep this frame
            kept_frame_idxs.append(fidx)
            last_kept_rep = current_rep
            frames_since_kept = 0

    elapsed = time.time() - t0
    print(f"[SELECT] Overlap pass: {len(kept_frame_idxs)}/{n_frames_total} frames "
          f"kept in {elapsed:.1f}s")

    # ── Adapt overlap threshold if we have too many/few frames ──
    # If overlap selection gave us way more frames than budget, tighten threshold
    # If way fewer, loosen. Do one adjustment pass.
    kept_tiles_count = sum(len(frames[f]) for f in kept_frame_idxs)

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

        # Sort by quality, remove lowest quality frames until within budget
        # But never remove first or last frame (boundary coverage)
        # And never create a gap > max_gap between remaining kept frames
        removable = sorted(kept_frame_idxs[1:-1], key=lambda f: frame_quality[f])
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

        # Score unkept frames
        unkept_scored = []
        for fidx in unkept:
            tiles = frames[fidx]
            composite = sum(
                t.sift_count * 0.7 + (t.sharpness / max_sharpness) * 1000 * 0.3
                for t in tiles
            ) / len(tiles)
            unkept_scored.append((fidx, composite))
        unkept_scored.sort(key=lambda x: -x[1])  # best first

        for fidx, _ in unkept_scored:
            if kept_tiles_count >= keep_budget:
                break
            kept_frame_idxs.append(fidx)
            kept_tiles_count += len(frames[fidx])

        kept_frame_idxs.sort()
        print(f"[SELECT] After gap fill: {len(kept_frame_idxs)} frames, "
              f"~{kept_tiles_count} tiles")

    # Collect tiles from kept frames, excluding featureless tiles
    kept_set = set(kept_frame_idxs)
    kept_tiles = set()
    n_featureless_dropped = 0
    for fidx in kept_set:
        for t in frames[fidx]:
            if t.sift_count < min_sift:
                n_featureless_dropped += 1
            else:
                kept_tiles.add(t.path)
    if n_featureless_dropped:
        print(f"[SELECT] Dropped {n_featureless_dropped} featureless tiles "
              f"(< {min_sift} SIFT features)")

    # If still over budget, drop lowest-scoring individual tiles
    if len(kept_tiles) > keep_budget:
        kept_list = []
        for fidx in kept_set:
            kept_list.extend(frames[fidx])
        kept_list.sort(key=lambda t: t.sift_count)
        while len(kept_tiles) > keep_budget and kept_list:
            drop = kept_list.pop(0)
            kept_tiles.discard(drop.path)

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
    parser.add_argument("--min-sift", type=int, default=50, help="Min SIFT features")
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
