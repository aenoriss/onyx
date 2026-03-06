"""Score and select best tiles from over-extracted candidate set.

Uses SIFT feature count to rank tiles by quality and temporal binning
for uniform coverage. Keeps target × overshoot tiles, expecting SfM
to reject some. COLMAP's registration is the ground truth — we just
remove obvious garbage (featureless ceiling/floor/blur) cheaply.
"""

import cv2
import os
import re
import sys
import time
from dataclasses import dataclass

SCORE_RESOLUTION = 512  # downscale for fast SIFT scoring

# Pattern: {video}_frame{XXXXXX}_{camera}.jpg
TILE_PATTERN = re.compile(r".*_frame(\d+)_(.+)\.(jpg|png)$", re.IGNORECASE)


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


def select_and_prune(
    scores: list[TileScore],
    target_count: int,
    min_sift: int = 50,
    min_frames_keep: int = 10,
    sfm_overshoot: float = 1.6,
) -> list[str]:
    """Select best tiles and return list of paths TO DELETE.

    Keeps target_count × sfm_overshoot tiles ranked by SIFT quality,
    with temporal binning for uniform timeline coverage.
    COLMAP decides which actually register — overshoot compensates.
    """
    if not scores:
        return []

    featureless = [s for s in scores if s.sift_count < min_sift]
    viable = [s for s in scores if s.sift_count >= min_sift]

    keep_budget = int(target_count * sfm_overshoot)
    print(f"[SELECT] {len(viable)} viable, {len(featureless)} featureless "
          f"(< {min_sift} SIFT)")
    print(f"[SELECT] Budget: {keep_budget} tiles "
          f"(target {target_count} × {sfm_overshoot}x overshoot)")

    if not viable:
        return [s.path for s in featureless]

    # Group viable tiles by frame
    frames: dict[int, list[TileScore]] = {}
    for s in viable:
        frames.setdefault(s.frame_idx, []).append(s)

    # Normalize sharpness for composite scoring
    max_sharpness = max(s.sharpness for s in viable) or 1.0

    # Score each frame by avg composite quality
    frame_scores: dict[int, float] = {}
    for fidx, tiles in frames.items():
        composite = sum(
            t.sift_count * 0.7 + (t.sharpness / max_sharpness) * 1000 * 0.3
            for t in tiles
        ) / len(tiles)
        frame_scores[fidx] = composite

    # How many frames to keep
    avg_tiles_per_frame = len(viable) / len(frames)
    target_frames = max(min_frames_keep, int(keep_budget / max(avg_tiles_per_frame, 1)))
    sorted_frame_idxs = sorted(frames.keys())

    if target_frames >= len(sorted_frame_idxs):
        print(f"[SELECT] Keeping all {len(viable)} viable tiles "
              f"({len(sorted_frame_idxs)} frames, budget {keep_budget})")
        return [s.path for s in featureless]

    # Temporal binning: divide timeline into equal bins, pick best frame per bin
    n_frames_total = len(sorted_frame_idxs)
    bin_size = n_frames_total / target_frames
    kept_frame_idxs = set()

    for bin_i in range(target_frames):
        bin_start = int(bin_i * bin_size)
        bin_end = min(int((bin_i + 1) * bin_size), n_frames_total)
        if bin_start >= bin_end:
            bin_end = bin_start + 1
        bin_frames = sorted_frame_idxs[bin_start:bin_end]
        best = max(bin_frames, key=lambda f: frame_scores[f])
        kept_frame_idxs.add(best)

    # Collect tiles from kept frames
    kept_tiles = set()
    for fidx in kept_frame_idxs:
        for t in frames[fidx]:
            kept_tiles.add(t.path)

    # If still over budget, drop lowest-scoring individual tiles
    if len(kept_tiles) > keep_budget:
        kept_list = []
        for fidx in kept_frame_idxs:
            kept_list.extend(frames[fidx])
        kept_list.sort(key=lambda t: t.sift_count)
        while len(kept_tiles) > keep_budget and kept_list:
            drop = kept_list.pop(0)
            kept_tiles.discard(drop.path)

    # Build delete list
    to_delete = [s.path for s in featureless]
    for s in viable:
        if s.path not in kept_tiles:
            to_delete.append(s.path)

    print(f"[SELECT] Keeping {len(kept_tiles)} tiles from "
          f"{len(kept_frame_idxs)}/{n_frames_total} frames, "
          f"deleting {len(to_delete)}")

    return to_delete


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score and select tiles")
    parser.add_argument("--dir", required=True, help="Image directory")
    parser.add_argument("--target", type=int, default=300, help="Target tile count")
    parser.add_argument("--min-sift", type=int, default=50, help="Min SIFT features")
    parser.add_argument("--overshoot", type=float, default=1.6, help="SfM overshoot factor")
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
        min_sift=args.min_sift, sfm_overshoot=args.overshoot,
    )
    remaining = len(scores) - len(to_delete)
    print(f"\nResult: keep {remaining}, delete {len(to_delete)} "
          f"(target was {args.target}, budget {int(args.target * args.overshoot)})")

    if not args.dry_run and to_delete:
        for path in to_delete:
            os.remove(path)
        print(f"Deleted {len(to_delete)} files")
    elif args.dry_run and to_delete:
        print("(dry run, no files deleted)")
