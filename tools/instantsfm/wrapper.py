#!/usr/bin/env python3
"""Onyx InstantSfM wrapper — feature extraction + SfM with progress.

Orchestrates two sequential tools (ins-feat -> ins-sfm) with per-stage
progress tracking.

Usage:
    python wrapper.py --data_path /data
"""

import argparse
import sys

from pipeline_progress import progress, run_with_progress


def main():
    parser = argparse.ArgumentParser(description="Onyx InstantSfM wrapper")
    parser.add_argument("--data_path", required=True,
                        help="Path to data directory (images/ must exist)")
    args = parser.parse_args()

    total_steps = 2

    # ── Stage 1: Feature extraction ───────────────────────────
    feat_patterns = {
        r"(\d+)\s*/\s*(\d+)\s*images": lambda m: (
            round(int(m.group(1)) / int(m.group(2)) * 100),
            f"Image {m.group(1)}/{m.group(2)}",
        ),
    }

    run_with_progress(
        ["ins-feat", "--data_path", args.data_path, "--single_camera"],
        stage="feature_extraction",
        step=1, total_steps=total_steps,
        patterns=feat_patterns,
    )

    # ── Stage 2: Structure from Motion ────────────────────────
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
