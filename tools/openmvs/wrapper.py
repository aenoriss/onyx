#!/usr/bin/env python3
"""Onyx OpenMVS wrapper — 4-step photogrammetry pipeline with progress.

Handles environment setup (ephemeral workspace in /tmp/) and orchestrates:
  1. InterfaceCOLMAP  — convert COLMAP to OpenMVS format
  2. DensifyPointCloud — dense stereo reconstruction
  3. ReconstructMesh   — mesh from dense cloud
  4. TextureMesh       — texture atlas generation

All workspace operations happen in /tmp/mvs_workspace/ to avoid
mutating the shared volume.

Usage:
    python wrapper.py --data_path /data
    python wrapper.py --data_path /data --decimate 0.2
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from pipeline_progress import progress, run_with_progress

TOTAL_STEPS = 4
WORKSPACE = "/tmp/mvs_workspace"


def setup_workspace(data_path):
    """Create ephemeral workspace with proper structure for OpenMVS."""
    ws = Path(WORKSPACE)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    # InterfaceCOLMAP expects sparse/cameras.bin at root level (not sparse/0/)
    sparse_src = Path(data_path) / "sparse" / "0"
    sparse_dst = ws / "sparse"
    sparse_dst.mkdir()
    for f in sparse_src.iterdir():
        shutil.copy2(str(f), str(sparse_dst / f.name))

    # InterfaceCOLMAP stores absolute image paths in .mvs file
    # Symlink images to a stable path that persists across steps
    images_src = Path(data_path) / "images"
    images_dst = Path("/root/images")
    if images_dst.exists() or images_dst.is_symlink():
        images_dst.unlink()
    images_dst.symlink_to(images_src)

    return ws


def main():
    parser = argparse.ArgumentParser(description="Onyx OpenMVS wrapper")
    parser.add_argument("--data_path", required=True,
                        help="Path to data directory (images/ + sparse/0/)")
    parser.add_argument("--decimate", type=float, default=None,
                        help="Mesh decimation factor (e.g. 0.2 for proto)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: data_path/output/openmvs)")
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    output_path = Path(args.output) if args.output else data_path / "output" / "openmvs"
    output_path.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if not (data_path / "sparse" / "0" / "cameras.bin").exists():
        print(f"Error: COLMAP sparse not found at {data_path}/sparse/0/")
        sys.exit(1)
    if not (data_path / "images").exists():
        print(f"Error: Images not found at {data_path}/images/")
        sys.exit(1)

    print("=" * 60)
    print("OpenMVS Photogrammetry Pipeline")
    print("=" * 60)
    print(f"Data:     {data_path}")
    print(f"Output:   {output_path}")
    print(f"Decimate: {args.decimate or 'none (full quality)'}")
    print("=" * 60)

    # Setup ephemeral workspace
    ws = setup_workspace(data_path)

    # ── Step 1: InterfaceCOLMAP ───────────────────────────────
    # Fast (<30s) — no patterns, indeterminate is fine
    run_with_progress(
        ["InterfaceCOLMAP",
         "--input-file", str(ws),
         "--image-folder", "/root/images",
         "--output-file", str(ws / "scene.mvs")],
        stage="interface_colmap",
        step=1, total_steps=TOTAL_STEPS,
    )

    # ── Step 2: DensifyPointCloud ─────────────────────────────
    # Longest step (30min–2hr). OpenMVS outputs lines like:
    #   "Estimating depth-map for image N/M..." or "depth-map N/M estimated"
    densify_patterns = {
        r"(?i)depth.{1,50}?(\d+)/(\d+)": lambda m: (
            round(int(m.group(1)) / int(m.group(2)) * 100),
            f"Depth {m.group(1)}/{m.group(2)}",
        ),
        r"\b(\d+)%(?!\|)": lambda m: (
            int(m.group(1)),
            f"{m.group(1)}%",
        ),
    }
    run_with_progress(
        ["DensifyPointCloud",
         "--input-file", str(ws / "scene.mvs"),
         "--output-file", str(ws / "scene_dense.mvs")],
        stage="densify_pointcloud",
        step=2, total_steps=TOTAL_STEPS,
        patterns=densify_patterns,
    )

    # ── Step 3: ReconstructMesh ───────────────────────────────
    mesh_cmd = [
        "ReconstructMesh",
        "--input-file", str(ws / "scene_dense.mvs"),
        "--output-file", str(ws / "scene_mesh.mvs"),
    ]
    if args.decimate:
        mesh_cmd.extend(["--decimate", str(args.decimate)])

    reconstruct_patterns = {
        r"\b(\d+)%(?!\|)": lambda m: (
            int(m.group(1)),
            f"{m.group(1)}%",
        ),
        r"(?i)(?:view|face|vertex).{1,30}?(\d+)/(\d+)": lambda m: (
            round(int(m.group(1)) / int(m.group(2)) * 100),
            f"Mesh {m.group(1)}/{m.group(2)}",
        ),
    }
    run_with_progress(
        mesh_cmd,
        stage="reconstruct_mesh",
        step=3, total_steps=TOTAL_STEPS,
        patterns=reconstruct_patterns,
    )

    # ── Step 4: TextureMesh ───────────────────────────────────
    # Note: --export-type obj segfaults on v2.3.0, use default MVS+PLY
    texture_patterns = {
        r"(?i)(?:view|patch|atlas).{1,30}?(\d+)/(\d+)": lambda m: (
            round(int(m.group(1)) / int(m.group(2)) * 100),
            f"Texture {m.group(1)}/{m.group(2)}",
        ),
        r"\b(\d+)%(?!\|)": lambda m: (
            int(m.group(1)),
            f"{m.group(1)}%",
        ),
    }
    run_with_progress(
        ["TextureMesh",
         "--input-file", str(ws / "scene_dense.mvs"),
         "--mesh-file", str(ws / "scene_mesh.ply"),
         "--output-file", str(ws / "scene_textured.mvs")],
        stage="texture_mesh",
        step=4, total_steps=TOTAL_STEPS,
        patterns=texture_patterns,
    )

    # ── Copy results to shared volume ─────────────────────────
    print(f"\nCopying results to {output_path}...")
    for ext in ("*.mvs", "*.ply", "*.png", "*.jpg"):
        for f in ws.glob(ext):
            if "scene" in f.name:
                shutil.copy2(str(f), str(output_path / f.name))

    # Also copy texture atlas files if present
    for f in ws.glob("scene_textured*"):
        dst = output_path / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(str(f), str(dst))

    progress("done", "completed", 100, step=TOTAL_STEPS, total_steps=TOTAL_STEPS)

    # Summary
    print(f"\n{'='*60}")
    print(f"[COMPLETE] OpenMVS pipeline finished")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    for f in sorted(output_path.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
