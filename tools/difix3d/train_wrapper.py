"""
Difix3D+ wrapper for the Onyx pipeline.

Runs gsplat training with Difix3D+ interleaved fix steps.
Optionally initializes from an existing MILo/3DGS .ply checkpoint.

Usage (inside container):
    # From scratch (COLMAP SfM init):
    python train_wrapper.py --scene /data

    # From MILo output (warm-start):
    python train_wrapper.py --scene /data --init-ply /data/output/milo/point_cloud/iteration_18000/point_cloud.ply
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


TRAINER = "/workspace/Difix3D/examples/gsplat/simple_trainer_difix3d.py"


def convert_ply_to_ckpt(ply_path: Path, ckpt_path: Path) -> None:
    """Convert a standard 3DGS .ply (MILo/inria format) to a gsplat .pt checkpoint."""
    import numpy as np
    import torch
    from plyfile import PlyData

    print(f"[difix_init] Loading MILo splat from {ply_path} ...")
    plydata = PlyData.read(str(ply_path))
    verts = plydata["vertex"]
    N = len(verts["x"])
    print(f"[difix_init] {N:,} Gaussians loaded")

    # Positions [N, 3]
    means = torch.tensor(
        np.stack([verts["x"], verts["y"], verts["z"]], axis=1), dtype=torch.float32
    )

    # Scales [N, 3] — already log-space in .ply
    scales = torch.tensor(
        np.stack([verts["scale_0"], verts["scale_1"], verts["scale_2"]], axis=1),
        dtype=torch.float32,
    )

    # Quaternions [N, 4] — [w, x, y, z], unnormalized is fine
    quats = torch.tensor(
        np.stack([verts["rot_0"], verts["rot_1"], verts["rot_2"], verts["rot_3"]], axis=1),
        dtype=torch.float32,
    )

    # Opacities [N,] — already logit-space in .ply
    opacities = torch.tensor(np.array(verts["opacity"]), dtype=torch.float32)

    # DC SH [N, 1, 3]
    sh0 = torch.tensor(
        np.stack([verts["f_dc_0"], verts["f_dc_1"], verts["f_dc_2"]], axis=1),
        dtype=torch.float32,
    ).unsqueeze(1)  # [N, 1, 3]

    # Higher-order SH [N, 15, 3]
    # inria .ply stores f_rest as [r0..r14, g0..g14, b0..b14] (45 values, degree=3)
    # gsplat wants [N, 15, 3] where last dim is RGB for each coefficient
    rest_keys = sorted([k for k in verts.data.dtype.names if k.startswith("f_rest_")],
                       key=lambda k: int(k.split("_")[-1]))
    if rest_keys:
        f_rest = np.stack([np.array(verts[k]) for k in rest_keys], axis=1)  # [N, 45]
        num_coeffs = f_rest.shape[1] // 3  # 15 for degree 3
        # Reshape [N, 3, 15] then transpose → [N, 15, 3]
        shN = torch.tensor(
            f_rest.reshape(N, 3, num_coeffs).transpose(0, 2, 1),
            dtype=torch.float32,
        )
    else:
        shN = torch.zeros(N, 0, 3, dtype=torch.float32)

    ckpt = {
        "step": 0,
        "splats": {
            "means":     means,
            "scales":    scales,
            "quats":     quats,
            "opacities": opacities,
            "sh0":       sh0,
            "shN":       shN,
        },
    }

    torch.save(ckpt, str(ckpt_path))
    print(f"[difix_init] Saved gsplat checkpoint → {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Difix3D+ wrapper for Onyx pipeline")
    parser.add_argument("--scene", required=True,
                        help="Path to scene directory (must contain sparse/0/ and images/)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: scene/output/difix3d)")
    parser.add_argument("--init-ply", default=None,
                        help="Path to MILo/3DGS .ply to warm-start from "
                             "(e.g. output/milo/point_cloud/iteration_18000/point_cloud.ply)")
    parser.add_argument("--steps", type=int, default=30000,
                        help="Total training steps (default: 30000)")
    parser.add_argument("--data-factor", type=int, default=1,
                        help="Image downsample factor (default: 1 = full resolution)")
    args = parser.parse_args()

    scene = Path(args.scene).resolve()
    output = Path(args.output) if args.output else scene / "output" / "difix3d"
    output.mkdir(parents=True, exist_ok=True)

    # Validate COLMAP data
    for p in [scene / "sparse" / "0", scene / "images"]:
        if not p.exists():
            print(f"[ERROR] Required path not found: {p}")
            sys.exit(1)

    # Build fix_steps: every 3k steps
    fix_steps = list(range(3000, args.steps + 1, 3000))

    print("=" * 60)
    print("Difix3D+ — Onyx Pipeline")
    print("=" * 60)
    print(f"Scene:       {scene}")
    print(f"Output:      {output}")
    print(f"Steps:       {args.steps}")
    print(f"Data factor: {args.data_factor}x")
    print(f"Fix steps:   {fix_steps}")
    print(f"Init PLY:    {args.init_ply or 'COLMAP SfM (default)'}")
    print("=" * 60)

    cmd = [
        "python3", "-u", TRAINER, "default",
        "--data_dir", str(scene),
        "--result_dir", str(output),
        "--max_steps", str(args.steps),
        "--data_factor", str(args.data_factor),
        "--fix_steps", str(fix_steps),
        "--novel_data_lambda", "0.3",
        "--render_traj_path", "interp",
    ]

    # Warm-start from MILo .ply
    if args.init_ply:
        ply_path = Path(args.init_ply).resolve()
        if not ply_path.exists():
            print(f"[ERROR] --init-ply not found: {ply_path}")
            sys.exit(1)
        ckpt_path = output / "milo_init.pt"
        convert_ply_to_ckpt(ply_path, ckpt_path)
        cmd.extend(["--ckpt", str(ckpt_path)])

    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
