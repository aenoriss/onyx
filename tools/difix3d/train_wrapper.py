"""
Difix3D+ post-processing refinement for the Onyx pipeline.

Takes an existing trained splat PLY + COLMAP scene data, and runs DiFix3D+
interleaved fix cycles to improve quality (especially novel-view synthesis).

Usage (inside container):
    # Post-processing refinement from gsplat/splatfacto PLY:
    python train_wrapper.py --scene /data --init-ply /data/output/splatfacto/ply/splat.ply

    # With custom steps and fix interval:
    python train_wrapper.py --scene /data --init-ply /data/splat.ply --steps 10000 --fix-interval 2000
"""

import argparse
import subprocess
import sys
from pathlib import Path


TRAINER = "/workspace/Difix3D/examples/gsplat/simple_trainer_difix3d.py"


def convert_ply_to_ckpt(ply_path: Path, ckpt_path: Path) -> None:
    """Convert a standard 3DGS/splatfacto .ply to a gsplat .pt checkpoint."""
    import numpy as np
    import torch
    from plyfile import PlyData

    print(f"[difix_init] Loading splat from {ply_path} ...")
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

    # Higher-order SH [N, K, 3]
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
    print(f"[difix_init] Saved gsplat checkpoint → {ckpt_path} ({N:,} Gaussians)")


def main():
    parser = argparse.ArgumentParser(description="Difix3D+ refinement for Onyx pipeline")
    parser.add_argument("--scene", required=True,
                        help="Path to scene directory (must contain sparse/0/ and images/)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: scene/output/difix3d)")
    parser.add_argument("--init-ply", default=None,
                        help="Path to trained splat .ply for warm-start refinement")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Total refinement steps (default: 10000)")
    parser.add_argument("--fix-interval", type=int, default=2000,
                        help="Run DiFix3D+ fix cycle every N steps (default: 2000)")
    parser.add_argument("--data-factor", type=int, default=1,
                        help="Image downsample factor (default: 1 = full resolution)")
    parser.add_argument("--novel-lambda", type=float, default=0.3,
                        help="Probability of novel data reuse per iteration (default: 0.3)")
    args = parser.parse_args()

    scene = Path(args.scene).resolve()
    output = Path(args.output) if args.output else scene / "output" / "difix3d"
    output.mkdir(parents=True, exist_ok=True)

    # Validate COLMAP data
    for p in [scene / "sparse" / "0", scene / "images"]:
        if not p.exists():
            print(f"[ERROR] Required path not found: {p}")
            sys.exit(1)

    # Build fix_steps schedule
    fix_steps = list(range(args.fix_interval, args.steps + 1, args.fix_interval))
    if not fix_steps:
        fix_steps = [args.steps // 2]

    print("=" * 60)
    print("Difix3D+ Post-Processing Refinement — Onyx Pipeline")
    print("=" * 60)
    print(f"Scene:        {scene}")
    print(f"Output:       {output}")
    print(f"Steps:        {args.steps}")
    print(f"Fix interval: every {args.fix_interval} steps")
    print(f"Fix schedule: {fix_steps}")
    print(f"Novel lambda: {args.novel_lambda}")
    print(f"Data factor:  {args.data_factor}x")
    print(f"Init PLY:     {args.init_ply or 'COLMAP SfM (default)'}")
    print("=" * 60)

    cmd = [
        "python3", "-u", TRAINER, "default",
        "--data_dir", str(scene),
        "--result_dir", str(output),
        "--max_steps", str(args.steps),
        "--data_factor", str(args.data_factor),
        "--fix_steps", str(fix_steps),
        "--novel_data_lambda", str(args.novel_lambda),
        "--render_traj_path", "interp",
    ]

    # Warm-start from existing PLY
    if args.init_ply:
        ply_path = Path(args.init_ply).resolve()
        if not ply_path.exists():
            print(f"[ERROR] --init-ply not found: {ply_path}")
            sys.exit(1)
        ckpt_path = output / "init_splat.pt"
        convert_ply_to_ckpt(ply_path, ckpt_path)
        cmd.extend(["--ckpt", str(ckpt_path)])

    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    # Export final PLY from the gsplat checkpoint
    if result.returncode == 0:
        print("\n[difix] Training complete. Exporting refined PLY...")
        _export_ply(output, scene)

    sys.exit(result.returncode)


def _export_ply(output_dir: Path, scene_dir: Path):
    """Export the final gsplat checkpoint to a standard 3DGS PLY."""
    import glob
    import torch
    import numpy as np
    from plyfile import PlyData, PlyElement

    # Find latest checkpoint
    ckpt_files = sorted(glob.glob(str(output_dir / "ckpts" / "*.pt")))
    if not ckpt_files:
        print("[difix] No checkpoint found for export")
        return

    ckpt_path = ckpt_files[-1]
    print(f"[difix] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    splats = ckpt["splats"]

    means = splats["means"].numpy()          # [N, 3]
    scales = splats["scales"].numpy()         # [N, 3]
    quats = splats["quats"].numpy()           # [N, 4]
    opacities = splats["opacities"].numpy()   # [N]
    sh0 = splats["sh0"].numpy()               # [N, 1, 3]
    shN = splats["shN"].numpy()               # [N, K, 3]

    N = means.shape[0]
    print(f"[difix] Exporting {N:,} Gaussians")

    # Build structured array
    props = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    # Higher-order SH
    n_rest = shN.shape[1] * 3
    for i in range(n_rest):
        props.append((f"f_rest_{i}", "f4"))
    props.extend([
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ])

    arr = np.zeros(N, dtype=props)
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    arr["f_dc_0"] = sh0[:, 0, 0]
    arr["f_dc_1"] = sh0[:, 0, 1]
    arr["f_dc_2"] = sh0[:, 0, 2]

    # SH rest: gsplat stores [N, K, 3], PLY wants [r0..rK, g0..gK, b0..bK]
    if shN.shape[1] > 0:
        K = shN.shape[1]
        # [N, K, 3] → [N, 3, K] → flatten to [N, 3*K]
        rest_flat = shN.transpose(0, 2, 1).reshape(N, -1)
        for i in range(n_rest):
            arr[f"f_rest_{i}"] = rest_flat[:, i]

    arr["opacity"] = opacities
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    el = PlyElement.describe(arr, "vertex")
    out_path = output_dir / "splat_refined.ply"
    PlyData([el], text=False).write(str(out_path))
    print(f"[difix] Refined PLY exported → {out_path} ({N:,} Gaussians)")


if __name__ == "__main__":
    main()
