#!/usr/bin/env python3
"""
YonoSplat inference wrapper for Onyx Pipeline.

Bridges raw JPEG images to YonoSplat's feed-forward Gaussian Splatting pipeline
without COLMAP or per-scene optimization (pose-free).

Three stages:
  1. Preprocess  — load JPEGs, resize to model input resolution, build batch tensor
  2. Inference   — run YonoSplat via its Hydra/Lightning CLI (subprocess)
  3. Export      — write standard 3DGS PLY from YonoSplat's output

YonoSplat repo:  https://github.com/cvg/YonoSplat (arXiv 2511.07321)
Checkpoints:     dl3dv.ckpt (outdoor) · re10k.ckpt (indoor)

Usage (called by run_pipeline.py via Docker):
    python train_wrapper.py \\
        --scene     /data/images \\
        --output    /data/output/yonosplat \\
        --scene-type outdoor \\
        --quality   yono
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

from pipeline_progress import progress

# ── Constants ─────────────────────────────────────────────────

YONOSPLAT_ROOT = "/workspace/YonoSplat"
WEIGHTS_DIR    = "/workspace/pretrained_weights"

# YoNoSplat model input resolution (224x224 per pretrained checkpoint config)
IMAGE_SIZE = 224

# Checkpoint selection by scene type
CHECKPOINT_MAP = {
    "outdoor": "dl3dv.ckpt",
    "indoor":  "re10k.ckpt",
}

# Experiment config names (match YoNoSplat's Hydra experiment files — from README)
EXPERIMENT_MAP = {
    "outdoor": "yono_dl3dv",
    "indoor":  "yono_re10k",
}

# How many images to feed the model per quality level.
# Yono quality uses all images extracted by ingest (150 outdoor / 100 indoor).
# Production/proto are exposed for future use but not currently triggered
# from the orchestrator with quality=yono.
TARGET_IMAGES = {
    ("outdoor", "yono"):       150,
    ("outdoor", "production"): 150,
    ("outdoor", "proto"):       60,
    ("indoor",  "yono"):       100,
    ("indoor",  "production"): 100,
    ("indoor",  "proto"):       40,
}


# ── Image Loading ──────────────────────────────────────────────

def load_images(images_dir, max_images):
    """Load and preprocess images from directory.

    Returns a float tensor of shape [N, 3, H, W] with values in [0, 1].
    Uniformly subsamples to max_images if more are present.
    """
    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
    all_files = sorted(
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )

    if not all_files:
        print(f"[ERROR] No images found in {images_dir}")
        sys.exit(1)

    # Uniform subsampling
    if len(all_files) > max_images:
        step = len(all_files) / max_images
        all_files = [all_files[int(i * step)] for i in range(max_images)]

    print(f"Loading {len(all_files)} images from {images_dir} "
          f"(target {max_images}, resize to {IMAGE_SIZE}×{IMAGE_SIZE})")

    tensors = []
    for fname in all_files:
        path = os.path.join(images_dir, fname)
        img = Image.open(path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0   # [H, W, 3]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        tensors.append(tensor)

    return torch.stack(tensors, dim=0)  # [N, 3, H, W]


# ── PLY Export ────────────────────────────────────────────────

def export_ply(gaussians, output_path):
    """Export Gaussian splat as binary PLY in standard 3DGS attribute schema.

    Attribute layout (matches Inria 3DGS / SuperSplat convention):
        position:  x, y, z
        normals:   nx, ny, nz  (zeros — not used by splat viewers)
        SH DC:     f_dc_0, f_dc_1, f_dc_2
        SH rest:   f_rest_0 … f_rest_44  (zeros — degree-0 only from feed-forward)
        opacity:   opacity (pre-sigmoid logit)
        scale:     scale_0, scale_1, scale_2  (log-scale)
        rotation:  rot_0, rot_1, rot_2, rot_3 (quaternion w, x, y, z)

    `gaussians` may be a dict or an object with attribute-access.  We try both
    common naming conventions (PixelSplat-style and plain 3DGS-style) so the
    wrapper stays compatible with variations in YonoSplat's output format.
    """
    def _get(obj, *keys):
        for k in keys:
            if isinstance(obj, dict):
                if k in obj:
                    return obj[k]
            elif hasattr(obj, k):
                return getattr(obj, k)
        return None

    means    = _get(gaussians, "means", "xyz", "positions")
    opacities = _get(gaussians, "opacities", "opacity")
    scales   = _get(gaussians, "scales", "log_scales", "scale")
    quats    = _get(gaussians, "quats", "rotations", "rotation")
    sh_dc    = _get(gaussians, "sh_dc", "features_dc", "colors")

    missing = [n for n, v in [("means", means), ("opacities", opacities),
                               ("scales", scales), ("quats", quats),
                               ("sh_dc", sh_dc)] if v is None]
    if missing:
        print(f"[ERROR] Missing Gaussian attributes: {missing}")
        print(f"Available keys: {list(gaussians.keys()) if isinstance(gaussians, dict) else dir(gaussians)}")
        sys.exit(1)

    # Move to CPU and convert
    means     = means.cpu().float().numpy()
    opacities = opacities.cpu().float().numpy()
    scales    = scales.cpu().float().numpy()
    quats     = quats.cpu().float().numpy()
    sh_dc     = sh_dc.cpu().float().numpy()

    # Normalise shapes
    if opacities.ndim == 2:
        opacities = opacities.squeeze(1)
    if sh_dc.ndim == 3:
        sh_dc = sh_dc.squeeze(1)   # [N, 1, 3] → [N, 3]
    if sh_dc.shape[-1] == 1:
        sh_dc = np.concatenate([sh_dc, np.zeros((sh_dc.shape[0], 2), dtype=np.float32)], axis=-1)

    N = means.shape[0]
    n_sh_rest = 45  # (deg4+1)^2 * 3 - 3 = 45 (standard 3DGS)

    attributes = (
        [("x", means[:, 0]), ("y", means[:, 1]), ("z", means[:, 2])]
        + [("nx", np.zeros(N, np.float32)),
           ("ny", np.zeros(N, np.float32)),
           ("nz", np.zeros(N, np.float32))]
        + [("f_dc_0", sh_dc[:, 0]),
           ("f_dc_1", sh_dc[:, 1]),
           ("f_dc_2", sh_dc[:, 2])]
        + [(f"f_rest_{i}", np.zeros(N, np.float32)) for i in range(n_sh_rest)]
        + [("opacity", opacities)]
        + [("scale_0", scales[:, 0]),
           ("scale_1", scales[:, 1]),
           ("scale_2", scales[:, 2])]
        + [("rot_0", quats[:, 0]),
           ("rot_1", quats[:, 1]),
           ("rot_2", quats[:, 2]),
           ("rot_3", quats[:, 3])]
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "wb") as f:
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {N}",
        ]
        for name, _ in attributes:
            header_lines.append(f"property float {name}")
        header_lines.append("end_header")
        f.write(("\n".join(header_lines) + "\n").encode("ascii"))

        # Pack all attributes row-major
        data = np.column_stack([arr.astype(np.float32) for _, arr in attributes])
        f.write(data.tobytes())

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"PLY exported: {output_path} ({size_mb:.1f} MB, {N:,} Gaussians)")


# ── Inference via direct Python API ───────────────────────────
#
# YonoSplat's Hydra config uses typed structs for the dataset — arbitrary CLI
# overrides (dataset.name, dataset.roots) are rejected.  The fix is to bypass
# the Hydra CLI entirely: load the model config with hydra.compose(), instantiate
# the encoder directly, and call its forward pass with a minimal context batch.
# This skips the dataset loader completely.

def run_yonosplat_inference(images, experiment, ckpt_path, output_dir):
    """Run YonoSplat inference via direct Python API (no Hydra CLI subprocess).

    Steps:
      1. Load model config via hydra.compose() (no CLI, no dataset).
      2. Instantiate the encoder from the config.
      3. Load encoder weights from the Lightning checkpoint.
      4. Build a minimal context dict: images + dummy normalized intrinsics.
         (The backbone predicts its own intrinsics when pose_free=True; the
         dummy values are a reasonable initialisation but are not used directly.)
      5. Call encoder.forward() to get Gaussians.
      6. Export PLY with YonoSplat's own ply_export module.
    """
    from pathlib import Path

    os.chdir(YONOSPLAT_ROOT)
    if YONOSPLAT_ROOT not in sys.path:
        sys.path.insert(0, YONOSPLAT_ROOT)

    # ── Load model config (no dataset, no CLI) ─────────────────
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(
            config_dir=os.path.join(YONOSPLAT_ROOT, "config"),
            version_base=None):
        cfg = compose(
            config_name="main",
            overrides=[
                f"+experiment={experiment}",
                "mode=test",
                "model.encoder.pose_free=true",
            ],
        )

    # ── Instantiate encoder via YonoSplat's factory ───────────
    # load_typed_root_config converts OmegaConf → typed dataclasses.
    # get_encoder() uses cfg.name="yonosplat" to look up EncoderYoNoSplat.
    # This mirrors exactly what src/main.py does.
    from src.config import load_typed_root_config
    from src.model.encoder import get_encoder

    print("Instantiating encoder via get_encoder()...")
    typed_cfg = load_typed_root_config(cfg)
    encoder, _ = get_encoder(typed_cfg.model.encoder)

    # ── Load checkpoint weights ────────────────────────────────
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    # Lightning saves the full model; filter to encoder sub-keys.
    encoder_state = {
        k[len("encoder."):]: v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"[INFO] {len(missing)} missing encoder keys (decoder/other params).")

    encoder = encoder.cuda().eval()

    # ── Build minimal context batch ────────────────────────────
    # Intrinsics: 3×3 normalized pinhole matrix (values relative to image size).
    # The 360Extractor always uses 90° FoV virtual cameras (default fov=90 in its config).
    # For 90° FoV: fx = fy = W / (2·tan(45°)) = W/2 → normalized = 0.5
    #              cx = cy = W/2                     → normalized = 0.5
    # These are the exact real intrinsics (not a guess), and they are
    # resolution-invariant so they remain correct after the 2048→224 resize.
    V = images.shape[0]
    K = torch.tensor(
        [[0.5, 0.0, 0.5],
         [0.0, 0.5, 0.5],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    intrinsics = K.view(1, 1, 3, 3).expand(1, V, 3, 3).clone().cuda()

    context = {
        "image":      images.unsqueeze(0).cuda(),        # [1, V, 3, H, W]
        "intrinsics": intrinsics,                         # [1, V, 3, 3]
        "index":      torch.arange(V).unsqueeze(0).cuda(),
    }

    # Apply encoder's data_shim — same as model_wrapper.test_step() does.
    # get_data_shim() calls encoder.get_data_shim() if available, so this
    # is encoder-specific preprocessing, not dataset-specific.
    from src.dataset.data_module import get_data_shim
    data_shim = get_data_shim(encoder)
    batch = data_shim({"context": context})
    context = batch["context"]

    # ── Encode ─────────────────────────────────────────────────
    print(f"Encoding {V} views (pose-free feed-forward)...")
    progress("inference", "running", 50, step=2, total_steps=3)

    with torch.no_grad():
        gaussians = encoder(context, global_step=0)

    N = gaussians.means.shape[1]
    print(f"Encoder complete — {N:,} Gaussians predicted")

    # ── Export PLY ─────────────────────────────────────────────
    from src.model.ply_export import export_ply

    ply_path = Path(output_dir) / "splat.ply"
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    export_ply(
        gaussians.means[0],       # [N, 3]
        gaussians.scales[0],      # [N, 3]
        gaussians.rotations[0],   # [N, 4]
        gaussians.harmonics[0],   # [N, 3, d_sh]
        gaussians.opacities[0],   # [N]
        ply_path,
        save_sh_dc_only=True,
    )

    return str(ply_path)


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YonoSplat pose-free Gaussian Splatting wrapper"
    )
    parser.add_argument("--scene", required=True,
                        help="Input image directory (/data/images)")
    parser.add_argument("--output", required=True,
                        help="Output directory (/data/output/yonosplat)")
    parser.add_argument("--scene-type", dest="scene_type",
                        choices=["indoor", "outdoor"], default="outdoor",
                        help="Scene type — selects checkpoint (dl3dv/re10k)")
    parser.add_argument("--quality",
                        choices=["production", "proto", "yono"], default="yono",
                        help="Quality level — controls image count")
    args = parser.parse_args()

    scene_type  = args.scene_type
    quality     = args.quality
    images_dir  = args.scene
    output_dir  = args.output
    total_steps = 3

    ckpt_name  = CHECKPOINT_MAP[scene_type]
    ckpt_path  = os.path.join(WEIGHTS_DIR, ckpt_name)
    experiment = EXPERIMENT_MAP[scene_type]
    max_images = TARGET_IMAGES.get((scene_type, quality), 100)
    ply_out    = os.path.join(output_dir, "splat.ply")

    print("=" * 60)
    print("YonoSplat — pose-free feed-forward Gaussian Splatting")
    print("=" * 60)
    print(f"Scene dir:   {images_dir}")
    print(f"Output:      {output_dir}")
    print(f"Scene type:  {scene_type}")
    print(f"Checkpoint:  {ckpt_path}")
    print(f"Experiment:  {experiment}")
    print(f"Max images:  {max_images}")
    print("=" * 60)

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print(f"        Expected weights in {WEIGHTS_DIR}/")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # ── Stage 1: Preprocess ────────────────────────────────────
    progress("preprocess", "running", 0, step=1, total_steps=total_steps)
    print("\n[Stage 1/3] Preprocessing images...")

    images = load_images(images_dir, max_images)
    print(f"Image tensor: {tuple(images.shape)}  (N×C×H×W)")

    progress("preprocess", "completed", 100, step=1, total_steps=total_steps)

    # ── Stage 2: Inference ────────────────────────────────────
    progress("inference", "running", 0, step=2, total_steps=total_steps)
    print("\n[Stage 2/3] Running YonoSplat inference...")

    gs_output = run_yonosplat_inference(images, experiment, ckpt_path, output_dir)

    if gs_output is None:
        print("[ERROR] YonoSplat did not produce a recognisable output file.")
        print(f"        Inspect {output_dir} and adjust 'candidates' list in wrapper.")
        sys.exit(1)

    print(f"YonoSplat output: {gs_output}")

    # ── Stage 3: Verify ───────────────────────────────────────
    progress("export", "running", 0, step=3, total_steps=total_steps)
    print("\n[Stage 3/3] Verifying PLY output...")
    # PLY is written directly by run_yonosplat_inference via YonoSplat's
    # own ply_export module; no conversion step needed.

    if not os.path.exists(ply_out) or os.path.getsize(ply_out) == 0:
        print(f"[ERROR] Output PLY missing or empty: {ply_out}")
        sys.exit(1)

    progress("export", "completed", 100, step=3, total_steps=total_steps)
    progress("done", "completed", 100, step=total_steps, total_steps=total_steps)

    size_mb = os.path.getsize(ply_out) / (1024 * 1024)
    print(f"\n[DONE] YonoSplat complete.")
    print(f"       Output: {ply_out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
