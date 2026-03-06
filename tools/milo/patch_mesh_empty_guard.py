#!/usr/bin/env python3
"""
Patch MILo to handle empty meshes gracefully in nvdiffrast rasterization.

Run once at Docker build time (AFTER other patches):
    python3 /workspace/patch_mesh_empty_guard.py

When marching tetrahedra + filtering produces zero triangles, or when
evaluate_mesh_occupancy renders a mesh that has no visible faces for a camera,
nvdiffrast crashes with: RuntimeError: tri must have shape [>0, 3]

This patch guards at TWO levels:
  1. nvdiff_rasterization() in scene/mesh.py — returns zero tensors when faces
     tensor is empty. Protects ALL callers (MeshRenderer, ScalableMeshRenderer,
     evaluate_mesh_occupancy, etc.)
  2. compute_mesh_regularization() in regularization/regularizer/mesh.py —
     skips the iteration when filtered faces are empty (avoids downstream
     rendering and loss computation entirely).
"""

import sys

# ── Patch 1: Guard nvdiff_rasterization at the source ─────────────────────────
# This is the lowest-level fix — prevents the crash regardless of caller.

MESH_PY = "/workspace/MILo/milo/scene/mesh.py"

with open(MESH_PY, "r") as f:
    code = f.read()

_P1_OLD = """\
    # Rasterize with NVDiffRast
    # TODO: WARNING: pix_to_face is not in the correct range [-1, F-1] but in [0, F],
    # With 0 indicating that no triangle was hit.
    # So we need to subtract 1.
    rast_out, _ = dr.rasterize(glctx, pos=pos, tri=faces, resolution=[image_height, image_width])"""

_P1_NEW = """\
    # Guard: empty mesh (0 triangles) crashes nvdiffrast
    if faces.shape[0] == 0:
        rast_out = torch.zeros(1, image_height, image_width, 4, device=device)
        bary_coords = rast_out[..., :2]
        zbuf = rast_out[..., 2]
        pix_to_face = (rast_out[..., 3].int() - 1)  # all -1
        if return_indices_only:
            return pix_to_face
        _output = (bary_coords, zbuf, pix_to_face)
        if return_rast_out:
            _output = _output + (rast_out,)
        if return_positions:
            _output = _output + (pos,)
        return _output

    # Rasterize with NVDiffRast
    # TODO: WARNING: pix_to_face is not in the correct range [-1, F-1] but in [0, F],
    # With 0 indicating that no triangle was hit.
    # So we need to subtract 1.
    rast_out, _ = dr.rasterize(glctx, pos=pos, tri=faces, resolution=[image_height, image_width])"""

assert _P1_OLD in code, "Patch 1 anchor not found in scene/mesh.py"
code = code.replace(_P1_OLD, _P1_NEW, 1)

with open(MESH_PY, "w") as f:
    f.write(code)

print("[patch-mesh-guard] 1/2  scene/mesh.py — nvdiff_rasterization empty faces guard added")

# ── Patch 2: Guard compute_mesh_regularization filtered faces ─────────────────
# Higher-level guard: skip entire mesh reg iteration when filtered faces are empty.
# Avoids unnecessary rendering + loss computation.

MESH_REG_PY = "/workspace/MILo/milo/regularization/regularizer/mesh.py"

with open(MESH_REG_PY, "r") as f:
    reg_code = f.read()

_P2_OLD = """\
        # --- Build and Render Mesh ---
        mesh = Meshes(verts=verts, faces=faces[faces_mask])

        mesh_render_pkg = mesh_renderer("""

_P2_NEW = """\
        # --- Build and Render Mesh ---
        _filtered_faces = faces[faces_mask]
        if _filtered_faces.shape[0] == 0:
            # Empty mesh after filtering — skip render, return zero losses
            print(f"[WARNING] Empty mesh at iter {iteration} (0 triangles after filtering). Skipping mesh regularization.")
            _device = gaussians._xyz.device
            mesh_state["delaunay_xyz_idx"] = delaunay_xyz_idx
            mesh_state["voronoi_occupancy_labels"] = voronoi_occupancy_labels
            mesh_state["delaunay_tets"] = delaunay_tets
            mesh_state["reset_delaunay_samples"] = reset_delaunay_samples
            mesh_state["reset_sdf_values"] = reset_sdf_values
            _H, _W = viewpoint_cam.image_height, viewpoint_cam.image_width
            return {
                "mesh_loss": torch.zeros(size=(), device=_device),
                "mesh_depth_loss": torch.zeros(size=(), device=_device),
                "mesh_normal_loss": torch.zeros(size=(), device=_device),
                "occupied_centers_loss": torch.zeros(size=(), device=_device),
                "occupancy_labels_loss": torch.zeros(size=(), device=_device),
                "updated_state": mesh_state,
                "mesh_render_pkg": {
                    "depth": torch.zeros(_H, _W, device=_device),
                    "normals": torch.zeros(_H, _W, 3, device=_device),
                },
                "voronoi_points_count": voronoi_points_count,
            }
        mesh = Meshes(verts=verts, faces=_filtered_faces)

        mesh_render_pkg = mesh_renderer("""

assert _P2_OLD in reg_code, "Patch 2 anchor not found in regularization/regularizer/mesh.py"
reg_code = reg_code.replace(_P2_OLD, _P2_NEW, 1)

with open(MESH_REG_PY, "w") as f:
    f.write(reg_code)

print("[patch-mesh-guard] 2/2  regularizer/mesh.py — filtered faces early-return added")
print("\n[patch-mesh-guard] Both guards active. Empty meshes handled at all call sites.")
