#!/usr/bin/env python3
"""
Patch MILo to handle empty meshes gracefully in nvdiffrast.

Run once at Docker build time (AFTER other patches):
    python3 /workspace/patch_mesh_empty_guard.py

When marching tetrahedra + filtering produces zero triangles, nvdiffrast
crashes with: RuntimeError: tri must have shape [>0, 3]
This affects dr.rasterize, dr.interpolate, and dr.antialias — all require
non-empty faces tensors.

This patch guards at FOUR levels:
  1. nvdiff_rasterization() — returns zero tensors for empty faces
  2. MeshRenderer.forward() — returns empty output dict for empty meshes
  3. ScalableMeshRenderer.forward() — same guard for scalable variant
     (used by evaluate_mesh_occupancy when use_scalable_renderer=True)
  4. compute_mesh_regularization() — early return with zero losses
"""

import sys

# ── Patch 1: Guard nvdiff_rasterization ───────────────────────────────────────

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

# ── Patch 1b: Guard MeshRenderer.forward ──────────────────────────────────────
# MeshRenderer.forward calls dr.interpolate and dr.antialias with mesh.faces
# AFTER rasterization. Even if rasterize returns zeros, these crash on empty faces.
# Guard at the top of forward() to return empty output immediately.

# Use a minimal anchor that won't break on whitespace differences
_P1B_OLD = (
    "        fragments, rast_out, pos = self.rasterizer(mesh, cameras, cam_idx, return_rast_out=True, return_positions=True)\n"
    "        if cameras is None:\n"
    "            cameras = self.rasterizer.cameras"
)

_P1B_NEW = (
    "        # Guard: empty mesh crashes dr.interpolate and dr.antialias\n"
    "        if mesh.faces.shape[0] == 0:\n"
    "            if cameras is None:\n"
    "                cameras = self.rasterizer.cameras\n"
    "            if isinstance(cameras, Camera):\n"
    "                _cam = cameras\n"
    "            else:\n"
    "                _cam = cameras[cam_idx]\n"
    "            _H, _W = _cam.image_height, _cam.image_width\n"
    "            _dev = mesh.verts.device\n"
    "            output_pkg = {}\n"
    "            if return_depth:\n"
    "                output_pkg['depth'] = torch.zeros(1, _H, _W, 1, device=_dev)\n"
    "            if mesh.verts_colors is not None:\n"
    "                output_pkg['rgb'] = torch.zeros(1, _H, _W, 3, device=_dev)\n"
    "            if return_normals:\n"
    "                output_pkg['normals'] = torch.zeros(1, _H, _W, 3, device=_dev)\n"
    "            if return_pix_to_face:\n"
    "                output_pkg['pix_to_face'] = torch.full((1, _H, _W, 1), -1, dtype=torch.long, device=_dev)\n"
    "            return output_pkg\n"
    "        fragments, rast_out, pos = self.rasterizer(mesh, cameras, cam_idx, return_rast_out=True, return_positions=True)\n"
    "        if cameras is None:\n"
    "            cameras = self.rasterizer.cameras"
)

assert _P1B_OLD in code, "Patch 1b anchor not found in scene/mesh.py (MeshRenderer.forward)"
code = code.replace(_P1B_OLD, _P1B_NEW, 1)

# ── Patch 1c: Guard ScalableMeshRenderer.forward ─────────────────────────────
# Same issue as MeshRenderer but for the scalable variant used by
# evaluate_mesh_occupancy when use_scalable_renderer=True. Per-view frustum
# culling can produce 0-face submeshes. Without guard: fragments=None → crash.

_P1C_OLD = (
    "        n_passes = (mesh.faces.shape[0] + max_triangles_in_batch - 1) // max_triangles_in_batch\n"
    "        \n"
    "        fragments = None"
)

_P1C_NEW = (
    "        # Guard: empty mesh crashes nvdiffrast (dr.interpolate, dr.antialias)\n"
    "        if mesh.faces.shape[0] == 0:\n"
    "            if cameras is None:\n"
    "                cameras = self.rasterizer.cameras\n"
    "            if isinstance(cameras, Camera):\n"
    "                _cam = cameras\n"
    "            else:\n"
    "                _cam = cameras[cam_idx]\n"
    "            _H, _W = _cam.image_height, _cam.image_width\n"
    "            _dev = mesh.verts.device\n"
    "            output_pkg = {}\n"
    "            if return_depth:\n"
    "                output_pkg['depth'] = torch.zeros(1, _H, _W, 1, device=_dev)\n"
    "            if mesh.verts_colors is not None:\n"
    "                output_pkg['rgb'] = torch.zeros(1, _H, _W, 3, device=_dev)\n"
    "            if return_normals:\n"
    "                output_pkg['normals'] = torch.zeros(1, _H, _W, 3, device=_dev)\n"
    "            if return_pix_to_face:\n"
    "                output_pkg['pix_to_face'] = torch.full((1, _H, _W, 1), -1, dtype=torch.long, device=_dev)\n"
    "            output_pkg['fragments'] = None\n"
    "            output_pkg['rast_out'] = None\n"
    "            return output_pkg\n"
    "        n_passes = (mesh.faces.shape[0] + max_triangles_in_batch - 1) // max_triangles_in_batch\n"
    "        \n"
    "        fragments = None"
)

assert _P1C_OLD in code, "Patch 1c anchor not found in scene/mesh.py (ScalableMeshRenderer.forward)"
code = code.replace(_P1C_OLD, _P1C_NEW, 1)

with open(MESH_PY, "w") as f:
    f.write(code)

print("[patch-mesh-guard] 1/4  scene/mesh.py — nvdiff_rasterization empty faces guard")
print("[patch-mesh-guard] 2/4  scene/mesh.py — MeshRenderer.forward empty mesh guard")
print("[patch-mesh-guard] 3/4  scene/mesh.py — ScalableMeshRenderer.forward empty mesh guard")

# ── Patch 2: Guard compute_mesh_regularization ────────────────────────────────

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

print("[patch-mesh-guard] 4/4  regularizer/mesh.py — filtered faces early-return")
print("\n[patch-mesh-guard] All guards active. Empty meshes handled at all call sites.")
