#!/usr/bin/env python3
"""
Patch MILo's mesh regularization to handle empty meshes gracefully.

Run once at Docker build time (AFTER other patches):
    python3 /workspace/patch_mesh_empty_guard.py

When marching tetrahedra + frustum/edge filtering produces zero triangles,
nvdiffrast crashes with: RuntimeError: tri must have shape [>0, 3]

This patch adds a guard at the mesh construction point in
regularization/regularizer/mesh.py: if the filtered faces tensor is empty,
skip mesh rendering and return zero losses for that iteration.
"""

MESH_REG_PY = "/workspace/MILo/milo/regularization/regularizer/mesh.py"

with open(MESH_REG_PY, "r") as f:
    code = f.read()

# The crash happens when faces[faces_mask] is empty. Guard right before mesh_renderer call.
# We wrap the mesh build + render + loss computation in a check.

_OLD = """\
        # --- Build and Render Mesh ---
        mesh = Meshes(verts=verts, faces=faces[faces_mask])

        mesh_render_pkg = mesh_renderer("""

_NEW = """\
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

assert _OLD in code, "Empty mesh guard patch anchor not found in mesh.py"
code = code.replace(_OLD, _NEW, 1)

with open(MESH_REG_PY, "w") as f:
    f.write(code)

print("[patch-mesh-guard] Empty mesh guard added to regularization/regularizer/mesh.py")
print("[patch-mesh-guard] Zero-triangle meshes now skip rendering instead of crashing.")
