#!/usr/bin/env python3
"""
Patch MILo to lazy-load training images from disk instead of pre-loading all
into RAM. This reduces startup memory from ~20GB to <1GB for 400+ camera scenes.

Run at Docker build time (BEFORE other patches):
    python3 /workspace/patch_lazy_images.py

Patches two files:
  1. scene/cameras.py — Camera.__init__ stores lazy fields; property + method appended
  2. utils/camera_utils.py — loadCam enables lazy loading after construction
"""

import sys

# ── Patch 1a: cameras.py — modify __init__ to add lazy fields ────────────────

CAMERAS_PY = "/workspace/MILo/milo/scene/cameras.py"

with open(CAMERAS_PY, "r") as f:
    cam_code = f.read()

# Replace the image storage in __init__ to add lazy fields
_INIT_OLD = """\
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]"""

_INIT_NEW = """\
        self._image_path = None
        self._image_resolution = None
        self._original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self._original_image.shape[2]
        self.image_height = self._original_image.shape[1]"""

assert _INIT_OLD in cam_code, "Camera __init__ patch anchor not found"
cam_code = cam_code.replace(_INIT_OLD, _INIT_NEW, 1)

# ── Patch 1b: Append property + method AFTER the Camera class ────────────────
# We monkey-patch because injecting methods inside __init__ breaks indentation.

_AFTER_CAMERA_CLASS = """\
class MiniCam:"""

_LAZY_METHODS_AND_MINICAM = """\
# --- Lazy image loading (patched by patch_lazy_images.py) ---
def _camera_original_image_getter(self):
    if self._original_image is not None:
        return self._original_image
    if self._image_path is None:
        raise RuntimeError("Camera has no image and no image_path set")
    from utils.general_utils import PILtoTorch
    from PIL import Image as _PILImage
    _pil = _PILImage.open(self._image_path)
    if len(_pil.split()) > 3:
        import torch as _torch
        _t = _torch.cat([PILtoTorch(ch, self._image_resolution) for ch in _pil.split()[:3]], dim=0)
    else:
        _t = PILtoTorch(_pil, self._image_resolution)
    return _t.clamp(0.0, 1.0).to(self.data_device)

Camera.original_image = property(_camera_original_image_getter)

def _camera_enable_lazy(self, image_path, resolution):
    self._image_path = image_path
    self._image_resolution = resolution
    self.image_width = resolution[0]
    self.image_height = resolution[1]
    self._original_image = None  # free the tensor

Camera.enable_lazy_loading = _camera_enable_lazy


class MiniCam:"""

assert _AFTER_CAMERA_CLASS in cam_code, "MiniCam anchor not found"
cam_code = cam_code.replace(_AFTER_CAMERA_CLASS, _LAZY_METHODS_AND_MINICAM, 1)

with open(CAMERAS_PY, "w") as f:
    f.write(cam_code)

print("[patch-lazy] 1/2  cameras.py — lazy original_image + enable_lazy_loading added")

# ── Patch 2: camera_utils.py — enable lazy after construction ────────────────

CAMUTILS_PY = "/workspace/MILo/milo/utils/camera_utils.py"

with open(CAMUTILS_PY, "r") as f:
    utils_code = f.read()

_UTILS_OLD = (
    "    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, \n"
    "                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, \n"
    "                  image=gt_image, gt_alpha_mask=loaded_mask,\n"
    "                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)"
)

_UTILS_NEW = (
    "    cam = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, \n"
    "                 FoVx=cam_info.FovX, FoVy=cam_info.FovY, \n"
    "                 image=gt_image, gt_alpha_mask=loaded_mask,\n"
    "                 image_name=cam_info.image_name, uid=id, data_device=args.data_device)\n"
    "    # Switch to lazy loading: free the image tensor, reload from disk on access\n"
    "    if hasattr(cam_info, 'image_path') and cam_info.image_path and args.data_device == 'cpu':\n"
    "        cam.enable_lazy_loading(cam_info.image_path, resolution)\n"
    "    return cam"
)

assert _UTILS_OLD in utils_code, "camera_utils patch anchor not found"
utils_code = utils_code.replace(_UTILS_OLD, _UTILS_NEW, 1)

with open(CAMUTILS_PY, "w") as f:
    f.write(utils_code)

print("[patch-lazy] 2/2  camera_utils.py — lazy loading enabled after Camera construction")

print("\n[patch-lazy] Done. Images loaded from disk on access when data_device=cpu.")
print("[patch-lazy] Pre-loaded mode (data_device=cuda) is unchanged.")
