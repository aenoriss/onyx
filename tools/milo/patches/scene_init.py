#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

def filter_point_cloud_by_masks(pcd_points, pcd_colors, cameras, threshold=0.3):
    """
    Filter SfM point cloud to remove points primarily visible in masked (sky) regions.
    
    Args:
        pcd_points: (N, 3) numpy array of 3D points
        pcd_colors: (N, 3) numpy array of RGB colors
        cameras: list of camera objects with gt_mask attribute
        threshold: minimum mask visibility to keep point (0-1)
    
    Returns:
        filtered_points, filtered_colors
    """
    import numpy as np
    from tqdm import tqdm
    
    N = len(pcd_points)
    visibility_scores = np.zeros(N)
    visibility_counts = np.zeros(N)
    
    for cam in tqdm(cameras, desc="Filtering point cloud by masks"):
        if not hasattr(cam, 'gt_mask') or cam.gt_mask is None:
            continue
        
        mask = cam.gt_mask
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        
        # Get camera params
        R = cam.R.T if hasattr(cam, 'R') else None
        T = cam.T if hasattr(cam, 'T') else None
        if R is None or T is None:
            continue
            
        # Project points: P_cam = R^T @ (P_world - C) where C = -R^T @ T
        # Simplified: P_cam = R^T @ P_world - R^T @ T... but actually for colmap:
        # world_to_cam: P_cam = R @ P_world + T
        points_cam = (cam.R @ pcd_points.T).T + cam.T
        
        in_front = points_cam[:, 2] > 0.1
        
        # Project to pixels
        fx = cam.image_width / (2 * np.tan(cam.FoVx / 2))
        fy = cam.image_height / (2 * np.tan(cam.FoVy / 2))
        cx, cy = cam.image_width / 2, cam.image_height / 2
        
        with np.errstate(divide='ignore', invalid='ignore'):
            u = (fx * points_cam[:, 0] / points_cam[:, 2] + cx).astype(int)
            v = (fy * points_cam[:, 1] / points_cam[:, 2] + cy).astype(int)
        
        in_bounds = (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height)
        valid = in_front & in_bounds
        
        valid_indices = np.where(valid)[0]
        for i in valid_indices:
            mask_val = float(mask[v[i], u[i]])
            if mask_val > 1:
                mask_val = mask_val / 255.0
            visibility_scores[i] += mask_val
            visibility_counts[i] += 1
    
    avg_visibility = np.zeros(N)
    seen = visibility_counts > 0
    avg_visibility[seen] = visibility_scores[seen] / visibility_counts[seen]
    
    keep_mask = (avg_visibility >= threshold) | (~seen)
    
    removed = (~keep_mask).sum()
    print(f"Point cloud filter: {N:,} -> {keep_mask.sum():,} points (removed {removed:,} sky points, {100*removed/N:.1f}%)")
    
    return pcd_points[keep_mask], pcd_colors[keep_mask], keep_mask



class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # Filter point cloud by mask visibility (remove sky points)
            has_masks = any(hasattr(cam, 'gt_mask') and cam.gt_mask is not None 
                           for cam in self.train_cameras[1.0])
            if has_masks:
                filtered_points, filtered_colors, keep_mask = filter_point_cloud_by_masks(
                    scene_info.point_cloud.points,
                    scene_info.point_cloud.colors,
                    self.train_cameras[1.0],
                    threshold=0.3
                )
                filtered_normals = None
                if scene_info.point_cloud.normals is not None:
                    filtered_normals = scene_info.point_cloud.normals[keep_mask]
                from scene.dataset_readers import BasicPointCloud
                filtered_pcd = BasicPointCloud(
                    points=filtered_points, 
                    colors=filtered_colors, 
                    normals=filtered_normals
                )
                self.gaussians.create_from_pcd(filtered_pcd, self.cameras_extent)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        
        # Auto-filter: remove low-opacity Gaussians in masked (sky) regions
        if hasattr(self, 'train_cameras') and len(self.train_cameras) > 0:
            self._filter_sky_gaussians(opacity_threshold=0.1, visibility_threshold=0.3)
        
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    
    def _filter_sky_gaussians(self, opacity_threshold=0.1, visibility_threshold=0.3):
        """Remove sky Gaussians using opacity + projection + magenta color."""
        import torch
        
        # Get cameras with masks
        cameras = self.getTrainCameras()
        cameras_with_masks = [c for c in cameras if c.gt_mask is not None]
        if len(cameras_with_masks) == 0:
            print("[Auto-filter] No masks found, skipping filter")
            return
        
        xyz = self.gaussians._xyz.detach()
        opacity = torch.sigmoid(self.gaussians._opacity.detach()).squeeze()
        n_gaussians = xyz.shape[0]
        
        print(f"[Auto-filter] Analyzing {n_gaussians} Gaussians...")
        
        # Compute mask visibility for each Gaussian (project to cameras)
        visibility_sum = torch.zeros(n_gaussians, device=xyz.device)
        visibility_count = torch.zeros(n_gaussians, device=xyz.device)
        
        sample_cams = cameras_with_masks[::max(1, len(cameras_with_masks)//100)][:100]
        for cam in sample_cams:
            R = torch.tensor(cam.R, device=xyz.device, dtype=xyz.dtype)
            T = torch.tensor(cam.T, device=xyz.device, dtype=xyz.dtype)
            xyz_cam = xyz @ R.T + T
            
            W, H = cam.image_width, cam.image_height
            z = xyz_cam[:, 2].clamp(min=0.01)
            
            # Simple projection (assumes principal point at center)
            fx = W / (2 * np.tan(cam.FoVx / 2)) if hasattr(cam, 'FoVx') else W
            fy = H / (2 * np.tan(cam.FoVy / 2)) if hasattr(cam, 'FoVy') else H
            u = (xyz_cam[:, 0] / z * fx + W / 2).long()
            v = (xyz_cam[:, 1] / z * fy + H / 2).long()
            
            in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0.1)
            
            mask = cam.gt_mask.to(xyz.device)
            if mask.dim() == 3:
                mask = mask[0]
            
            valid_idx = torch.where(in_bounds)[0]
            u_valid = u[valid_idx].clamp(0, W-1)
            v_valid = v[valid_idx].clamp(0, H-1)
            visibility_sum[valid_idx] += mask[v_valid, u_valid]
            visibility_count[valid_idx] += 1
        
        # Average visibility
        valid = visibility_count > 0
        mask_visibility = torch.ones(n_gaussians, device=xyz.device)
        mask_visibility[valid] = visibility_sum[valid] / visibility_count[valid]
        
        # Check magenta color
        C0 = 0.28209479
        f_dc = self.gaussians._features_dc.detach().squeeze()
        rgb = f_dc * C0 + 0.5
        dist_to_magenta = torch.sqrt((rgb[:, 0] - 1)**2 + (rgb[:, 1] - 0)**2 + (rgb[:, 2] - 1)**2)
        is_magenta = dist_to_magenta < 0.4
        
        # Filter criteria:
        # 1. Low opacity AND projects to sky
        # 2. Magenta color AND projects mostly to sky
        remove_opacity = (opacity < opacity_threshold) & (mask_visibility < visibility_threshold)
        remove_magenta = is_magenta & (mask_visibility < 0.5)
        remove = remove_opacity | remove_magenta
        
        n_opacity = remove_opacity.sum().item()
        n_magenta = (remove_magenta & ~remove_opacity).sum().item()
        n_total = remove.sum().item()
        
        if n_total > 0:
            print(f"[Auto-filter] Removing {n_total} sky Gaussians:")
            print(f"  - {n_opacity} by low opacity + projection")
            print(f"  - {n_magenta} by magenta color + projection")
            self.gaussians.prune_points(remove)
        else:
            print("[Auto-filter] No sky Gaussians found to remove")

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getTrainCameras_warn_up(self, iteration, warn_until_iter, scale=1.0, scale2=2.0):
        if iteration<=warn_until_iter:
            return self.train_cameras[scale2]
        else:
            return self.train_cameras[scale]