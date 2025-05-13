# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import numpy as np
import quaternion
import habitat_sim
import json
from sklearn.neighbors import NearestNeighbors
import cv2
import magnum as mn
import time
import torch
# OpenCV to habitat camera convention transformation
R_OPENCV2HABITAT = np.stack((habitat_sim.geo.RIGHT, -habitat_sim.geo.UP, habitat_sim.geo.FRONT), axis=0)
R_HABITAT2OPENCV = R_OPENCV2HABITAT.T
DEG2RAD = np.pi / 180

def compute_camera_intrinsics(height, width, hfov):
    f = width/2 / np.tan(hfov/2 * np.pi/180)
    cu, cv = width/2, height/2
    return f, cu, cv

def compute_camera_pose_opencv_convention(camera_position, camera_orientation):
    R_cam2world = quaternion.as_rotation_matrix(camera_orientation) @ R_OPENCV2HABITAT
    t_cam2world = np.asarray(camera_position)
    return R_cam2world, t_cam2world

def compute_pointmap(depthmap, hfov):
    """ Compute a HxWx3 pointmap in camera frame from a HxW depth map."""
    height, width = depthmap.shape
    f, cu, cv = compute_camera_intrinsics(height, width, hfov)
    # Cast depth map to point
    z_cam = depthmap
    u, v = np.meshgrid(range(width), range(height))
    x_cam = (u - cu) / f * z_cam
    y_cam = (v - cv) / f * z_cam
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1)
    return X_cam

def compute_pointcloud(depthmap, hfov, camera_position, camera_rotation):
    """Return a 3D point cloud corresponding to valid pixels of the depth map"""
    R_cam2world, t_cam2world = compute_camera_pose_opencv_convention(camera_position, camera_rotation)

    X_cam = compute_pointmap(depthmap=depthmap, hfov=hfov)
    valid_mask = (X_cam[:,:,2] != 0.0)

    X_cam = X_cam.reshape(-1, 3)[valid_mask.flatten()]
    X_world = X_cam @ R_cam2world.T + t_cam2world.reshape(1, 3)
    return X_world

def min_dis(A, B):

    from pytorch3d.ops import knn_points
    dis, _, _ = knn_points(B[None], A[None]) # B querying in A, dis: [1, B.shape[0], 1]
    return dis[0,:,0]

def cover(pc1_, pc2_, thres, device): # querying pc2 in pc1
    import torch
    pc1 = torch.from_numpy(pc1_).to(device).reshape(-1, 3)
    pc2 = torch.from_numpy(pc2_).to(device).reshape(-1, 3)
    
    distances = min_dis(pc1, pc2)
    
    return distances[(distances > 0) * (distances < thres)].shape[0], distances.shape[0]

def compute_pointcloud_overlaps_scikit(pointcloud1, pointcloud2, distance_threshold, compute_symmetric=False):
    """
    Compute 'overlapping' metrics based on a distance threshold between two point clouds.
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm = 'kd_tree').fit(pointcloud2)
    distances, indices = nbrs.kneighbors(pointcloud1)
    intersection1 = np.count_nonzero(distances.flatten() < distance_threshold)

    data = {"intersection1": intersection1,
            "size1": len(pointcloud1)} # should be query size
    if compute_symmetric:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm = 'kd_tree').fit(pointcloud1)
        distances, indices = nbrs.kneighbors(pointcloud2)
        intersection2 = np.count_nonzero(distances.flatten() < distance_threshold)
        data["intersection2"] = intersection2
        data["size2"] = len(pointcloud2)

    return data

def _append_camera_parameters(observation, hfov, camera_location, camera_rotation):
    """
    Add camera parameters to the observation dictionnary produced by Habitat-Sim
    In-place modifications.
    """
    R_cam2world, t_cam2world = compute_camera_pose_opencv_convention(camera_location, camera_rotation)
    height, width = observation['depth'].shape
    f, cu, cv = compute_camera_intrinsics(height, width, hfov)
    K = np.asarray([[f, 0, cu],
                    [0, f, cv],
                    [0, 0, 1.0]])
    observation["camera_intrinsics"] = K
    observation["t_cam2world"] = t_cam2world
    observation["R_cam2world"] = R_cam2world

def look_at(eye, center, up, return_cam2world=True):
    """
    Return camera pose looking at a given center point.
    Analogous of gluLookAt function, using OpenCV camera convention.
    """
    z = center - eye
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    y = -up
    y = y - np.sum(np.asarray(y) * z, axis=-1, keepdims=True) * z
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    x = np.cross(y, z, axis=-1)

    if return_cam2world:
        R = np.stack((x, y, z), axis=-1)
        t = eye
    else:
        # World to camera transformation
        # Transposed matrix
        R = np.stack((x, y, z), axis=-2)
        t = - np.einsum('...ij, ...j', R, eye)
    return R, t

def look_at_for_habitat(eye, center, up, return_cam2world=True):
    R, t = look_at(eye, center, up)
    orientation = quaternion.from_rotation_matrix(R @ R_OPENCV2HABITAT.T)
    return orientation, t

def generate_orientation_noise(pan_range, tilt_range, roll_range):
    return (quaternion.from_rotation_vector(np.random.uniform(*pan_range) * DEG2RAD * habitat_sim.geo.UP)
            * quaternion.from_rotation_vector(np.random.uniform(*tilt_range) * DEG2RAD * habitat_sim.geo.RIGHT)
            * quaternion.from_rotation_vector(np.random.uniform(*roll_range) * DEG2RAD * habitat_sim.geo.FRONT))

def compute_overlap_ratio(self, pcd_world, R, t, depth_map, f, c, depth_threshold=0.05):
    """
    计算点云与深度图的重合比例。

    参数：
    - pcd_world: 点云，形状为 (N, 3)，在世界坐标系中
    - depth_map: 深度图，形状为 (H, W)
    - intrinsics: 相机内参矩阵，形状为 (3, 3)
    - extrinsics: 相机外参矩阵，形状为 (4, 4)，从世界坐标系到相机坐标系的转换
    - depth_threshold: 深度差异的阈值，默认 0.05 米

    返回：
    - overlap_ratio: 重合比例，介于 0 到 1 之间
    """
    pcd_world = torch.from_numpy(pcd_world).to(self.device)
    R = torch.from_numpy(R).to(self.device)
    t = torch.from_numpy(t).to(self.device)
    depth_map = torch.from_numpy(depth_map).to(self.device)
    # # 将点云从世界坐标系转换到相机坐标系
    # # 添加一列 1，变为齐次坐标
    # ones = torch.ones((pcd_world.shape[0], 1), device=device)
    # pcd_world_homogeneous = torch.cat([pcd_world, ones], dim=1)  # (N, 4)

    # # 计算相机坐标系下的点云
    # extrinsics_inv = torch.inverse(extrinsics)  # 如果 extrinsics 是从相机到世界，需要取逆
    # pcd_camera_homogeneous = (extrinsics_inv @ pcd_world_homogeneous.t()).t()  # (N, 4)
    # X_cam @ R_cam2world.T + t_cam2world.reshape(1, 3)
    pcd_camera = (pcd_world - t.reshape(1, 3)) @ R
    total_count = pcd_world.shape[0]

    # 过滤在相机后方的点（z <= 0）
    valid_indices = pcd_camera[:, 2] > 0
    pcd_camera = pcd_camera[valid_indices]

    x = pcd_camera[:, 0]
    y = pcd_camera[:, 1]
    z = pcd_camera[:, 2]

    u = (x * f) / z + c
    v = (y * f) / z + c

    # 将像素坐标转换为整数索引
    u_int = torch.round(u).long()
    v_int = torch.round(v).long()

    # 获取图像的尺寸
    H, W = depth_map.shape

    # 过滤掉超出图像范围的点
    valid_u = (u_int >= 0) & (u_int < W)
    valid_v = (v_int >= 0) & (v_int < H)
    valid_indices = valid_u & valid_v

    u_int = u_int[valid_indices]
    v_int = v_int[valid_indices]
    z = z[valid_indices]

    # 获取深度图中的深度值
    depth_map_flat = depth_map.view(-1)
    depth_indices = v_int * W + u_int
    depth_values = depth_map_flat[depth_indices]

    # 比较深度值
    depth_diff = z - depth_values
    valid_depth = depth_values > 0  # 深度图中有效的深度值
    overlap = (depth_diff < depth_threshold) & valid_depth
    mask = torch.zeros_like(depth_map_flat)
    mask[depth_indices] = overlap.float()
    # 计算重合比例
    overlap_count = overlap.sum().item()
    

    overlap_ratio = overlap_count / total_count

    return overlap_ratio, mask

class NoNaviguableSpaceError(RuntimeError):
    def __init__(self, *args):
            super().__init__(*args)

class MultiviewHabitatSimGenerator:
    def __init__(self,
                scene,
                navmesh,
                scene_dataset_config_file,
                resolution = (240, 320),
                rerender_resolution = (512, 512),
                views_count=2,
                hfov = 60,
                gpu_id = 0,
                size = 10000,
                minimum_covisibility = 0.5,
                maximum_covisibility = 1.0,
                transform = None,
                device = "cpu",
                random_step_variance = 2.0,
                random_step_variance_render = 0.1 / 1.4,
                height_render_delta = 0.05 / 1.4,
                sample_type = "first",
                n_render = 0,
                render_overlap = 0.95,
                ):
        self.scene = scene
        self.navmesh = navmesh
        self.scene_dataset_config_file = scene_dataset_config_file
        self.resolution = resolution
        self.rerender_resolution = rerender_resolution
        self.views_count = views_count
        self.n_render = n_render
        self.n_inference = views_count - n_render
        self.render_overlap = render_overlap
        # print('views count', self.views_count)
        assert(self.views_count >= 1)
        self.hfov = hfov
        self.gpu_id = gpu_id
        self.size = size
        self.transform = transform
        self.sample_type = sample_type

        # Noise added to camera orientation
        self.pan_range = (-3, 3)
        self.tilt_range = (-10, 10)
        self.roll_range = (-5, 5)

        # Height range to sample cameras
        self.height_range = (1.2, 1.8)
        self.height_render_delta = height_render_delta

        # Random steps between the camera views
        self.random_steps_count = 5
        self.random_step_variance = random_step_variance
        self.random_step_variance_render = random_step_variance_render
        

        # Minimum fraction of the scene which should be valid (well defined depth)
        self.minimum_valid_fraction = 0.7

        # Distance threshold to see  to select pairs
        self.distance_threshold = 0.05
        # Minimum IoU of a view point cloud with respect to the reference view to be kept.
        self.minimum_covisibility = minimum_covisibility
        self.maximum_covisibility = maximum_covisibility

        # Maximum number of retries.
        self.max_attempts_count = 5000

        self.seed = None
        self._lazy_initialization()
        self.device = device

        self.render_range = [
            [4 * i, 4 * (i + 1)]
            for i in range(6)
        ]

    def _lazy_initialization(self):
        # Lazy random seeding and instantiation of the simulator to deal with multiprocessing properly
        if self.seed == None:
            # Re-seed numpy generator
            np.random.seed()
            self.seed = np.random.randint(2**32-1)
            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.scene_id = self.scene
            if self.scene_dataset_config_file is not None and self.scene_dataset_config_file != "":
                    sim_cfg.scene_dataset_config_file = self.scene_dataset_config_file
            sim_cfg.random_seed = self.seed
            sim_cfg.load_semantic_mesh = False
            sim_cfg.gpu_device_id = self.gpu_id

            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = self.resolution
            depth_sensor_spec.hfov = self.hfov
            depth_sensor_spec.position = mn.Vector3(0.0, 0.0, 0)
            depth_sensor_spec.orientation

            rgb_sensor_spec = habitat_sim.CameraSensorSpec()
            rgb_sensor_spec.uuid = "color"
            rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            rgb_sensor_spec.resolution = self.resolution
            rgb_sensor_spec.hfov = self.hfov
            rgb_sensor_spec.position = mn.Vector3(0.0, 0.0, 0)
            agent_cfg = habitat_sim.agent.AgentConfiguration(sensor_specifications=[rgb_sensor_spec, depth_sensor_spec])

            # self.cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
            # self.test_sim = habitat_sim.Simulator(self.cfg)
            # self.sim = self.test_sim
            # self.rerender_state = False

            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = self.rerender_resolution
            depth_sensor_spec.hfov = self.hfov
            depth_sensor_spec.position = mn.Vector3(0.0, 0.0, 0)
            depth_sensor_spec.orientation

            rgb_sensor_spec = habitat_sim.CameraSensorSpec()
            rgb_sensor_spec.uuid = "color"
            rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            rgb_sensor_spec.resolution = self.rerender_resolution
            rgb_sensor_spec.hfov = self.hfov
            rgb_sensor_spec.position = mn.Vector3(0.0, 0.0, 0)
            agent_cfg_rerender = habitat_sim.agent.AgentConfiguration(sensor_specifications=[rgb_sensor_spec, depth_sensor_spec])
            self.agent_id = 0

            self.cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg, agent_cfg_rerender])
            # print('rerender_resolution', self.rerender_resolution)
            # input()
            self.sim = habitat_sim.Simulator(self.cfg)
            
            if self.navmesh is not None and self.navmesh != "":
                # Use pre-computed navmesh when available (usually better than those generated automatically)
                self.sim.pathfinder.load_nav_mesh(self.navmesh)

            if not self.sim.pathfinder.is_loaded:
                # Try to compute a navmesh
                navmesh_settings = habitat_sim.NavMeshSettings()
                navmesh_settings.set_defaults()
                self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings, True)

            # Ensure that the navmesh is not empty
            if not self.sim.pathfinder.is_loaded:
                raise NoNaviguableSpaceError(f"No naviguable location (scene: {self.scene} -- navmesh: {self.navmesh})")

            self.agent = [self.sim.initialize_agent(agent_id=0), self.sim.initialize_agent(agent_id=1)]

    def close(self):
        self.sim.close()

    def __del__(self):
        self.sim.close()

    def __len__(self):
        return self.size

    def sample_random_viewpoint(self):
        """ Sample a random viewpoint using the navmesh """
        nav_point = self.sim.pathfinder.get_random_navigable_point()

        # Sample a random viewpoint height
        viewpoint_height = np.random.uniform(*self.height_range)
        viewpoint_position = nav_point + viewpoint_height * habitat_sim.geo.UP
        viewpoint_orientation = quaternion.from_rotation_vector(np.random.uniform(0, 2 * np.pi) * habitat_sim.geo.UP) * generate_orientation_noise(self.pan_range, self.tilt_range, self.roll_range)
        return viewpoint_position, viewpoint_orientation, nav_point, viewpoint_height

    def sample_other_random_viewpoint(self, observed_point, nav_point, only_render = False, previous_poh = None):
        """ Sample a random viewpoint close to an existing one, using the navmesh and a reference observed point."""
        other_nav_point = nav_point
        if only_render:
            walk_directions = self.random_step_variance_render * np.asarray([1,0,1])
        else:
            walk_directions = self.random_step_variance * np.asarray([1,0,1])
        for i in range(self.random_steps_count):
            # print('step', i)
            temp = self.sim.pathfinder.snap_point(other_nav_point + walk_directions * np.random.normal(size=3))
            # Snapping may return nan when it fails
            if not np.isnan(temp[0]):
                    other_nav_point = temp
        if only_render:
            height = previous_poh[-1]
            h_range = [height - self.height_render_delta, height + self.height_render_delta]
            other_viewpoint_height = np.random.uniform(*h_range)
        else:
            other_viewpoint_height = np.random.uniform(*self.height_range)
            
        other_viewpoint_position = other_nav_point + other_viewpoint_height * habitat_sim.geo.UP

        # Set viewing direction towards the central point
        if only_render:
            position = other_viewpoint_position
            rotation = previous_poh[1]
        else:
            rotation, position = look_at_for_habitat(eye=other_viewpoint_position, center=observed_point, up=habitat_sim.geo.UP, return_cam2world=True)
            rotation = rotation * generate_orientation_noise(self.pan_range, self.tilt_range, self.roll_range)
        return position, rotation, other_nav_point, other_viewpoint_height

    def is_other_pointcloud_overlapping(self, ref_pointcloud, other_pointcloud):
        """ Check if a viewpoint is valid and overlaps significantly with a reference one. """
        # Observation
        pixels_count = self.resolution[0] * self.resolution[1]
        valid_fraction = len(other_pointcloud) / pixels_count
        assert valid_fraction <= 1.0 and valid_fraction >= 0.0
        overlap = compute_pointcloud_overlaps_scikit(ref_pointcloud, other_pointcloud, self.distance_threshold, compute_symmetric=True)
        covisibility = (overlap["intersection1"] / pixels_count + overlap["intersection2"] / pixels_count) / 2
        is_valid = (valid_fraction >= self.minimum_valid_fraction) and (covisibility >= self.minimum_covisibility)
        return is_valid, valid_fraction, covisibility

    # def compute_overlap_ratio(pcd_world, R, t, depth_map, f, c, depth_threshold=0.05):
    def except_i(self, a, id):
        return [a[i] for i in range(len(a)) if i != id]

    def is_other_pointcloud_overlappings2(self, ref_pcd, other_pcds, ref_depth, other_depths, ref_Rt, other_Rts, f, c, is_valid, valid_fraction, contact_id = None):
        """ Check if a viewpoint is valid and overlaps significantly with a reference one. """
        # Observation
        covs = []
        masks = []
        R_r, t_r = ref_Rt
        for Rt, depth, pcd in zip(other_Rts, other_depths, other_pcds):
            R, t = Rt
            overlap, _ =  compute_overlap_ratio(self, ref_pcd, R, t, depth, f, c)
            overlap2, mask = compute_overlap_ratio(self,     pcd, R_r, t_r, ref_depth, f, c)
            masks.append(mask)
            covisibility = (overlap + overlap2) / 2
            covs.append(covisibility)
        covisibility = np.max(covs)
        if type(contact_id) is not list:
            if self.sample_type == "chain":
                covisibility = covs[-1]
                others_cov = 0 if len(covs) == 1 else np.max(covs[:-1])
                cov_ok = (covisibility > self.minimum_covisibility and covisibility < self.maximum_covisibility)
                cov_ok = others_cov < self.minimum_covisibility and cov_ok
            elif self.sample_type == "tree" or self.sample_type == "tree_diverse_first":
                covisibility = covs[contact_id]
                others_cov = 0 if len(covs) == 1 else np.max(self.except_i(covs, contact_id))
                cov_ok = (covisibility > self.minimum_covisibility and covisibility < self.maximum_covisibility)
                cov_ok = others_cov < self.minimum_covisibility and cov_ok
                # cov_ok = others_cov < 0.6 and cov_ok
            elif self.sample_type == "tree_loose":
                covisibility = covs[contact_id]
                cov_ok = (covisibility > self.minimum_covisibility and covisibility < self.maximum_covisibility)
            else:
                cov_ok = (covisibility > self.minimum_covisibility and covisibility < self.maximum_covisibility)
        else: # considering only_render views now
            masks = torch.stack(masks[contact_id[0]:contact_id[1]], 0) # [n, -1]
            masks = masks.sum(0) > 0.5
            covisibility = masks.sum().item() / masks.shape[0]
            cov_ok = covisibility > self.render_overlap
            
        is_valid = (is_valid and cov_ok)
        return is_valid, valid_fraction, covisibility, covs

    def is_other_pointcloud_overlappings_tiny(self, ref_pointclouds, other_pointcloud, other_depth):
        """ Check if a viewpoint is valid and overlaps significantly with a reference one. """
        # Observation
        pixels_count = self.resolution[0] * self.resolution[1]
        valid_fraction = len(other_pointcloud) / pixels_count
        is_depth_valid = (other_depth < 0.5).mean() < 0.3
        is_valid = (valid_fraction >= self.minimum_valid_fraction) & is_depth_valid
        return is_valid, valid_fraction

    def is_other_pointcloud_overlappings(self, ref_pointclouds, other_pointcloud):
        """ Check if a viewpoint is valid and overlaps significantly with a reference one. """
        # Observation
        pixels_count = self.resolution[0] * self.resolution[1]
        valid_fraction = len(other_pointcloud) / pixels_count
        is_valid = (valid_fraction >= self.minimum_valid_fraction)
        covs = []
        for ref_pointcloud in ref_pointclouds:
            assert valid_fraction <= 1.0 and valid_fraction >= 0.0
            overlap = compute_pointcloud_overlaps_scikit(ref_pointcloud, other_pointcloud, self.distance_threshold, compute_symmetric=True)
            covisibility = (overlap["intersection1"] / overlap["size1"] + overlap["intersection2"] / overlap["size2"]) / 2
            covs.append(covisibility)
        covisibility = np.max(covs)
        cov_ok = (covisibility > self.minimum_covisibility and covisibility < self.maximum_covisibility)
        is_valid = (is_valid and cov_ok)
        return is_valid, valid_fraction, covisibility

    def is_other_viewpoint_overlapping(self, ref_pointcloud, observation, position, rotation):
        """ Check if a viewpoint is valid and overlaps significantly with a reference one. """
        # Observation
        other_pointcloud = compute_pointcloud(observation['depth'], self.hfov, position, rotation)
        return self.is_other_pointcloud_overlapping(ref_pointcloud, other_pointcloud)
    
    def set_rerender_state(self, state):
        self.agent_id = int(state)
    
    # def set # TODO set rerender only once 
    def render_viewpoint(self, viewpoint_position, viewpoint_orientation, rerender = False):
        self.set_rerender_state(rerender)
        agent_state = habitat_sim.AgentState()
        agent_state.position = viewpoint_position
        agent_state.rotation = viewpoint_orientation
        self.agent[self.agent_id].set_state(agent_state)
        # print(self.agent_id)
        viewpoint_observations = self.sim.get_sensor_observations(agent_ids=self.agent_id)
        _append_camera_parameters(viewpoint_observations, self.hfov, viewpoint_position, viewpoint_orientation)
        return viewpoint_observations

    def __getitem__(self, useless_idx):
        while 1:
            ref_position, ref_orientation, nav_point, height = self.sample_random_viewpoint()
            po_history = [[ref_position, ref_orientation, height]]
            ref_observations = self.render_viewpoint(ref_position, ref_orientation)
            if self.sample_type == "tree_diverse_first":
                break
            if ref_observations['depth'].mean() > 1.0 and (ref_observations['depth'] < 0.5).mean() < 0.3:
                break
        # Extract point cloud
        ref_pointcloud = compute_pointcloud(depthmap=ref_observations['depth'], hfov=self.hfov,
                                        camera_position=ref_position, camera_rotation=ref_orientation)

        pixels_count = self.resolution[0] * self.resolution[1]
        ref_valid_fraction = len(ref_pointcloud) / pixels_count
        assert ref_valid_fraction <= 1.0 and ref_valid_fraction >= 0.0
        if ref_valid_fraction < self.minimum_valid_fraction:
                # This should produce a recursion error at some point when something is very wrong.
                return self[0]
        # Pick an reference observed point in the point cloud
        observed_point = np.mean(ref_pointcloud, axis=0)

        # Add the first image as reference
        all_depth = [ref_observations['depth']]
        all_Rt = [[ref_observations['R_cam2world'], ref_observations['t_cam2world']]]
        all_nav_point = [nav_point]
        all_height = [height]
        all_obs_point = [observed_point]

        viewpoints_observations = [ref_observations]
        viewpoints_covisibility = [ref_valid_fraction]
        viewpoints_positions = [ref_position]
        viewpoints_orientations = [quaternion.as_float_array(ref_orientation)]
        viewpoints_clouds = [ref_pointcloud]
        viewpoints_valid_fractions = [ref_valid_fraction]
        pairwise_visibility_ratios = np.eye(self.views_count, dtype=float)
        

        for view_id in range(self.views_count - 1):
            print(view_id)
            real_view_id = view_id + 1
            # Generate an other viewpoint using some dummy random walk
            successful_sampling = False
            for sampling_attempt in range(self.max_attempts_count):
                only_render = real_view_id >= self.n_inference
                # print(sampling_attempt, view_id)
                contact_id = None
                if self.sample_type == "all_rnd":
                    position, rotation, other_nav_point, height = self.sample_random_viewpoint()
                elif self.sample_type == "first":
                    position, rotation, other_nav_point, height = self.sample_other_random_viewpoint(observed_point, nav_point, only_render)
                elif self.sample_type == "random" or self.sample_type == "tree" or self.sample_type == "tree_diverse_first" or self.sample_type == "tree_old" or self.sample_type == "tree_loose":
                    if only_render:
                        view_id_render = real_view_id - self.n_inference
                        previous_id = self.render_range[view_id_render][0] + np.random.randint(self.render_range[view_id_render][1] - self.render_range[view_id_render][0])
                        contact_id = previous_id
                    else:
                        previous_id = np.random.randint(real_view_id)
                        contact_id = previous_id
                    previous_po = po_history[previous_id]
                    position, rotation, other_nav_point, height = self.sample_other_random_viewpoint(all_obs_point[previous_id], all_nav_point[previous_id], only_render, previous_po)
                elif self.sample_type == "chain":
                    previous_id = view_id
                    position, rotation, other_nav_point, height = self.sample_other_random_viewpoint(all_obs_point[previous_id], all_nav_point[previous_id], only_render)
                # Observation
                t = [time.time()]
                other_viewpoint_observations = self.render_viewpoint(position, rotation)
                ref_Rt = [other_viewpoint_observations['R_cam2world'], other_viewpoint_observations['t_cam2world']]
                t.append(time.time())
                other_pointcloud = compute_pointcloud(other_viewpoint_observations['depth'], self.hfov, position, rotation)
                f, c, _ = compute_camera_intrinsics(other_viewpoint_observations['depth'].shape[0], other_viewpoint_observations['depth'].shape[1], self.hfov)
                t.append(time.time())

                # is_valid, valid_fraction, covisibility = self.is_other_pointcloud_overlapping(ref_pointcloud, other_pointcloud)
                # is_valid, valid_fraction, covisibility = self.is_other_pointcloud_overlappings(viewpoints_clouds, other_pointcloud)
                is_valid, valid_fraction = self.is_other_pointcloud_overlappings_tiny(viewpoints_clouds, other_pointcloud, other_viewpoint_observations['depth'])
                if not is_valid:
                    continue
                # def is_other_pointcloud_overlappings2(self, ref_pcd, other_pcds, ref_depth, other_depths, ref_Rt, other_Rts, f, c)
                t.append(time.time())
                if only_render:
                    view_id_render = real_view_id - self.n_inference
                    contact_id = [self.render_range[view_id_render][0], self.render_range[view_id_render][1]]
                is_valid, valid_fraction, covisibility, covs = self.is_other_pointcloud_overlappings2(other_pointcloud, viewpoints_clouds, other_viewpoint_observations['depth'], all_depth, ref_Rt, all_Rt, f, c, is_valid, valid_fraction, contact_id = contact_id)
                pairwise_visibility_ratios[view_id+1,:view_id+1] = covs
                pairwise_visibility_ratios[:view_id+1,view_id+1] = covs
                
                t.append(time.time())
                # print('cov', covisibility, cov_new)
                # print('all ts', t[1] - t[0], t[2] - t[1], t[3] - t[2], t[4] - t[3]) # 0.0020208358764648438 0.0007464885711669922 0.5119760036468506
                if is_valid:
                        successful_sampling = True
                        break
            if not successful_sampling:
                print("WARNING: Maximum number of attempts reached.")
                # Dirty hack, try using a novel original viewpoint
                return self[0]
            po_history.append([position, rotation, height])
            viewpoints_observations.append(other_viewpoint_observations)
            viewpoints_covisibility.append(covisibility)
            viewpoints_positions.append(position)
            viewpoints_orientations.append(quaternion.as_float_array(rotation)) # WXYZ convention for the quaternion encoding.
            viewpoints_clouds.append(other_pointcloud)
            viewpoints_valid_fractions.append(valid_fraction)
            all_depth.append(other_viewpoint_observations['depth'])
            all_Rt.append(ref_Rt)
            all_nav_point.append(other_nav_point)
            all_height.append(height)
            all_obs_point.append(observed_point)
        for i in range(len(viewpoints_observations)):
            # print('rerendering', i)
            viewpoints_observations[i] = self.render_viewpoint(po_history[i][0], po_history[i][1], rerender = True)
        # Estimate relations between all pairs of images
        # pairwise_visibility_ratios = np.ones((len(viewpoints_observations), len(viewpoints_observations)))
        # for i in range(len(viewpoints_observations)):
        #     pairwise_visibility_ratios[i,i] = viewpoints_valid_fractions[i]
        #     for j in range(i+1, len(viewpoints_observations)):
        #         overlap = compute_pointcloud_overlaps_scikit(viewpoints_clouds[i], viewpoints_clouds[j], self.distance_threshold, compute_symmetric=True)
        #         score = (overlap['intersection1'] / overlap['size1'] + overlap['intersection2'] / overlap['size2']) / 2
        #         pairwise_visibility_ratios[i,j] = score
        #         pairwise_visibility_ratios[j,i] = score

        # IoU is relative to the image 0
        # viewpoints_observations_ = []
        for obs in viewpoints_observations:
            f, cu, cv = compute_camera_intrinsics(self.rerender_resolution[0], self.rerender_resolution[1], self.hfov)
            K = np.asarray([[f, 0, cu],
                    [0, f, cv],
                    [0, 0, 1.0]])
            obs['camera_intrinsics'] = K
        #     viewpoints_observations_.append(obs)
        # viewpoints_observations = viewpoints_observations_

        data = {"observations": viewpoints_observations,
                "positions": np.asarray(viewpoints_positions),
                "orientations": np.asarray(viewpoints_orientations),
                "covisibility_ratios": np.asarray(viewpoints_covisibility),
                "valid_fractions": np.asarray(viewpoints_valid_fractions, dtype=float),
                "pairwise_visibility_ratios": np.asarray(pairwise_visibility_ratios, dtype=float),
                }

        if self.transform is not None:
            data = self.transform(data)
        return  data

    def generate_random_spiral_trajectory(self, images_count = 100, max_radius=0.5, half_turns=5, use_constant_orientation=False):
        """
        Return a list of images corresponding to a spiral trajectory from a random starting point.
        Useful to generate nice visualisations.
        Use an even number of half turns to get a nice "C1-continuous" loop effect 
        """
        ref_position, ref_orientation, navpoint = self.sample_random_viewpoint()
        ref_observations = self.render_viewpoint(ref_position, ref_orientation)
        ref_pointcloud = compute_pointcloud(depthmap=ref_observations['depth'], hfov=self.hfov,
                                                        camera_position=ref_position, camera_rotation=ref_orientation)
        pixels_count = self.resolution[0] * self.resolution[1]
        if len(ref_pointcloud) / pixels_count < self.minimum_valid_fraction:
            # Dirty hack: ensure that the valid part of the image is significant
            return self.generate_random_spiral_trajectory(images_count, max_radius, half_turns, use_constant_orientation)

        # Pick an observed point in the point cloud
        observed_point = np.mean(ref_pointcloud, axis=0)
        ref_R, ref_t = compute_camera_pose_opencv_convention(ref_position, ref_orientation)

        images = []
        is_valid = []
        # Spiral trajectory, use_constant orientation
        for i, alpha in enumerate(np.linspace(0, 1, images_count)):
            r = max_radius * np.abs(np.sin(alpha * np.pi)) # Increase then decrease the radius
            theta = alpha * half_turns * np.pi 
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = 0.0
            position = ref_position + (ref_R @ np.asarray([x, y, z]).reshape(3,1)).flatten()
            if use_constant_orientation:
                orientation = ref_orientation
            else:
                # trajectory looking at a mean point in front of the ref observation
                orientation, position = look_at_for_habitat(eye=position, center=observed_point, up=habitat_sim.geo.UP)
            observations = self.render_viewpoint(position, orientation)
            images.append(observations['color'][...,:3])
            _is_valid, valid_fraction, iou = self.is_other_viewpoint_overlapping(ref_pointcloud, observations, position, orientation)
            is_valid.append(_is_valid)
        return images, np.all(is_valid)
