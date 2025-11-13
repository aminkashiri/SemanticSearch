# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import torch
import numpy as np
import torch.nn as nn
import skimage.morphology
from skimage.draw import disk
import matplotlib.pyplot as plt
from bresenham import bresenham
from collections import defaultdict
import home_robot.utils.pose as pu
from torch import IntTensor, Tensor
import home_robot.utils.depth as du
from torch.nn import functional as F
import home_robot.utils.rotation as ru
from typing import Optional, Tuple, List
import home_robot.mapping.map_utils as mu
from home_robot.utils.logger import get_logger
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.utils.spot import draw_circle_segment, fill_convex_hull
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory

# For debugging input and output maps - shows matplotlib visuals
debug_maps = False

logger = get_logger()

def compute_known_cells_map(
    obstacle_map_tensor, robot_pos, max_range, gaze_width, num_beams=360
):
    device = obstacle_map_tensor.device
    H, W = obstacle_map_tensor.shape
    
    # Dilate obstacles
    obstacle_map = (obstacle_map_tensor >= 1).float()
    kernel = torch.ones(1, 1, 3, 3, device=device)
    obstacle_map = torch.nn.functional.conv2d(
        obstacle_map.unsqueeze(0).unsqueeze(0),
        kernel,
        padding=1
    ).squeeze() > 0
    
    cx, cy = robot_pos
    center_angle = 0.0
    half_fov_rad = np.radians(gaze_width / 2.0)
    
    angles = torch.linspace(
        center_angle - half_fov_rad,
        center_angle + half_fov_rad,
        num_beams,
        device=device
    )
    
    dx = torch.cos(angles)
    dy = torch.sin(angles)
    
    max_steps = int(max_range * 1.5)
    t = torch.linspace(0, max_range, max_steps, device=device)
    
    x_coords = cx + dx.unsqueeze(1) * t.unsqueeze(0)
    y_coords = cy + dy.unsqueeze(1) * t.unsqueeze(0)
    
    x_int = torch.round(x_coords).long()
    y_int = torch.round(y_coords).long()
    
    valid_mask = (x_int >= 0) & (x_int < W) & (y_int >= 0) & (y_int < H)
    
    # just to avoid index issues
    x_clamped = torch.clamp(x_int, 0, W - 1)
    y_clamped = torch.clamp(y_int, 0, H - 1)
    
    obstacle_at_point = obstacle_map[x_clamped, y_clamped] | ~valid_mask
    
    cumsum_obstacles = torch.cumsum(obstacle_at_point.float(), dim=1)
    visible_cells = valid_mask & (cumsum_obstacles <= 1)
    
    visible_x = x_int[visible_cells]
    visible_y = y_int[visible_cells]
    
    known_map = torch.zeros((H, W), dtype=torch.float32, device=device)
    known_map[visible_x, visible_y] = 1
    
    return known_map

def get_fp_exp_pred(self, fp_map_pred):
    if self.exploration_type == "raycast":
        fp_exp_pred = compute_known_cells_map(
            fp_map_pred,
            (0, fp_map_pred.shape[1] // 2),
            self.gaze_distance * 100 / self.resolution,
            self.gaze_width,
            num_beams=360,
        )
        return fp_exp_pred
    elif self.exploration_type == "default":
        fp_exp_pred = fp_map_pred.copy()
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
    elif self.exploration_type == "hull":
        fp_exp_pred = fp_map_pred.copy()
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_exp_pred = fp_exp_pred.clip(0, 1)
        # set the current agent position as 1
        fp_exp_pred[0, fp_exp_pred.shape[-1] // 2] = 1

        # fill convex hull
        filled = fill_convex_hull(fp_exp_pred.cpu())
        fp_exp_pred = torch.tensor(filled)

    # uses a fixed cone infront of the camerea
    elif self.exploration_type == "gaze":
        fp_exp_pred = torch.zeros_like(fp_map_pred)
        view_image = torch.zeros(fp_map_pred.shape[-2:])
        # get the desired radius in cells
        dist = self.gaze_distance * 100 / self.resolution
        view_image = draw_circle_segment(
            view_image, (0, fp_exp_pred.shape[-1] // 2), dist, 0, self.gaze_width
        )
        fp_exp_pred = view_image
    # uses depth point projections but limits the fov and distance using the code
    elif self.exploration_type == "gaze_projected":
        fp_exp_pred = fp_map_pred.copy()
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        view_image = torch.zeros(fp_map_pred.shape[-2:])
        # get the desired radius in cells
        dist = self.gaze_distance * 100 / self.resolution
        view_image = (
            draw_circle_segment(
                view_image,
                (0, fp_exp_pred.shape[-1] // 2),
                dist,
                0,
                self.gaze_width,
            )
            / 255
        )
        fp_exp_pred *= view_image.to(fp_exp_pred.device)
    else:
        raise Exception("not implemented")
    return fp_exp_pred


class Categorical2DSemanticMapModule(nn.Module):
    """
    This class is responsible for updating a dense 2D semantic map with one channel
    per object category, the local and global maps and poses, and generating
    map features — it is a stateless PyTorch module with no trainable parameters.

    Map proposed in:
    Object Goal Navigation using Goal-Oriented Semantic Exploration
    https://arxiv.org/pdf/2007.00643.pdf
    https://github.com/devendrachaplot/Object-Goal-Navigation
    """

    # If true, display point cloud visualizations using Open3d
    debug_mode = False

    def __init__(
        self,
        device,
        frame_height: int,
        frame_width: int,
        camera_height: int,
        hfov: int,
        num_sem_categories: int,
        map_size_cm: int,
        map_resolution: int,
        vision_range: int,
        explored_radius: int,
        been_close_to_radius: int,
        global_downscaling: int,
        du_scale: int,
        cat_pred_threshold: float,
        exp_pred_threshold: float,
        map_pred_threshold: float,
        min_depth: float = 0.5,
        max_depth: float = 3.5,
        min_obs_height_cm: int = 25,
        target_blacklisting_radius: int = None,
        record_instance_ids: bool = False,
        instance_memory: Optional[InstanceMemory] = None,
        max_instances: int = 0,
        dilation_for_instances: int = 5,
        padding_for_instance_overlap: int = 5,
        exploration_type="default",
        gaze_width=30,
        gaze_distance=3,
        agent_cell_radius: int = 1,
    ):
        """
        Arguments:
            frame_height: first-person frame height
            frame_width: first-person frame width
            camera_height: camera sensor height (in metres)
            hfov: horizontal field of view (in degrees)
            num_sem_categories: number of semantic segmentation categories
            map_size_cm: global map size (in centimetres)
            map_resolution: size of map bins (in centimeters)
            vision_range: diameter of the circular region of the local map
             that is visible by the agent located in its center (unit is
             the number of local map cells)
            explored_radius: radius (in centimeters) of region of the visual cone
             that will be marked as explored
            been_close_to_radius: radius (in centimeters) of been close to region
            target_blacklisting_radius: radius (in centimeters) of region
             around target that will be blacklisted (if invalid target)
            global_downscaling: ratio of global over local map
            du_scale: frame downscaling before projecting to point cloud
            cat_pred_threshold: number of depth points to be in bin to
             classify it as a certain semantic category
            exp_pred_threshold: number of depth points to be in bin to
             consider it as explored
            map_pred_threshold: number of depth points to be in bin to
             consider it as obstacle
            min_obs_height_cm: minimum height of obstacles (in centimetres)
            record_instance_ids: whether to record instance ids in the 2d semantic map
            exploration_type: how to define explored area
            gaze_width: hfov in degrees for use with the gaze based exploration
            gaze_distance: depth to be considered explored with gaze based exploration
        """
        super().__init__()

        self.device = device
        self.screen_h = frame_height
        self.screen_w = frame_width
        self.hfov = hfov
        aspect_ratio = self.screen_h / self.screen_w
        hfov_rad = np.deg2rad(self.hfov)
        vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) * aspect_ratio)
        self.vfov = np.rad2deg(vfov_rad)

        self.camera_matrix = du.get_camera_matrix(self.screen_w, self.screen_h, hfov)
        self.num_sem_categories = num_sem_categories

        self.map_size_parameters = mu.MapSizeParameters(
            map_resolution, map_size_cm, global_downscaling
        )
        self.resolution = map_resolution
        self.global_map_size_cm = map_size_cm
        self.global_downscaling = global_downscaling
        self.local_map_size_cm = self.global_map_size_cm // self.global_downscaling
        self.global_map_size = self.global_map_size_cm // self.resolution
        self.local_map_size = self.local_map_size_cm // self.resolution
        self.xy_resolution = self.z_resolution = map_resolution
        self.vision_range = vision_range
        self.explored_radius = explored_radius
        self.been_close_to_radius = been_close_to_radius
        if target_blacklisting_radius is not None:
            self.target_blacklisting_radius = target_blacklisting_radius
        self.du_scale = du_scale
        self.cat_pred_threshold = cat_pred_threshold
        self.exp_pred_threshold = exp_pred_threshold
        self.map_pred_threshold = map_pred_threshold

        self.max_depth = max_depth * 100.0
        self.min_depth = min_depth * 100.0
        self.agent_height = camera_height * 100.0
        self.max_voxel_height = int(360 / self.z_resolution)
        self.min_voxel_height = int(-40 / self.z_resolution)
        self.min_obs_height_cm = min_obs_height_cm
        self.min_mapped_height = int(
            self.min_obs_height_cm / self.z_resolution - self.min_voxel_height
        )

        # self.old_x = None
        # self.old_y = None

        self.max_mapped_height = int(
            (self.agent_height + 1) / self.z_resolution - self.min_voxel_height
        )
        self.shift_loc = [self.vision_range * self.xy_resolution // 2, 0, np.pi / 2.0]

        # For cleaning up maps
        self.record_instance_ids = record_instance_ids
        self.padding_for_instance_overlap = padding_for_instance_overlap
        self.dilation_for_instances = dilation_for_instances
        self.instance_memory = instance_memory
        self.max_instances = max_instances
        self.exploration_type = exploration_type
        self.gaze_width = gaze_width
        self.gaze_distance = gaze_distance
        self.agent_cell_radius = agent_cell_radius
        self.vis_dir = None
        self.timestep = 0

        self.avg_pooling_layer = nn.AvgPool2d(self.du_scale)
        self._disk_masks = {}

    @torch.no_grad()
    def forward(
        self,
        obs: Tensor,
        pose_delta: Tensor,
        init_local_map: Tensor,
        init_global_map: Tensor,
        init_local_pose: Tensor,
        init_global_pose: Tensor,
        init_lmb: Tensor,
        init_origins: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, IntTensor, Tensor]:
        """Update maps and poses with a sequence of observations and generate map
        features at each time step.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation)
             of shape (3 + 1 + num_sem_categories,
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (3)
            seq_dones: binary flag that indicate episode restarts
            seq_camera_poses: sequence of (4, 4) extrinsic camera matrices
            init_local_map: initial local map before any updates of shape
             (MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            init_global_map: initial global map before any updates of shape
             (MC.NON_SEM_CHANNELS + num_sem_categories, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (3)
            init_global_pose: initial global pose before any updates of shape
             (3)
            init_lmb: initial local map boundaries of shape (4)
            init_origins: initial local map origins of shape (3)

        Returns:
            seq_map_features: sequence of semantic map features of shape
             (2 * MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            final_local_map: final local map after all updates of shape
             (MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            final_global_map: final global map after all updates of shape
             (MC.NON_SEM_CHANNELS + num_sem_categories, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (3)
            seq_global_pose: sequence of global poses of shape
             (3)
            seq_lmb: sequence of local map boundaries of shape
             (4)
            seq_origins: sequence of local map origins of shape
             (3)
        """
        logger.debug(f"Updating maps and current position")

        global_map, global_pose = init_global_map.clone(), init_global_pose.clone()
        lmb, origins = init_lmb.clone(), init_origins.clone()
        local_map, local_pose = self._update_local_map_and_pose(
            obs,
            pose_delta,
            init_local_map.clone(),
            init_local_pose.clone(),
            origins,
            lmb,
        )
        # updates in place
        self._update_global_map_and_pose(
            local_map, global_map, local_pose, global_pose, lmb, origins
        )
        local_map, local_pose, lmb, origins = mu.get_local_parameters_from_global_pose(
            global_map,
            global_pose,
            self.map_size_parameters,
        )

        #! myTODO: It doesn't look like this is used anywhere, so I'm commenting it out.
        # map_features = self._get_map_features(local_map, global_map)

        logger.debug(f"Updated global pose is: {global_pose}")
        logger.debug(f"Updated local pose is: {local_pose}")
        logger.debug(f"Updated local map boundaries are: {lmb}")
        return (
            # map_features,
            local_map,
            global_map,
            local_pose,
            global_pose,
            lmb,
            origins,
        )

    def _aggregate_instance_map_channels_per_category(
        self, curr_map, num_instance_channels
    ):
        """Aggregate map channels for instances (input: one binary channel per instance in [0, 1])
        by category (output: one channel per category containing instance IDs).
        curr_map is the map created using current observation before merging with global map.
        """

        # Called init, because it is not aggregated per semantic channel. No "id"s yet.
        temp_instance_map = curr_map[
            MC.NON_SEM_CHANNELS
            + self.num_sem_categories : MC.NON_SEM_CHANNELS
            + self.num_sem_categories
            + num_instance_channels,
        ]
        # now we add all instances with the same category to a single channel. Again called temp, because ids are temp_ids and not global ids. Note that 0 means nothing here.
        aggregated_temp_instance_map = torch.zeros(
            self.num_sem_categories,
            curr_map.shape[1],
            curr_map.shape[2],
            device=curr_map.device,
            dtype=curr_map.dtype,
        )

        if num_instance_channels > 0:
            # create category id to instance id list mapping
            category_id_to_temp_id_list = defaultdict(list)
            # retrieve unprocessed instances
            unprocessed_instances = self.instance_memory.unprocessed_views
            # loop over unprocessed instances
            for temp_id, instance in unprocessed_instances.items():
                category_id_to_temp_id_list[instance.category_id].append(temp_id)

            # TODO Can we vectorize this across categories? (Only needed if speed bottleneck)
            for category_id in category_id_to_temp_id_list.keys():
                assert len(category_id_to_temp_id_list[category_id]) != 0
                # get all temp ids for this category
                temp_ids = category_id_to_temp_id_list[category_id]
                # Instance channel 0 corresponds to temp id 1
                instance_map_onehot = temp_instance_map[[i - 1 for i in temp_ids]]
                instance_map_onehot = torch.cat(
                    (
                        1e-5 * torch.ones_like(instance_map_onehot[:1]),
                        instance_map_onehot,
                    ),
                    dim=0,
                )

                # Each entry is either a temp id, or 0 for no instance
                category_instance_map = instance_map_onehot.argmax(dim=0)
                idx_to_temp_id = [0] + temp_ids

                category_instance_map = torch.tensor(
                    idx_to_temp_id, device=category_instance_map.device
                )[category_instance_map]
                # update the per category instance map
                #! 1
                aggregated_temp_instance_map[category_id - 1] = category_instance_map
                # logger.debug(f"Aggregated category {category_id} with temp instance ids {temp_ids}")

        assert not curr_map[
            MC.NON_SEM_CHANNELS + self.num_sem_categories + num_instance_channels :,
        ].any()
        assert (
            curr_map[
                MC.NON_SEM_CHANNELS + self.num_sem_categories + num_instance_channels :,
            ].shape[0]
            == 0
        )

        curr_map = torch.cat(
            (
                curr_map[: MC.NON_SEM_CHANNELS + self.num_sem_categories],
                aggregated_temp_instance_map,
            ),
            dim=0,
        )

        return curr_map

    def draw_line(self, matrix, x1, y1, x2, y2, padding=1):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if x1 < x2:
            sx = 1
        else:
            sx = -1
        if y1 < y2:
            sy = 1
        else:
            sy = -1

        err = dx - dy

        while True:
            for i in range(-padding, padding + 1):
                for j in range(-padding, padding + 1):
                    x = x1 + i
                    y = y1 + j
                    if 0 <= x < matrix.shape[2] and 0 <= y < matrix.shape[1]:
                        matrix[0, y, x] = 1  # Set x-value to 1
                        matrix[1, y, x] = 1  # Set y-value to 1

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def get_stairs(self, voxels, visible_ground):
        """
        Heuristic to mark stair-like regions as obstacles
        based on absence of ground in visible region.
        """
        visible_ground = visible_ground.cpu().numpy()
        #! myTODO: Hardcoded. Fix this later
        visible_ground[80:] = 0

        #! myTODO: x is hardcoded. This means if you don't see anything with z between -x to x (which right now is min_obs_height cm) in a location, this means it is a downward stair.
        x = int(self.min_obs_height_cm / self.z_resolution)
        ground_plane = voxels[
            :, :, -4 * x - self.min_voxel_height : x - self.min_voxel_height
        ]
        ground_plane = ground_plane.sum(axis=2).cpu().numpy()
        ground_plane = np.where(ground_plane >= 1, 1, 0).astype(np.uint8)

        # ground_points = np.column_stack(np.nonzero(ground_plane))
        # points = ground_points[:, [1, 0]].astype(np.float32)  # (x, y)
        # polygon = alpha_shape(points, alpha=0.005)
        # ground_plane = rasterize_polygon(polygon, ground_plane.shape)
        selem = np.ones((7, 7), dtype=np.uint8)
        ground_plane = cv2.dilate(ground_plane, selem, iterations=1)

        X, Y = ground_plane.shape
        robot_x = 0
        robot_y = Y // 2

        xx, yy = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
        dx = (xx - robot_x) * self.xy_resolution
        dy = (yy - robot_y) * self.xy_resolution

        dx[dx == 0] = 1e-6  # avoid division by zero

        # Convert FOVs to radians
        hfov_rad = np.deg2rad(self.hfov)
        vfov_rad = np.deg2rad(self.vfov)

        horizontal_angle = np.arctan2(dy, dx)
        within_hfov = np.abs(horizontal_angle) <= (hfov_rad / 2)

        min_visible_dist = (
            int(self.agent_height / np.tan(vfov_rad / 2) / self.z_resolution) + 10
        )
        within_vfov = np.zeros_like(within_hfov, dtype=bool)
        within_vfov[min_visible_dist:, :] = 1
        # ground_dist = np.sqrt(dx**2 + dy**2)
        # within_vfov = ground_dist >= min_visible_dist

        within_fov = within_hfov & within_vfov

        stair_mask = (ground_plane == 0) & within_fov & visible_ground
        stair_mask_vis = stair_mask.copy()

        #! myTODO: 10 is hardcoded
        # Extend to agents location
        x_indices = np.where(stair_mask[min_visible_dist] == 1)[0]
        rows = np.arange(10, min_visible_dist + 1).reshape(
            -1, 1
        )  # shape: (min_visible_dist-10+1, 1)
        rr, cc = np.meshgrid(rows, x_indices, indexing="ij")
        stair_mask[rr, cc] = 1
        if False:
            import matplotlib

            # matplotlib.use("TkAgg")
            # matplotlib.use("Agg")
            plt.clf()
            plt.subplot(321)
            plt.title("ground plane")
            plt.imshow(np.flipud(ground_plane))
            # plt.subplot(322)
            # plt.title("hfov")
            # plt.imshow(np.flipud(within_hfov))
            # plt.subplot(323)
            # plt.title("vfov")
            # plt.imshow(np.flipud(within_vfov))
            plt.subplot(322)
            plt.title("withinfov")
            plt.imshow(np.flipud(within_fov))
            plt.subplot(323)
            plt.title("visible_ground")
            plt.imshow(np.flipud(visible_ground))
            plt.subplot(324)
            plt.title("stairs_mask")
            plt.imshow(np.flipud(stair_mask_vis))
            plt.subplot(325)
            plt.title("stairs_mask extended")
            plt.imshow(np.flipud(stair_mask))
            plt.savefig(self.vis_dir + f"/{self.timestep}_1.stairs.png")
        return torch.tensor(stair_mask, dtype=torch.uint8).to(
            voxels.device
        ), torch.tensor(ground_plane, dtype=torch.uint8).to(voxels.device)

    def _update_local_map_and_pose(  # noqa: C901
        self,
        obs: Tensor,
        pose_delta: Tensor,
        prev_map: Tensor,
        prev_pose: Tensor,
        origins: Tensor,
        lmb: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Update local map and sensor pose given a new observation using parameter-free
        differentiable projective geometry.

        Args:
            obs: current frame containing (rgb, depth, segmentation) of shape
             (batch_size, 3 + 1 + num_sem_categories, frame_height, frame_width)
            pose_delta: delta in pose since last frame of shape (batch_size, 3)
            prev_map: previous local map of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            prev_pose: previous pose of shape (batch_size, 3)
            camera_pose: current camera poseof shape (batch_size, 4, 4)

        Returns:
            current_map: current local map updated with current observation
             and location of shape (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            current_pose: current pose updated with pose delta of shape (batch_size, 3)
        """
        obs_channels, h, w = obs.size()
        device, dtype = obs.device, obs.dtype
        # if camera_pose is not None: # It is none in our case
        #     # TODO: make consistent between sim and real
        #     # hab_angles = pt.matrix_to_euler_angles(camera_pose[:, :3, :3], convention="YZX")
        #     # angles = pt.matrix_to_euler_angles(camera_pose[:, :3, :3], convention="ZYX")
        #     # angles = torch.Tensor(
        #     #     [tra.euler_from_matrix(p[:3, :3].cpu(), "rzyx") for p in camera_pose]
        #     # )
        #     angles = torch.Tensor(tra.euler_from_matrix(camera_pose[:3, :3], "rzyx"))

        #     # For habitat - pull x angle
        #     # tilt = angles[:, -1]
        #     # For real robot
        #     tilt = angles[1]
        #     # angles gives roll, pitch, yaw
        #     yaw = angles[-1]

        #     # Get the agent pose
        #     # hab_agent_height = camera_pose[:, 1, 3] * 100
        #     agent_pos = camera_pose[:3, 3] * 100
        #     agent_height = agent_pos[2]

        # else:
        yaw = 0
        tilt = torch.zeros(0)
        agent_height = self.agent_height

        yaw = torch.tensor(yaw)
        depth = obs[3, :, :].float()
        depth[depth > self.max_depth] = 0

        # * This point cloud is with respect to cameras location. Is it not converted to world's coords.
        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, device, scale=self.du_scale
        )

        if self.debug_mode:
            # import matplotlib
            # matplotlib.use("TkAgg")
            # import matplotlib.pyplot as plt
            # i = np.random.random((100,100,3))
            # plt.imshow(i)
            # plt.show()
            from home_robot.utils.point_cloud import show_point_cloud

            rgb = obs[:3, :: self.du_scale, :: self.du_scale].permute(1, 2, 0)
            print("Point cloud shape: ", point_cloud_t.shape)
            xyz = point_cloud_t.reshape(-1, 3)
            rgb = rgb.reshape(-1, 3)
            print("-> Showing point cloud in camera coords")
            show_point_cloud(
                (xyz / 100.0).cpu().numpy(),
                (rgb / 255.0).cpu().numpy(),
                orig=np.zeros(3),
            )

        tilt_deg = torch.rad2deg(tilt).item() if tilt.numel() > 0 else 0.0
        point_cloud_base_coords = du.transform_camera_view_t(
            point_cloud_t, agent_height, tilt_deg, device
        )

        # Show the point cloud in base coordinates for debugging
        if self.debug_mode:
            print()
            print("------------------------------")
            print("agent angles =", angles)
            print("agent tilt   =", tilt)
            print("agent height =", agent_height, "preset =", self.agent_height)
            xyz = point_cloud_base_coords.reshape(-1, 3)
            print("-> Showing point cloud in base coords")
            show_point_cloud(
                (xyz / 100.0).cpu().numpy(),
                (rgb / 255.0).cpu().numpy(),
                orig=np.zeros(3),
            )

        point_cloud_map_coords = du.transform_pose_t(
            point_cloud_base_coords, self.shift_loc, device
        )

        if self.debug_mode:
            xyz = point_cloud_base_coords.reshape(-1, 3)
            print("-> Showing point cloud in map coords")
            show_point_cloud(
                (xyz / 100.0).cpu().numpy(),
                (rgb / 255.0).cpu().numpy(),
                orig=np.zeros(3),
            )

        voxel_channels = 1 + self.num_sem_categories
        num_instance_channels = 0
        if self.record_instance_ids:
            num_instance_channels = obs_channels - 4 - self.num_sem_categories
            voxel_channels += num_instance_channels

        assert obs.shape[0] == 4 + self.num_sem_categories + num_instance_channels

        init_grid = torch.zeros(
            voxel_channels,
            self.vision_range,
            self.vision_range,
            self.max_voxel_height - self.min_voxel_height,
            device=device,
            dtype=torch.float32,
        )
        feat = torch.ones(
            voxel_channels,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale,
            device=device,
            dtype=torch.float32,
        )

        semantic_channels = obs[4 : 4 + self.num_sem_categories]

        current_pose = pu.get_new_pose(prev_pose.clone(), pose_delta)

        if self.record_instance_ids:
            instance_channels = obs[4 + self.num_sem_categories :]
            if num_instance_channels > 0:
                self.instance_memory.process_instances(
                    semantic_channels,
                    instance_channels,
                    point_cloud_t.squeeze(0),
                    torch.concat([current_pose + origins, lmb], axis=0),
                    image=obs[:3],
                )

        feat[1:, :] = self.avg_pooling_layer(obs[4:, :, :]).view(
            obs_channels - 4, h // self.du_scale * w // self.du_scale
        )

        XYZ_cm_std = point_cloud_map_coords.float()
        XYZ_cm_std[..., :2] = XYZ_cm_std[..., :2] / self.xy_resolution
        XYZ_cm_std[..., :2] = (
            (XYZ_cm_std[..., :2] - self.vision_range // 2.0) / self.vision_range * 2.0
        )
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / self.z_resolution
        XYZ_cm_std[..., 2] = (
            (
                XYZ_cm_std[..., 2]
                - (self.max_voxel_height + self.min_voxel_height) // 2.0
            )
            / (self.max_voxel_height - self.min_voxel_height)
            * 2.0
        )
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(
            XYZ_cm_std.shape[0],
            XYZ_cm_std.shape[1],
            XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3],
        )

        voxels = du.splat_feat_nd(init_grid, feat, XYZ_cm_std).transpose(1, 2)

        agent_height_proj = voxels[
            ..., self.min_mapped_height : self.max_mapped_height
        ].sum(3)
        all_height_proj = voxels.sum(3)
        # * Shape is: [voxech_channels, height, width]

        fp_map_pred = agent_height_proj[0, :, :]

        # +rows is away from the camera, with the camra origin at row 0
        # +cols is to the right of the image frame, the camera origin is at num_cols/2
        # so the camera origin is at [0,num_cols/2]

        # self.local_map_size_cm
        # plt.imshow(fp_exp_pred[0,0].cpu())
        # plt.pause(0.01)
        fp_exp_pred = get_fp_exp_pred(self, fp_map_pred)

        # NOTE: Only works in fp_exp_pred is 'raycast'
        stairs_map, ground_plane = self.get_stairs(voxels[0], fp_exp_pred >= 1)
        # fp_map_pred += stairs_map

        num_channels = MC.NON_SEM_CHANNELS + self.num_sem_categories
        if self.record_instance_ids:
            num_channels += num_instance_channels

        agent_view = torch.zeros(
            num_channels,
            self.local_map_size_cm // self.xy_resolution,
            self.local_map_size_cm // self.xy_resolution,
            device=device,
            dtype=dtype,
        )

        x1 = self.local_map_size_cm // (self.xy_resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.local_map_size_cm // (self.xy_resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[MC.GROUND_PLANE, y1:y2, x1:x2] = ground_plane * 1.0
        agent_view[MC.STAIRS, y1:y2, x1:x2] = stairs_map
        agent_view[MC.OBSTACLE_MAP, y1:y2, x1:x2] = fp_map_pred
        agent_view[MC.EXPLORED_MAP, y1:y2, x1:x2] = fp_exp_pred

        agent_view[MC.NON_SEM_CHANNELS :, y1:y2, x1:x2] = (
            all_height_proj[1:] / self.cat_pred_threshold
        )

        st_pose = current_pose.clone().detach()
        st_pose[:2] = -(
            (
                st_pose[:2] * 100.0 / self.xy_resolution
                - self.local_map_size_cm // (self.xy_resolution * 2)
            )
            / (self.local_map_size_cm // (self.xy_resolution * 2))
        )
        st_pose[2] = 90.0 - (st_pose[2])

        # st_pose is current pose, last term in degrees
        # account for camera yaw here by rotating the new map based on camera yaw
        st_pose_adjusted = st_pose.clone()
        # yaw has to be inverted here, below rotates the map clockwise
        st_pose_adjusted[2] -= yaw.to(st_pose_adjusted.device) * 180 / np.pi

        rot_mat, trans_mat = ru.get_grid(st_pose_adjusted, agent_view.size(), dtype)
        rotated = F.grid_sample(agent_view.unsqueeze(0), rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)
        # plt.imshow(rotated[0, 0].cpu())

        # Clamp to [0, 1] after transform agent view to map coordinates
        translated = torch.clamp(translated, min=0.0, max=1.0)
        translated = translated.squeeze(0)

        # update instance channels
        if self.record_instance_ids:
            translated = self._aggregate_instance_map_channels_per_category(
                translated, num_instance_channels
            )

        # Remove people from the last map if people are detected
        # TODO Handle people more cleanly
        #! Commented this because channel 11 is not a person in my case
        # if translated[:, MC.NON_SEM_CHANNELS + 11, :, :].sum() > 0.99:
        #     print("Detected a person, removing previous people from the map")
        #     prev_map[:, MC.NON_SEM_CHANNELS + 11, :, :] = 0

        # Aggregate by taking the max of the previous map and current map — this is robust
        # to false negatives in one frame but makes it impossible to remove false positives
        current_map = torch.maximum(prev_map, translated)

        # plt.clf()
        # plt.subplot(221)
        # plt.title("ground plane")
        # plt.imshow(np.flipud((current_map[MC.GROUND_PLANE]>0).cpu()))
        # plt.subplot(222)
        # plt.title("stairs")
        # plt.imshow(np.flipud((current_map[MC.STAIRS]>0).cpu()))
        # plt.subplot(223)
        # plt.title("Obstacle map")
        # plt.imshow(np.flipud((current_map[MC.OBSTACLE_MAP]>0).cpu()))
        # plt.subplot(224)

        # Add stairs to obstacle map
        current_map[MC.OBSTACLE_MAP] = (current_map[MC.OBSTACLE_MAP] > 0) | (
            (current_map[MC.STAIRS] > 0) & (current_map[MC.GROUND_PLANE] == 0.0)
        )

        # plt.title("Final obstacle map")
        # plt.imshow(np.flipud((current_map[MC.OBSTACLE_MAP]>0).cpu()))
        # plt.savefig(self.vis_dir + f"/{self.timestep}_1.stairs2.png")

        # Aggregate by trusting the current map — this is not robust to false negatives in
        # one frame, but it makes it possible to remove false positives
        # TODO Implement this properly for num_environments > 1
        # current_mask = translated[0, 1, :, :] > 0
        # current_map = prev_map.clone()
        # current_map[0, :, current_mask] = translated[0, :, current_mask]

        # Set people as not obstacles for planning
        # TODO Handle people more cleanly
        # TODO Implement this properly for num_environments > 1
        # people_mask = (
        #     skimage.morphology.binary_dilation(
        #         current_map[0, 5 + 11, :, :].cpu().numpy(), skimage.morphology.disk(2)
        #     )
        #     * 1.0
        # )
        # current_map[0, 0, :, :] *= 1 - torch.from_numpy(people_mask).to(device)

        if self.record_instance_ids:
            # overwrite channels containing instance IDs
            current_map[MC.NON_SEM_CHANNELS + self.num_sem_categories :] = translated[
                MC.NON_SEM_CHANNELS + self.num_sem_categories :
            ]

        # Reset current location
        current_map[MC.CURRENT_LOCATION, :, :].fill_(0.0)
        curr_loc = current_pose[:2].flip(0)
        curr_loc = (curr_loc * 100.0 / self.xy_resolution).int()

        prev_loc = prev_pose[:2].flip(0)
        prev_loc = (prev_loc * 100.0 / self.xy_resolution).int()

        y, x = curr_loc
        current_map[
            MC.CURRENT_LOCATION,
            y - 2 : y + 3,
            x - 2 : x + 3,
        ].fill_(1.0)

        current_map[MC.VISITED_MAP] = self._get_update_visited_map(
            curr_loc.tolist(), prev_loc.tolist(), current_map[MC.VISITED_MAP]
        )

        # Set a disk around the agent to explored
        # This is around the current agent - we just sort of assume we know where we are
        self._set_disk_to_one(
            self.explored_radius, current_map, MC.EXPLORED_MAP, curr_loc
        )

        # Record the region the agent has been close to using a disc centered at the agent
        radius = self.been_close_to_radius // self.resolution
        self._set_disk_to_one(radius, current_map, MC.BEEN_CLOSE_MAP, curr_loc)

        # Record the region the agent has been close to using a disc centered at the agent
        radius = self.target_blacklisting_radius // self.resolution
        self._set_disk_to_one(radius, current_map, MC.BLACKLISTED_TARGETS_MAP, curr_loc)

        # debug_maps = True
        if debug_maps:
            import matplotlib

            matplotlib.use("TkAgg")
            current_map = current_map.cpu()
            explored = current_map[0, MC.EXPLORED_MAP].numpy()
            been_close = current_map[0, MC.BEEN_CLOSE_MAP].numpy()
            obstacles = current_map[0, MC.OBSTACLE_MAP].numpy()
            plt.subplot(331)
            plt.axis("off")
            plt.title("explored")
            plt.imshow(explored)
            plt.subplot(332)
            plt.axis("off")
            plt.title("been close")
            plt.imshow(been_close)
            plt.subplot(333)
            plt.axis("off")
            plt.imshow(been_close * explored)
            plt.subplot(334)
            plt.axis("off")
            plt.title("obstacles")
            plt.imshow(obstacles)
            plt.subplot(335)
            plt.axis("off")
            plt.title("obstacles_eroded")

            obs_eroded = cv2.erode(obstacles, np.ones((5, 5)), iterations=5)
            plt.imshow(obs_eroded)
            plt.subplot(336)
            plt.axis("off")
            plt.imshow(been_close * obstacles)
            plt.subplot(337)
            plt.axis("off")
            # rgb = obs[0, :3, :: self.du_scale, :: self.du_scale].permute(1, 2, 0)
            rgb = obs[0, :3].permute(1, 2, 0)
            # print("rgs.shape", rgb.shape)
            plt.imshow(rgb.cpu().numpy().astype(np.uint8))
            plt.subplot(338)
            plt.imshow(depth.cpu().numpy())
            plt.axis("off")
            plt.subplot(339)
            seg = np.zeros_like(depth.cpu().numpy())
            for i in range(4, obs_channels):
                seg += (i - 4) * obs[0, i].cpu().numpy()
            #     print("class =", i, np.sum(obs[0, i].cpu().numpy()), "pts")
            plt.imshow(seg)
            plt.axis("off")
            plt.show()

        return current_map, current_pose

    def _update_global_map_instances_for_one_channel(
        self,
        global_instances: Tensor,
        local_map: Tensor,
        x_range: tuple,
        y_range: tuple,
        max_instance_id: int,
    ) -> Tensor:
        """
        Update one instance channels in the global map from one instance channels in the local map:
        aggregate local instances with existing global instances or create new global instances.

        Args:
            global_instances (Tensor): The global map tensor.
            local_map (Tensor): The local map tensor.
            x_range (tuple): The range of indices in the x-axis for the local map in the global map.
            y_range (tuple): The range of indices in the y-axis for the local map in the global map.

        Returns:
            Tensor: The updated global instances tensor.

        """
        p = self.padding_for_instance_overlap  # default: 1
        d = self.dilation_for_instances  # default: 0

        H = global_instances.shape[0]
        W = global_instances.shape[1]

        x1, x2 = x_range
        y1, y2 = y_range

        # padding added on each side
        t_p = min(x1, p)
        b_p = min(H - x2, p)
        l_p = min(y1, p)
        r_p = min(W - y2, p)

        # the indices of the padded local_map in the global map
        x_start = x1 - t_p
        x_end = x2 + b_p
        y_start = y1 - l_p
        y_end = y2 + r_p

        local_map = torch.round(local_map)

        # pad the local map
        extended_local_map = F.pad(local_map.float(), (l_p, r_p), mode="replicate")
        extended_local_map = F.pad(
            extended_local_map.transpose(1, 0), (t_p, b_p), mode="replicate"
        ).transpose(1, 0)

        self.instance_dilation_selem = skimage.morphology.disk(d)
        # dilate the extended local map
        if d > 0:
            extended_dilated_local_map = torch.round(
                torch.tensor(
                    cv2.dilate(
                        extended_local_map.cpu().numpy(),
                        self.instance_dilation_selem,
                        iterations=1,
                    ),
                    device=local_map.device,
                    dtype=local_map.dtype,
                )
            )
        else:
            extended_dilated_local_map = torch.clone(extended_local_map)
        # Get the instances from the global map within the local map's region

        self._create_or_update_global_instances(
            extended_dilated_local_map,
            global_instances[x_start:x_end, y_start:y_end],
            max_instance_id,
            torch.unique(extended_local_map).tolist(),
        )

        # Update the global map with the associated instances from the local map

        # only to speed up
        max_temp_id = int(max(self.instance_memory.temp_id_to_global_id.keys()))
        temp_id_lookup = np.full(max_temp_id + 1, -1, dtype=np.int16)  # -1 for unmapped
        for temp_id, global_id in self.instance_memory.temp_id_to_global_id.items():
            temp_id_lookup[temp_id] = global_id
        global_instances_in_local = temp_id_lookup[local_map.cpu().numpy().astype(int)]

        global_instances[x1:x2, y1:y2] = torch.maximum(
            global_instances[x1:x2, y1:y2],
            torch.tensor(
                global_instances_in_local,
                dtype=torch.int64,
                device=global_instances.device,
            ),
        )
        return global_instances

    def _create_or_update_global_instances(
        self,
        extended_local_labels: Tensor,
        global_instances_within_local: Tensor,
        max_instance_id: int,
        temp_instance_ids: List[int],
    ) -> dict:
        """
        Creates a global instance for each local instance if it does not already exist, otherwise only update the global instance.
        It also creates a mapping of local instance IDs to global instance IDs internally by calling update_temp_id.

        Args:
            extended_local_labels: Labels of instances in the extended local map.
            global_instances_within_local: Instances from the global map within the local map's region.
        """
        # Associate instances in the local map with corresponding instances in the global map
        for temp_id in temp_instance_ids:
            if temp_id == 0:
                # ignore 0 as it does not correspond to an instance
                continue
            # pixels corresponding to
            local_instance_pixels = extended_local_labels == temp_id

            # Check for overlapping instances in the global map
            overlapping_instances = global_instances_within_local[local_instance_pixels]
            unique_overlapping_instances = torch.unique(overlapping_instances)

            unique_overlapping_instances = unique_overlapping_instances[
                unique_overlapping_instances != 0
            ]
            if len(unique_overlapping_instances) >= 1:
                # If there is a corresponding instance in the global map, pick the first one and associate it
                global_instance_id = int(unique_overlapping_instances[0].item())
            else:
                # If there are no corresponding instances, create a new instance
                max_instance_id += 1
                global_instance_id = max_instance_id
            # update the id in instance memory
            # logger.debug(f"Mapping temp id {temp_id} to global id {global_instance_id}")
            self.instance_memory.update_temp_id(temp_id, global_instance_id)

    def _update_global_map_instances(
        self, global_map: Tensor, local_map: Tensor, lmb: Tensor
    ) -> Tensor:
        """
        Update instance channels in the global map from instance channels in the local map:
        aggregate local instances with existing global instances or create new global instances.

        Args:
            e (int): The index of the environment.
            global_map (Tensor): The global map tensor.
            local_map (Tensor): The local map tensor.
            lmb (Tensor): The tensor containing the ranges of indices for the local map in the global map.

        Returns:
            Used to return global map, but if we are updating it in place, we don't need to return anything.
        """
        # TODO Can we vectorize this across categories? (Only needed if speed bottleneck)
        # logger.debug("Updating global map instances.")
        for i in range(self.num_sem_categories):
            if (
                torch.sum(local_map[MC.NON_SEM_CHANNELS + i + self.num_sem_categories])
                > 0
            ):
                max_instance_id = (
                    torch.max(
                        global_map[
                            MC.NON_SEM_CHANNELS
                            + self.num_sem_categories : MC.NON_SEM_CHANNELS
                            + 2 * self.num_sem_categories,
                        ]
                    )
                    .int()
                    .item()
                )
                # if the local map has any object instances, update the global map with instance ids
                # logger.debug(f"Updating global map instances for category {i}, current max id {max_instance_id}.")
                instance_channel = self._update_global_map_instances_for_one_channel(
                    global_map[MC.NON_SEM_CHANNELS + self.num_sem_categories + i],
                    local_map[MC.NON_SEM_CHANNELS + self.num_sem_categories + i],
                    (lmb[0], lmb[1]),
                    (lmb[2], lmb[3]),
                    max_instance_id,
                )
                global_map[MC.NON_SEM_CHANNELS + self.num_sem_categories + i] = (
                    instance_channel
                )

    def _update_global_map_and_pose(
        self,
        local_map: Tensor,
        global_map: Tensor,
        local_pose: Tensor,
        global_pose: Tensor,
        lmb: Tensor,
        origins: Tensor,
    ):
        """Update global map and pose and re-center local map and pose for a
        particular environment.
        """
        assert global_map.shape[0] == MC.NON_SEM_CHANNELS + self.num_sem_categories * 2

        if self.record_instance_ids:
            self._update_global_map_instances(global_map, local_map, lmb)
            global_map[
                : MC.NON_SEM_CHANNELS + self.num_sem_categories,
                lmb[0] : lmb[1],
                lmb[2] : lmb[3],
            ] = local_map[: MC.NON_SEM_CHANNELS + self.num_sem_categories]
        else:
            global_map[:, lmb[0] : lmb[1], lmb[2] : lmb[3]] = local_map

        local_map[:] = global_map[:, lmb[0] : lmb[1], lmb[2] : lmb[3]]
        global_pose[:] = local_pose + origins

    def merge_neighbor_maps(
        self,
        neighbor_global_map: Tensor,
        global_map: Tensor,
    ):
        # These channels should not be changed with other agents info
        protected_channels = torch.tensor(
            [
                MC.CURRENT_LOCATION,
                MC.VISITED_MAP,
                MC.BEEN_CLOSE_MAP,
                MC.BLACKLISTED_TARGETS_MAP,
            ],
            device=global_map.device,
        )
        all_channels = torch.arange(global_map.shape[0], device=global_map.device)
        #! We should not merge instance map channels too, until we find a way to do it properly
        merge_mask = ~torch.isin(all_channels, protected_channels) & (
            all_channels < (MC.NON_SEM_CHANNELS + self.num_sem_categories)
        )

        temp_copy = global_map.clone()
        global_map[merge_mask] = torch.maximum(
            global_map[merge_mask],
            neighbor_global_map[merge_mask],
        )
        assert torch.equal(
            temp_copy[MC.NON_SEM_CHANNELS + self.num_sem_categories :],
            global_map[MC.NON_SEM_CHANNELS + self.num_sem_categories :],
        )

    def _get_map_features(self, local_map: Tensor, global_map: Tensor) -> Tensor:
        """Get global and local map features.

        Arguments:
            local_map: local map of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            global_map: global map of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M * ds, M * ds)

        Returns:
            map_features: semantic map features of shape
             (batch_size, 2 * MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
        """
        map_features_channels = 2 * MC.NON_SEM_CHANNELS + self.num_sem_categories

        if self.record_instance_ids:
            map_features_channels += self.num_sem_categories

        map_features = torch.zeros(
            map_features_channels,
            self.local_map_size,
            self.local_map_size,
            device=local_map.device,
            dtype=local_map.dtype,
        )

        # Local obstacles, explored area, and current and past position
        map_features[0 : MC.NON_SEM_CHANNELS, :, :] = local_map[
            0 : MC.NON_SEM_CHANNELS, :, :
        ]
        # Global obstacles, explored area, and current and past position
        map_features[MC.NON_SEM_CHANNELS : 2 * MC.NON_SEM_CHANNELS, :, :] = (
            nn.MaxPool2d(self.global_downscaling)(
                global_map[0 : MC.NON_SEM_CHANNELS, :, :]
            )
        )
        # Local semantic categories
        map_features[2 * MC.NON_SEM_CHANNELS :, :, :] = local_map[
            MC.NON_SEM_CHANNELS :, :, :
        ]

        if debug_maps:
            plt.subplot(131)
            plt.imshow(local_map[0, 7])  # second object = cup
            plt.subplot(132)
            plt.imshow(local_map[0, 6])  # first object = chair
            # This is the channel in MAP FEATURES mode
            plt.subplot(133)
            plt.imshow(map_features[0, 12])
            plt.show()

        return map_features.detach()

    def _get_disk_mask(self, radius):
        """Cache disk masks for reuse"""
        if radius not in self._disk_masks:
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = x**2 + y**2 <= radius**2
            self._disk_masks[radius] = torch.from_numpy(mask).to(self.device)
        return self._disk_masks[radius]

    def _set_disk_to_one(self, radius, current_map, channel, curr_loc):
        y, x = curr_loc
        disk_mask = self._get_disk_mask(radius)

        H, W = current_map.shape[1:]
        y_min = max(y - radius, 0)
        y_max = min(y + radius + 1, H)
        x_min = max(x - radius, 0)
        x_max = min(x + radius + 1, W)

        disk_y_min = y_min - (y - radius)
        disk_y_max = disk_y_min + (y_max - y_min)
        disk_x_min = x_min - (x - radius)
        disk_x_max = disk_x_min + (x_max - x_min)

        current_map[channel, y_min:y_max, x_min:x_max][
            disk_mask[disk_y_min:disk_y_max, disk_x_min:disk_x_max] == 1
        ] = 1

    def _get_update_visited_map(self, current_loc, prev_loc, visited_map):
        """
        Marks the visited_map with a thick line between start and end.
        """
        thickness = self.agent_cell_radius + 1
        line_points = list(
            bresenham(prev_loc[0], prev_loc[1], current_loc[0], current_loc[1])
        )

        for x, y in line_points:
            if 0 <= x < visited_map.shape[0] and 0 <= y < visited_map.shape[1]:
                rr, cc = disk((x, y), radius=thickness, shape=visited_map.shape)
                visited_map[rr, cc] = 1
        visited_map[
            current_loc[0]
            - self.agent_cell_radius : current_loc[0]
            + self.agent_cell_radius
            + 1,
            current_loc[1]
            - self.agent_cell_radius : current_loc[1]
            + self.agent_cell_radius
            + 1,
        ] = 1
        return visited_map
