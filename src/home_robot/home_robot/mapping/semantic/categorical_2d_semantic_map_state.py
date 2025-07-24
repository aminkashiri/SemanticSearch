# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import os
import torch
import numpy as np
from typing import Optional
import torch.nn.functional as F
from scipy.ndimage import label
from home_robot.utils.logger import get_logger
from home_robot.mapping.map_utils import MapSizeParameters, init_map_and_pose
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory


logger = get_logger()


class Categorical2DSemanticMapState:
    """
    This class holds a dense 2D semantic map with one channel per object
    category, the global and local map and sensor pose, as well as the agent's
    current goal in the local map.

    Map proposed in:
    Object Goal Navigation using Goal-Oriented Semantic Exploration
    https://arxiv.org/pdf/2007.00643.pdf
    https://github.com/devendrachaplot/Object-Goal-Navigation
    """

    def __init__(
        self,
        device: torch.device,
        num_sem_categories: int,
        map_resolution: int,
        map_size_cm: int,
        global_downscaling: int,
        record_instance_ids: bool = False,
        evaluate_instance_tracking: bool = False,
        instance_memory: Optional[InstanceMemory] = None,
        max_instances: int = 0,
        close_frontier_radius: int = 1,
    ):
        """
        Arguments:
            device: torch device on which to store map state
            num_environments: number of parallel maps (always 1 in real-world but
             multiple in simulation)
            num_sem_categories: number of semantic channels in the map
            map_resolution: size of map bins (in centimeters)
            map_size_cm: global map size (in centimetres)
            global_downscaling: ratio of global over local map
            record_instance_ids: whether to predict and store instance ids in the map
        """
        self.device = device
        self.num_sem_categories = num_sem_categories

        self.map_size_parameters = MapSizeParameters(
            map_resolution, map_size_cm, global_downscaling
        )
        self.resolution = map_resolution
        self.global_map_size_cm = map_size_cm
        self.global_downscaling = global_downscaling
        self.local_map_size_cm = self.global_map_size_cm // self.global_downscaling
        self.global_map_size = self.global_map_size_cm // self.resolution
        self.local_map_size = self.local_map_size_cm // self.resolution

        # Map consists of multiple channels (5 NON_SEM_CHANNELS followed by semantic channels) containing the following:
        # 0: Obstacle Map
        # 1: Explored Area
        # 2: Current Agent Location
        # 3: Past Agent Locations
        # 4: Regions agent has been close to
        # 5, 6, 7, .., num_sem_categories + 5: Semantic Categories
        num_channels = self.num_sem_categories + MC.NON_SEM_CHANNELS
        if record_instance_ids:
            # num_sem_categories + 5, ..., 2 * num_sem_categories + 5: Instance ids per semantic category
            num_channels += self.num_sem_categories
            self.instance_memory = instance_memory

        if evaluate_instance_tracking:
            num_channels += max_instances + 1
        
        self.num_channels = num_channels
        self.vis_dir = None
        self.close_frontier_radius = close_frontier_radius

    def init_map_and_pose(self):
        """Initialize global and local map and sensor pose variables."""
        self.global_map, self.global_pose, self.local_map, self.local_pose, self.lmb, self.origins = init_map_and_pose(
            self.map_size_parameters,
            self.device,
            self.num_channels,
        )
        self.frontier_map = np.zeros(
            (self.local_map_size, self.local_map_size)
        )

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_obstacle_map(self, local=True) -> np.ndarray:
        """Get local obstacle map for an environment."""
        if local:
            return (np.copy(self.local_map[MC.OBSTACLE_MAP, :, :].cpu().numpy()) > 0).astype(np.uint8)
        else:
            return (np.copy(self.global_map[MC.OBSTACLE_MAP, :, :].cpu().numpy()) > 0).astype(np.uint8)

    def get_explored_map(self, local=True) -> np.ndarray:
        """Get local explored map for an environment."""
        if local:
            return (np.copy(self.local_map[MC.EXPLORED_MAP, :, :].cpu().numpy()) > 0).astype(np.uint8)
        else:
            return (np.copy(self.global_map[MC.EXPLORED_MAP, :, :].cpu().numpy()) > 0).astype(np.uint8)

    def get_visited_map(self, local=True) -> np.ndarray:
        """Get local visited map for an environment."""
        if local:
            return (np.copy(self.local_map[MC.VISITED_MAP, :, :].cpu().numpy()) > 0).astype(np.uint8)
        else:
            return (np.copy(self.global_map[MC.VISITED_MAP, :, :].cpu().numpy()) > 0).astype(np.uint8)

    def get_been_close_map(self, local=True) -> np.ndarray:
        """Get map showing regions the agent has been close to"""
        if local:
            return (np.copy(self.local_map[MC.BEEN_CLOSE_MAP, :, :].cpu().numpy()) > 0).astype(np.uint8)
        else:
            return (np.copy(self.global_map[MC.BEEN_CLOSE_MAP, :, :].cpu().numpy()) > 0).astype(np.uint8)

    def get_blacklisted_targets_map(self, local=True) -> np.ndarray:
        """Get map showing regions the agent has been close to"""
        if local:
            return (np.copy(
                self.local_map[MC.BLACKLISTED_TARGETS_MAP, :, :].cpu().numpy()
            ) > 0).astype(np.uint8)
        else:
            return (np.copy(
                self.global_map[MC.BLACKLISTED_TARGETS_MAP, :, :].cpu().numpy()
            ) > 0).astype(np.uint8)

    def get_semantic_map(self, local=True, full=False) -> np.ndarray:
        """Get local map of semantic categories for an environment."""
        if local:
            map = self.local_map
        else:
            map = self.global_map

        semantic_map = np.copy(map.cpu().float().numpy())[MC.NON_SEM_CHANNELS : MC.NON_SEM_CHANNELS + self.num_sem_categories]
        if not full:
            semantic_map[
                self.num_sem_categories - 1, :, :
            ] = 1e-5  # Last category is unlabeled
            semantic_map = semantic_map.argmax(0)
        return semantic_map

    def get_instances_map(self, local=True) -> np.ndarray:
        if local:
            map = self.local_map
        else:
            map = self.global_map
        instance_map = map.cpu().float().numpy()
        instance_map = instance_map[
            MC.NON_SEM_CHANNELS
            + self.num_sem_categories : MC.NON_SEM_CHANNELS
            + 2 * self.num_sem_categories,
            :,
            :,
        ]
        return instance_map


    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def local_to_global(self, row_local, col_local):
        lmb = self.lmb.cpu()
        row_global = row_local + lmb[0] - self.global_map_size // 2
        col_global = col_local + lmb[2] - self.global_map_size // 2
        return row_global, col_global

    def global_to_local(self, row_global, col_global):
        lmb = self.lmb.cpu()
        row_local = row_global - lmb[0] + self.global_map_size // 2
        col_local = col_global - lmb[2] + self.global_map_size // 2
        return row_local, col_local
    
    def global_pose_to_global_location(self, global_pose):
        global_location = [int(global_pose[1] * 100.0 / self.resolution), int(global_pose[0] * 100.0 /self.resolution)]
        return global_location
    
    def global_location_to_local_location(self, global_location):
        local_location = [global_location[0] - self.lmb[0].item(), global_location[1] - self.lmb[2].item()]
        return local_location

    def global_pose_to_local_location(self, global_pose):
        return self.global_location_to_local_location(self.global_pose_to_global_location(global_pose))
    
    def is_location_in_local_map(self, local_location):
        return 0 <= local_location[0] < self.local_map_size and 0 <= local_location[1] < self.local_map_size

    @property
    def local_loc(self):
        """local_loc is the index in local map as it is (don't need to flip)"""
        location = self.local_pose[:2]
        location = (location * 100.0 / self.resolution).int().tolist()
        return location[1], location[0]

    @property
    def global_loc(self):
        location = self.global_pose[:2]
        location = (location * 100.0 / self.resolution).int().tolist()
        return location[1], location[0]
    
    def get_frontier_map(self, local=True, timestep=None):
        """
        Detect frontiers: free cells adjacent to unknown areas.

        Args:
            known_map (np.ndarray): binary 2D map (1=known, 0=unknown)
            obstacle_map (np.ndarray): binary 2D map (1=obstacle, 0=non-obstacle)

        Returns:
            frontier_map (np.ndarray): binary map (1=frontier, 0=non-frontier)
        """

        def remove_small_frontiers(frontier_map, min_size=10):
            labeled_map, num_features = label(frontier_map)
            cleaned_map = np.zeros_like(frontier_map)
            for region_id in range(1, num_features + 1):
                region = labeled_map == region_id
                if np.sum(region) >= min_size:
                    cleaned_map[region] = 1
            return cleaned_map

        known_map = self.get_explored_map(local)
        obstacle_map = self.get_obstacle_map(local)

        free_space = (known_map == 1) & (obstacle_map == 0)

        unknown = torch.tensor(known_map == 0, dtype=torch.float32)

        kernel = torch.tensor(
                [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)

        unknown_neighbors = F.conv2d(unknown.unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze(0).squeeze(0).numpy()
        frontier_map = free_space & (unknown_neighbors > 0)
        frontier_map2 = remove_small_frontiers(frontier_map, min_size=10)
        frontier_map3 = self.remove_close_frontiers(frontier_map2)
        frontier_map4 = frontier_map3 & (1-self.get_unreachable_frontiers_map(local))
        self.print_maps(
            frontier_map=frontier_map,
            frontier_map2=frontier_map2,
            frontier_map3=frontier_map3,
            frontier_map4=frontier_map4,
            obstacle_map=obstacle_map,
            known_map=known_map,
            timestep=timestep,
            local=local,
            # robot_pos=(50, 50),  # optional
        )
        return frontier_map4

    def remove_close_frontiers(self, frontier_map: np.ndarray) -> np.ndarray:
        """
        Remove frontiers closer than 'radius' to 'location' from the frontier map.

        Args:
            frontier_map (torch.Tensor): shape [ H, W] binary map of frontiers
            radius (float): distance threshold (in pixels)

        Returns:
            torch.Tensor: updated frontier_map with close frontiers removed
        """
        logger.debug("Removing close frontiers from the frontier map.")
        H, W = frontier_map.shape
        y_coords = np.arange(H).reshape(-1, 1).repeat(W, axis=1)
        x_coords = np.arange(W).reshape(1, -1).repeat(H, axis=0)

        dist = np.sqrt((x_coords - self.local_loc[1]) ** 2 + (y_coords - self.local_loc[0]) ** 2)

        close_mask = dist <= self.close_frontier_radius
        new_frontier_map = frontier_map.copy()
        new_frontier_map[close_mask] = 0
        if not np.any(new_frontier_map == 1):
            logger.warning("No frontiers left after removing close frontiers, returning original frontier map.")
            return frontier_map

        return new_frontier_map

    def print_maps(
        self, frontier_map, obstacle_map, known_map, local, timestep=None, frontier_map2=None, frontier_map3=None, frontier_map4=None
    ):
        """
        Visualizes the frontier map alongside obstacles and explored area.

        Args:
            frontier_map (torch.Tensor or np.ndarray): binary map of frontiers (1 = frontier)
            obstacle_map (torch.Tensor or np.ndarray): binary map of obstacles (1 = obstacle)
            known_map (torch.Tensor or np.ndarray): binary map of known/explored cells (1 = known)
            robot_pos (tuple or None): (x, y) position to plot (optional)
        """
        if timestep is None:
            return

        #! myTODO: Can use visualize_map function from home_robot.visualization.visualize_map
        H, W = known_map.shape
        vis_map = np.ones((H, W, 3), dtype=np.uint8) * 255
        vis_map[known_map == 1] = [117, 117, 117]
        vis_map[obstacle_map == 1] = [0, 0, 0]

        vis_map = np.concatenate([vis_map]*4, axis=1)

        vis_map[:,:W,:][frontier_map == 1] = [255, 0, 0]
        vis_map[:,W:2*W,:][frontier_map2 == 1] = [255, 0, 0]
        vis_map[:,2*W:3*W,:][frontier_map3 == 1] = [255, 0, 0]
        vis_map[:,3*W:4*W,:][frontier_map4 == 1] = [255, 0, 0]

        cv2.imwrite(
            os.path.join(self.vis_dir, f"{timestep}_2.frontiers{'' if local else '_global'}.png"),
            np.flipud(vis_map)
        )
    def get_unreachable_frontiers_map(self, local=True) -> np.ndarray:
        if local:
            return (np.copy(
                self.local_map[MC.UNREACHABLE_FRONTIERS_MAP, :, :].cpu().numpy() > 0).astype(np.uint8)
            )
        else:
            return (np.copy(
                self.global_map[MC.UNREACHABLE_FRONTIERS_MAP, :, :].cpu().numpy() > 0).astype(np.uint8)
            )


    def set_unreachable_frontier(self, frontier_map, local=True) -> np.ndarray:
        """Get map showing regions the agent has been close to"""
        if local:
            unreachable_frontiers_map = self.local_map[MC.UNREACHABLE_FRONTIERS_MAP]
            new_unreachable_frontiers_map = torch.logical_or(torch.tensor(frontier_map, device=self.device), unreachable_frontiers_map)
            self.local_map[MC.UNREACHABLE_FRONTIERS_MAP] = new_unreachable_frontiers_map
            self.global_map[MC.UNREACHABLE_FRONTIERS_MAP, self.lmb[0]:self.lmb[1], self.lmb[2]:self.lmb[3]] = new_unreachable_frontiers_map 
        else:
            unreachable_frontiers_map = self.global_map[MC.UNREACHABLE_FRONTIERS_MAP]
            new_unreachable_frontiers_map = torch.logical_or(torch.tensor(frontier_map, device=self.device), unreachable_frontiers_map)
            self.global_map[MC.UNREACHABLE_FRONTIERS_MAP] = torch.logical_or(torch.tensor(frontier_map, device=self.device), unreachable_frontiers_map)
            self.local_map[MC.UNREACHABLE_FRONTIERS_MAP] = self.global_map[MC.UNREACHABLE_FRONTIERS_MAP, self.lmb[0]:self.lmb[1], self.lmb[2]:self.lmb[3]]
            