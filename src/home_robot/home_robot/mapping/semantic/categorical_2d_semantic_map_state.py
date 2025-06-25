# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import numpy as np
import torch

from home_robot.mapping.map_utils import MapSizeParameters, init_map_and_pose
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory


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

    def get_obstacle_map(self) -> np.ndarray:
        """Get local obstacle map for an environment."""
        return np.copy(self.local_map[MC.OBSTACLE_MAP, :, :].cpu().float().numpy())

    def get_explored_map(self) -> np.ndarray:
        """Get local explored map for an environment."""
        return np.copy(self.local_map[MC.EXPLORED_MAP, :, :].cpu().float().numpy())

    def get_visited_map(self) -> np.ndarray:
        """Get local visited map for an environment."""
        return np.copy(self.local_map[MC.VISITED_MAP, :, :].cpu().float().numpy())

    def get_been_close_map(self) -> np.ndarray:
        """Get map showing regions the agent has been close to"""
        return np.copy(self.local_map[MC.BEEN_CLOSE_MAP, :, :].cpu().float().numpy())

    def get_blacklisted_targets_map(self) -> np.ndarray:
        """Get map showing regions the agent has been close to"""
        return np.copy(
            self.local_map[MC.BLACKLISTED_TARGETS_MAP, :, :].cpu().float().numpy()
        )

    def get_semantic_map(self) -> np.ndarray:
        """Get local map of semantic categories for an environment."""
        semantic_map = np.copy(self.local_map.cpu().float().numpy())
        semantic_map[
            MC.NON_SEM_CHANNELS + self.num_sem_categories - 1, :, :
        ] = 1e-5  # Last category is unlabeled
        semantic_map = semantic_map[
            MC.NON_SEM_CHANNELS : MC.NON_SEM_CHANNELS + self.num_sem_categories, :, :
        ].argmax(0)
        return semantic_map

    def get_instance_map(self) -> np.ndarray:
        instance_map = self.local_map.cpu().float().numpy()
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


    @property
    def local_loc(self):
        """local_loc is the index in local map as it is (don't need to flip)"""
        location = self.local_pose[:2]
        location = (location * 100.0 / self.resolution).int()
        return location[1], location[0]