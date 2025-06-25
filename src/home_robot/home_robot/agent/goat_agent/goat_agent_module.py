import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory

from .goat_matching import GoatMatching

from home_robot.utils.logger import get_logger
logger = get_logger()


class GoatAgentModule(nn.Module):
    def __init__(
        self,
        config,
        matching: GoatMatching,
        instance_memory: Optional[InstanceMemory] = None,
    ):
        super().__init__()
        self.matching = matching
        self.instance_memory = instance_memory
        self.goal_inst = None
        self.instance_goal_found = False
        self.num_sem_categories = config.AGENT.SEMANTIC_MAP.num_sem_categories
        self.semantic_map_module = Categorical2DSemanticMapModule(
            frame_height=config.ENVIRONMENT.frame_height,
            frame_width=config.ENVIRONMENT.frame_width,
            camera_height=config.ENVIRONMENT.camera_height,
            hfov=config.ENVIRONMENT.hfov,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            max_depth=config.AGENT.SEMANTIC_MAP.max_depth,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            vision_range=config.AGENT.SEMANTIC_MAP.vision_range,
            explored_radius=config.AGENT.SEMANTIC_MAP.explored_radius,
            been_close_to_radius=config.AGENT.SEMANTIC_MAP.been_close_to_radius,
            target_blacklisting_radius=config.AGENT.SEMANTIC_MAP.target_blacklisting_radius,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            du_scale=config.AGENT.SEMANTIC_MAP.du_scale,
            cat_pred_threshold=config.AGENT.SEMANTIC_MAP.cat_pred_threshold,
            exp_pred_threshold=config.AGENT.SEMANTIC_MAP.exp_pred_threshold,
            map_pred_threshold=config.AGENT.SEMANTIC_MAP.map_pred_threshold,
            must_explore_close=config.AGENT.SEMANTIC_MAP.must_explore_close,
            min_obs_height_cm=config.AGENT.SEMANTIC_MAP.min_obs_height_cm,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            instance_memory=instance_memory,
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            evaluate_instance_tracking=getattr(
                config.ENVIRONMENT, "evaluate_instance_tracking", False
            ),
            exploration_type=config.AGENT.SEMANTIC_MAP.exploration_type,
            gaze_width=(
                40
                if config.AGENT.SEMANTIC_MAP.exploration_type == "raycast"
                else 30
            ), #! myTODO: Hardcoded 3
            gaze_distance=(
                config.AGENT.SEMANTIC_MAP.max_depth
                if config.AGENT.SEMANTIC_MAP.exploration_type == "raycast"
                else 3
            ), #! myTODO: Hardcoded 3
        )
        self.goal_map = None
        self.goal_pose = None

    def reset_sub_episode(self):
        self.instance_goal_found = False
        self.goal_inst = None
        self.goal_map = None
        self.goal_pose = None

    def reset(self):
        self.reset_sub_episode()

    def forward(
        self,
        obs,
        pose_delta,
        init_local_map,
        init_global_map,
        init_local_pose,
        init_global_pose,
        init_lmb,
        init_origins,
        camera_pose=None,
        reject_visited_targets=False,
        blacklist_target=False,
        confidence=None,
        frame_matches_local_instance_ids=None,
        all_confidences=None,
        instance_ids=None,
        score_thresh=0.0,
        obstacle_locations=None,
        free_locations=None,
    ):
        """Update maps and poses with a sequence of observations, and predict
        high-level goals from map features.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation, instance_segmentation)
             of shape (batch_size, sequence_length, 3 + 1 + num_sem_categories + num_instances,
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_dones: sequence of (batch_size, sequence_length) done flags that
             indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            seq_camera_poses: sequence of (batch_size, 4, 4) camera poses
            init_local_map: initial local map before any updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            init_global_map: initial global map before any updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (batch_size, 3)
            init_global_pose: initial global pose before any updates of shape
             (batch_size, 3)
            init_lmb: initial local map boundaries of shape (batch_size, 4)
            init_origins: initial local map origins of shape (batch_size, 3)
        Returns:
            seq_goal_map: sequence of binary maps encoding goal(s) of shape
             (batch_size, sequence_length, M, M)
            final_local_map: final local map after all updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            final_global_map: final global map after all updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (batch_size, sequence_length, 3)
            seq_global_pose: sequence of global poses of shape
             (batch_size, sequence_length, 3)
            seq_lmb: sequence of local map boundaries of shape
             (batch_size, sequence_length, 4)
            seq_origins: sequence of local map origins of shape
             (batch_size, sequence_length, 3)
        """
        last_step_local_id_to_global_id_map = (
            self.instance_memory.local_id_to_global_id_map.copy()
        )
        # Update map with observations and generate map features
        (
            final_local_map,
            final_global_map,
            local_pose,
            global_pose,
            lmb,
            origins,
        ) = self.semantic_map_module(
            obs,
            pose_delta,
            camera_pose,
            init_local_map,
            init_global_map,
            init_local_pose,
            init_global_pose,
            init_lmb,
            init_origins,
            obstacle_locations=obstacle_locations,
            free_locations=free_locations,
            blacklist_target=blacklist_target,
        )
 

        instance_map = final_local_map[
            MC.NON_SEM_CHANNELS + self.num_sem_categories : MC.NON_SEM_CHANNELS + 2 * self.num_sem_categories,
            :,
            :,
        ]
        if self.instance_goal_found:
            logger.info(f"Already found instance goal, not searching anymore.")
            self.goal_map, self.goal_pose = self.matching.get_goal_map_from_goal_instance(
                instance_map, lmb, self.goal_inst
            )
        else:
            logger.info(f"Searching for instance goal.")
            logger.debug(f"candidate matches in memory: {len(all_confidences)}, candidate matches in observation: {confidence is not None}")
            if len(all_confidences) > 0 or confidence is not None:
                (
                    self.goal_map,
                    self.goal_pose,
                    self.instance_goal_found,
                    self.goal_inst,
                ) = self.matching.select_and_localize_instance(
                    instance_map,
                    lmb,
                    confidence,
                    frame_matches_local_instance_ids,
                    last_step_local_id_to_global_id_map,
                    all_confidences=all_confidences,
                    instance_ids=instance_ids,
                    score_thresh=score_thresh,
                )
            else:
                logger.info(f"No candidate matches found in memory or observation.")
        
        return (
            self.goal_map,
            self.goal_pose,
            final_local_map,
            final_global_map,
            local_pose,
            global_pose,
            lmb,
            origins,
        )
