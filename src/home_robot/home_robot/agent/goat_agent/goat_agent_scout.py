#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Scout GOAT Agent - Real-world multi-task navigation agent
Standalone version (no inheritance) with GoatAgent syntax/patterns
"""

import os
import json
import torch
import psutil
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import home_robot.utils.pose as pu
from home_robot.utils.logger import get_logger
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory
from home_robot.navigation_planner.fixed_discrete_planner import DiscretePlanner
from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)

# Import GoatMatching (same as GoatAgent)
from home_robot.agent.goat_agent.goat_matching import GoatMatching

logger = get_logger()


@dataclass
class Task:
    """Task dataclass matching GoatAgent's Task class."""
    type: str
    goal_semantic_id: int
    goal_image: np.ndarray = None
    goal_image_processed: np.ndarray = None
    goal_image_keypoints: np.ndarray = None
    goal_description: str = None


class ScoutGoatAgent(Agent):
    """
    Real-world GOAT agent for Scout robot.
    Standalone version with GoatAgent's syntax and patterns.
    
    Key differences from GoatAgent:
    - Uses config.ENVIRONMENT for camera params (not config.habitat.simulator)
    - No built-in perception (semantics provided externally)
    - Simplified action return format
    - Additional goal proximity checking for real-world stopping
    """
    
    def __init__(self, config, vocabulary, agent_id=None, device_id: int = 0):
        """
        Initialize ScoutGoatAgent.
        
        Args:
            config: Configuration object with ENVIRONMENT, AGENT sections
            vocabulary: List of semantic category names
            agent_id: Optional agent identifier for multi-agent scenarios
            device_id: CUDA device ID
        """
        # Agent identification (from GoatAgent)
        if agent_id is None:
            self.is_multiagent = False
        else:
            self.is_multiagent = True
        
        self.agent_id = agent_id
        self.log = get_logger(agent_id=agent_id)
        self.verbose = getattr(config, 'VERBOSE', False)
        
        if self.verbose:
            print(f"\n{'='*60}\n[INIT] ScoutGoatAgent\n{'='*60}")
        
        # Configuration (from GoatAgent)
        self.config = config
        self.max_steps = config.AGENT.max_steps
        self.task_type = getattr(config, 'TASK_TYPE', 'Goat-v1')  # Scout default
        self.seq_goals = bool(getattr(config, 'SEQ', True))  # Scout uses sequential goals
        self.use_yolo = False  # Scout doesn't use built-in YOLO
        self.record_instance_ids = True  # Required for instance tracking
        self.visualization_level = getattr(config, 'VISUALIZATION_LEVEL', 0)
        self.max_subtasks_per_episode = getattr(
            config.ENVIRONMENT, 'max_subtasks_per_episode',
            getattr(config.ENVIRONMENT, 'max_num_sub_task_episodes', 10)
        )
        
        # Device setup (from GoatAgent)
        if config.NO_GPU:
            self.device = torch.device("cpu")
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")
        
        self.num_sem_categories = len(vocabulary)
        
        if self.verbose:
            print(f"[INIT] Device: {self.device}, Categories: {self.num_sem_categories}")
        
        # Instance memory setup (from GoatAgent)
        self.instance_memory = None
        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                config=config,
                mask_cropped_instances=False,
                padding_cropped_instances=200,
            )
        
        # Goal matching setup (from GoatAgent)
        self.matching = GoatMatching(
            device=device_id if not config.NO_GPU else 0,
            config=config.AGENT.SUPERGLUE,
            default_vis_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
            print_images=self.visualization_level > 1,
            instance_memory=self.instance_memory,
            logger=self.log,
            cat_match_threshold=getattr(config.AGENT, 'cat_match_threshold', 0.5),
        )
        
        # Semantic map module (CHANGED: uses config.ENVIRONMENT)
        agent_cell_radius = int(
            np.ceil(config.AGENT.radius * 100.0 / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        
        self.semantic_map_module = Categorical2DSemanticMapModule(
            device=self.device,
            frame_height=config.ENVIRONMENT.frame_height,
            frame_width=config.ENVIRONMENT.frame_width,
            camera_height=config.ENVIRONMENT.camera_height,
            hfov=config.ENVIRONMENT.hfov,
            num_sem_categories=self.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            max_depth=config.ENVIRONMENT.max_depth,
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
            min_obs_height_cm=config.AGENT.SEMANTIC_MAP.min_obs_height_cm,
            record_instance_ids=self.record_instance_ids,
            instance_memory=self.instance_memory,
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            exploration_type=config.AGENT.exploration_type,
            gaze_width=40 if config.AGENT.exploration_type == "raycast" else 30,
            gaze_distance=(
                config.ENVIRONMENT.max_depth
                if config.AGENT.exploration_type == "raycast"
                else 3
            ),
            agent_cell_radius=agent_cell_radius,
            print_images=self.visualization_level > 2
        )
        
        self.inst_goal_id = None
        
        # Semantic map state (from GoatAgent)
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_sem_categories=self.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            record_instance_ids=self.record_instance_ids,
            instance_memory=self.instance_memory,
            visualization_level=self.visualization_level,
            agent_id=agent_id,
        )
        
        # Panorama setup (CHANGED: uses config.ENVIRONMENT.turn_angle)
        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0
        
        if self.verbose:
            print(f"[INIT] Panorama: {self.panorama_start_steps} steps")
        
        # Planner setup (CHANGED: uses config.ENVIRONMENT.turn_angle)
        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualization_level=self.visualization_level,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=config.AGENT.PLANNER.discrete_actions,
            min_goal_distance_cm=config.AGENT.PLANNER.min_goal_distance_cm,
            panorama_start_steps=self.panorama_start_steps,
            instance_memory=self.instance_memory,
            goal_filtering=config.AGENT.SEMANTIC_MAP.goal_filtering,
            semantic_map=self.semantic_map,
            frontier_metric=config.AGENT.frontier_metric,
            agent_id=self.agent_id,
            ground_truth_semantics=False,  # Always false for real world
            task_type=self.task_type
        )
        
        # Episode state (from GoatAgent)
        self.subtask_timesteps = None
        self.sub_task_timesteps = None  # Alias for backwards compatibility
        self.total_timesteps = None
        self.last_pose = None
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.current_task_idx = 0
        self.navigate_to_best = False
        self.stuck_counter = 0
        self.match_memory = True
        
        # CHANGED: Always False for real world
        self.ground_truth_semantics = False
        
        # Task storage
        self.tasks: List[Task] = []
        self._curr_obs = None
        self.pose_delta = None
        
        if self.verbose:
            print(f"[INIT] Complete\n")
    
    # =========================================================================
    # Core Methods (from GoatAgent)
    # =========================================================================
    
    def get_subtask_timestep(self) -> int:
        """Get current subtask timestep (from GoatAgent)."""
        if self.seq_goals:
            return self.subtask_timesteps[self.current_task_idx]
        else:
            return self.total_timesteps
    
    def reset(self, scene_id, episode_id, current_task_idx=0):
        """
        Initialize agent state for new episode (from GoatAgent).
        
        Args:
            scene_id: Scene identifier
            episode_id: Episode identifier
            current_task_idx: Starting task index (Scout extension)
        """
        if self.verbose:
            print(f"\n[RESET] Episode {episode_id}")
        
        self.total_timesteps = 0
        if self.seq_goals:
            self.subtask_timesteps = [0] * self.max_subtasks_per_episode
            self.sub_task_timesteps = self.subtask_timesteps  # Alias for backwards compatibility
        self.last_pose = np.zeros(3)
        
        self.semantic_map.init_map_and_pose()
        
        if self.verbose:
            print(f"[RESET] Map shape: global={self.semantic_map.global_map.shape}, "
                  f"local={self.semantic_map.local_map.shape}")
        
        if self.instance_memory is not None:
            self.instance_memory.reset()
        
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.navigate_to_best = False
        
        self.current_task_idx = -1
        self.reset_for_next_task()
        self.planner.reset()
        self.matching.step = 0
        self.inst_goal_id = None
        
        self.stuck_counter = 0
        self.reset_vis_dir(scene_id, episode_id, current_task_idx)
    
    def handle_stop(self, action):
        """Handle stop action (from GoatAgent)."""
        self.reset_for_next_task()
    
    def reset_for_next_task(self) -> None:
        """Reset for a new task within same episode (from GoatAgent)."""
        self.inst_goal_id = None
        if self.seq_goals:
            self.current_task_idx += 1
        self.navigate_to_best = False
        self.match_memory = True
        self.planner.reset_for_next_task()
    
    def _update_steps(self):
        """Update timestep counters (from GoatAgent)."""
        self.total_timesteps += 1
        if self.seq_goals:
            self.subtask_timesteps[self.current_task_idx] += 1
            self.sub_task_timesteps = self.subtask_timesteps  # Keep alias in sync
        self.matching.step = self.total_timesteps
        self.semantic_map_module.timestep = self.get_subtask_timestep()
        self.planner.total_timesteps = self.total_timesteps
        self.planner.timestep = self.get_subtask_timestep()
        self.log.info(
            f"---------------- Updating state - step:{self.get_subtask_timestep()} ----------------"
        )
        self.log.debug(
            f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB"
        )
    
    def update_state(self, obs: Observations):
        """Update agent state with new observation (from GoatAgent)."""
        if self.verbose:
            print(f"\n{'='*80}\n[UPDATE_STATE] Step {self.get_subtask_timestep() + 1}\n{'='*80}")
        
        self._curr_obs = obs
        self._update_steps()
        self._update_pose()
        self._update_maps()
        self._search_for_goal()
        
        if self.verbose:
            print(f"{'='*80}\n")
    
    def act(self, **kwargs) -> Tuple[DiscreteNavigationAction, Dict[str, Any], bool]:
        """
        Generate action based on current state (from GoatAgent + Scout extensions).
        
        Returns:
            action: DiscreteNavigationAction
            info: Visualization info dict
            stuck: Whether agent is stuck/timed out
        """
        if self.verbose:
            print(f"\n[ACT] Step {self.get_subtask_timestep()}")
        
        # Check stuck/timeout (from GoatAgent)
        stuck = False
        if self.get_subtask_timestep() >= self.max_steps or self.stuck_counter > 30:
            self.log.warning(
                "Reached max number of steps for subgoal, or stuck somewhere, calling STOP"
            )
            stuck = True
        
        # Scout-specific: Check goal proximity
        if not stuck and self._check_goal_reached():
            if self.verbose:
                print("[ACT] ✓✓✓ GOAL REACHED ✓✓✓")
            return DiscreteNavigationAction.STOP, {}, False
        
        # Get action (from GoatAgent)
        action, vis_inputs = self._get_best_action(**kwargs)
        action = self._process_action(action)
        info = self._get_vis_info(vis_inputs, action)
        
        if self.verbose:
            print(f"[ACT] Action: {action}\n")
        
        if action["action"] == DiscreteNavigationAction.STOP:
            self.handle_stop(action)
        
        # Return raw action for Scout (not dict)
        return action["action"], info if info else {}, stuck
    
    # =========================================================================
    # Map Update (from GoatAgent)
    # =========================================================================
    
    def _update_pose(self):
        """Update pose from observations (from GoatAgent)."""
        obs = self._curr_obs
        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_pose), device=self.device
        )
        self.last_pose = curr_pose
        self.pose_delta = pose_delta
        
        if torch.norm(self.pose_delta[:2]).item() < 0.05:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
    
    def _update_maps(self):
        """Update semantic maps with current observation."""
        obs_preprocessed, instance_scores, category_scores = self._preprocess_obs(self._curr_obs)
        
        if self.verbose:
            print(f"[UPDATE_MAPS] obs shape: {obs_preprocessed.shape}")
            print(f"[UPDATE_MAPS] pose_delta shape: {self.pose_delta.shape}")
            print(f"[UPDATE_MAPS] local_map shape: {self.semantic_map.local_map.shape}")
            print(f"[UPDATE_MAPS] global_map shape: {self.semantic_map.global_map.shape}")
        
        # semantic_map_module.forward expects:
        #   obs: [C, H, W] - NO batch dimension
        #   pose_delta: [3] - NO batch dimension  
        #   state: semantic_map state object
        #   instance_scores: numpy array or None
        #   category_scores: dict or None
        
        # Ensure no batch dimensions
        if obs_preprocessed.dim() == 4:
            obs_preprocessed = obs_preprocessed.squeeze(0)
        if self.pose_delta.dim() == 2:
            pose_delta = self.pose_delta.squeeze(0)
        else:
            pose_delta = self.pose_delta
        
        if self.verbose:
            print(f"[UPDATE_MAPS] Final obs shape: {obs_preprocessed.shape}")
            print(f"[UPDATE_MAPS] Final pose_delta shape: {pose_delta.shape}")
        
        # Call semantic_map_module with state object (GoatAgent style)
        self.semantic_map_module(
            obs_preprocessed,
            pose_delta,
            self.semantic_map,
            instance_scores,
            category_scores
        )
        
        if self.verbose:
            local_map = self.semantic_map.local_map
            global_map = self.semantic_map.global_map
            print(f"[UPDATE_MAPS] local_map[0] (obstacles): sum={local_map[0].sum():.2f}")
            print(f"[UPDATE_MAPS] local_map[1] (explored): sum={local_map[1].sum():.2f}")
            print(f"[UPDATE_MAPS] global_pose: {self.semantic_map.global_pose}")
    
    def _preprocess_tasks(self, tasks_obs) -> List[Task]:
        """Preprocess task observations into Task objects (from GoatAgent)."""
        tasks = []
        for task_obs in tasks_obs:
            task = Task(
                type=task_obs["type"],
                goal_semantic_id=task_obs["semantic_id"]
            )
            if task.type == "imagenav":
                task.goal_image = task_obs.get("image")
                if task.goal_image is not None:
                    task.goal_image_processed, task.goal_image_keypoints = (
                        self.matching.get_goal_image_keypoints(task.goal_image)
                    )
            elif task.type == "languagenav":
                task.goal_description = task_obs.get("description", "")
            tasks.append(task)
        return tasks
    
    def _preprocess_obs(self, obs: Observations):
        """
        Preprocess observations for map update.
        
        CHANGED from GoatAgent: No Detic/YOLO inference - semantics come pre-computed.
        
        NOTE: The observation is NOT downscaled here. The semantic_map_module should
        handle downscaling internally via avg_pooling_layer. If you get shape errors,
        check that line ~728 in categorical_2d_semantic_map_module.py uses:
            feat[1:, :] = self.avg_pooling_layer(obs[4:, :, :]).view(...)
        NOT:
            feat[1:, :] = obs[4:, :, :].view(...)
        """
        if self.verbose:
            print(f"[PREPROCESS] Input depth: min={obs.depth.min():.3f}, max={obs.depth.max():.3f}")
            print(f"[PREPROCESS] Input rgb shape: {obs.rgb.shape}")
            print(f"[PREPROCESS] Input semantic shape: {obs.semantic.shape}")
        
        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0  # m to cm
        
        if self.verbose:
            print(f"[PREPROCESS] Depth (cm): min={depth.min():.1f}, max={depth.max():.1f}")
        
        # One-hot encode semantics (from GoatAgent)
        semantic = torch.eye(self.num_sem_categories + 1, device=self.device)[
            torch.from_numpy(obs.semantic).to(self.device).long()
        ][:, :, 1:]  # Remove background class
        
        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)
        
        # Instance tracking (from GoatAgent)
        inst_scores = None
        if self.record_instance_ids and "instance_frame" in obs.task_observations:
            instance_frame = obs.task_observations["instance_frame"]
            unique_ids, new_instance_frame = np.unique(instance_frame, return_inverse=True)
            new_instance_frame = new_instance_frame.reshape(instance_frame.shape)
            new_instance_frame = torch.from_numpy(new_instance_frame).to(self.device).long()
            
            # One-hot encode based on remapped indices (0 to len(unique_ids)-1)
            instance_frame_onehot = torch.eye(len(unique_ids), device=self.device)[new_instance_frame]
            
            # Check if background (0) is in unique_ids (it's always first if present due to sorting)
            has_background = len(unique_ids) > 0 and unique_ids[0] == 0
            if has_background:
                instance_frame_onehot = instance_frame_onehot[:, :, 1:]  # Remove background channel
            
            # Get instance scores
            # After np.unique with return_inverse, new_instance_frame has values 0 to N-1
            # where 0 is background (if present), 1 is first instance, etc.
            # inst_scores should have one score per non-background instance
            num_instances = len(unique_ids) - 1 if has_background else len(unique_ids)
            
            if "instance_scores" in obs.task_observations and num_instances > 0:
                raw_scores = obs.task_observations["instance_scores"]
                
                if len(raw_scores) >= num_instances:
                    # Take first num_instances scores
                    inst_scores = np.array(raw_scores[:num_instances])
                else:
                    # Pad with 1.0 if not enough scores
                    inst_scores = np.ones(num_instances)
                    inst_scores[:len(raw_scores)] = raw_scores
            else:
                # Default scores of 1.0 for all instances
                inst_scores = np.ones(max(0, num_instances))
            
            obs_preprocessed = torch.cat([obs_preprocessed, instance_frame_onehot], dim=-1)
            
            if self.verbose:
                print(f"[PREPROCESS] Instances: {len(unique_ids)}, has_bg: {has_background}")
                print(f"[PREPROCESS] unique_ids: {unique_ids[:10]}...")  # Show first 10
                print(f"[PREPROCESS] inst_scores: {inst_scores[:10] if len(inst_scores) > 0 else 'empty'}...")
        
        # Permute to [C, H, W] - NO batch dimension, NO downscaling
        # The semantic_map_module handles downscaling internally
        obs_preprocessed = obs_preprocessed.permute(2, 0, 1)
        
        # Process tasks
        self.tasks = self._preprocess_tasks(obs.task_observations["tasks"])
        
        if self.verbose:
            print(f"[PREPROCESS] Output shape: {obs_preprocessed.shape}")
        
        # Scout doesn't use YOLO, category_scores is None
        return obs_preprocessed, inst_scores, None
    
    # =========================================================================
    # Goal Search (from GoatAgent)
    # =========================================================================
    
    @torch.no_grad()
    def _search_for_goal(self, select_best=False):
        """
        Search for goal in current observation and memory (from GoatAgent).
        
        Args:
            select_best: Force matching to best object even if below threshold
        """
        if self.inst_goal_id is not None and self.get_subtask_timestep() % 10 != 0:
            self.log.info(f"Already found instance goal, not searching anymore.")
        else:
            # Match goal against instances (from GoatAgent)
            inst_goal_id = self.matching.search_for_goal(
                self.tasks[self.current_task_idx],
                self.match_memory,
                self.semantic_map.global_pose,
                score_thresh=0 if select_best else None
            )
            if inst_goal_id is not None:
                self.inst_goal_id = inst_goal_id
                if self.verbose:
                    print(f"[SEARCH] ✓ Found goal: instance {self.inst_goal_id}")
        
        self.match_memory = False
    
    def _get_best_action(self, **kwargs):
        """Get best action from planner (from GoatAgent)."""
        if self.verbose:
            print(f"[PLAN] inst_goal_id={self.inst_goal_id}")
        
        task = self.tasks[self.current_task_idx]
        action, vis_input = self.planner.plan(
            self.inst_goal_id,
            task.goal_semantic_id,
        )
        
        if action is not None:
            if self.verbose:
                print(f"[PLAN] ✓ Planner: {action}")
            return action, vis_input
        
        if self.inst_goal_id is not None:
            self.log.info("Couldn't navigate to goal, stopping")
            return DiscreteNavigationAction.STOP, {}
        
        # No reachable frontiers, force memory match (from GoatAgent)
        self.log.info("No reachable frontiers, forcing a match against memory")
        self.match_memory = True
        self._search_for_goal(select_best=True)
        
        if self.inst_goal_id is None:
            self.log.info("No match found in memory, stopping.")
            return DiscreteNavigationAction.STOP, {}
        
        # Retry planning with new goal
        return self._get_best_action(**kwargs)
    
    # =========================================================================
    # Visualization (from GoatAgent)
    # =========================================================================
    
    def _get_vis_info(self, vis_inputs, action):
        """Get visualization info dict (from GoatAgent)."""
        if self.visualization_level < 1:
            return None
        
        if vis_inputs is None:
            vis_inputs = {}
        
        is_local = vis_inputs.get("is_local", True)
        obs = self._curr_obs
        
        info = {
            "agent_id": self.agent_id,
            "rgb_frame": obs.rgb[:, :, ::-1] if obs else None,
            "depth_frame": obs.depth if obs else None,
            "semantic_frame": (
                obs.semantic if obs.task_observations.get("semantic_frame") is None
                else obs.task_observations["semantic_frame"]
            ) if obs else None,
            "top_down_map": obs.task_observations.get("top_down_map") if obs else None,
            "is_collision": False,
            "inst_goal_id": self.inst_goal_id,
            "timestep": self.get_subtask_timestep(),
            "explored_map": self.semantic_map.get_explored_map(is_local),
            "semantic_map_1D": self.semantic_map.get_semantic_map_1D(is_local),
            "been_close_map": self.semantic_map.get_been_close_map(is_local),
            "visited_map": self.semantic_map.get_visited_map(is_local),
            "robot_loc": self.semantic_map.get_loc(is_local),
            "robot_orientation": self.semantic_map.global_pose.cpu()[2],
            "instance_memory": self.instance_memory,
            **vis_inputs,
        }
        
        if "obstacle_map" not in info:
            info["obstacle_map"] = self.semantic_map.get_obstacle_map(is_local)
        
        self._get_task_info(obs, action, info)
        return info
    
    def _get_task_info(self, obs, action, info):
        """Add task-specific info to visualization dict (from GoatAgent)."""
        if obs is None:
            return
        
        current_task = obs.task_observations["tasks"][self.current_task_idx]
        info["task_type"] = current_task["type"]
        goal_text_desc = {x: y for x, y in current_task.items() if x != "image"}
        info["caption"] = str(goal_text_desc)
        
        if current_task["type"] == "imagenav":
            info["goal_image"] = current_task.get("image")
        else:
            info["third_person_image"] = getattr(obs, 'third_person_image', None)
        
        action_name = action.get("action", action) if isinstance(action, dict) else action
        info["caption"] += f" | Action: {str(action_name).split('.')[-1]}"
    
    def reset_vis_dir(self, scene_id, episode_id, current_task_idx=None):
        """Reset visualization directory (from GoatAgent)."""
        if self.seq_goals:
            dir_name = f"{scene_id}_{episode_id}_{current_task_idx}"
        else:
            dir_name = f"{scene_id}_{episode_id}"
        
        if self.agent_id is not None:
            dir_name = os.path.join(dir_name, f"agent_{self.agent_id}")
        
        self.planner.set_vis_dir(dir_name)
        self.matching.set_vis_dir(dir_name)
        self.semantic_map.vis_dir = self.planner.vis_dir
        self.semantic_map_module.vis_dir = self.planner.vis_dir
    
    def _process_action(self, action):
        """Wrap action in dict format (from GoatAgent)."""
        return {
            "action": action,
            "action_args": {
                "agent_id": 0 if self.agent_id is None else self.agent_id,
                "task_idx": self.current_task_idx,
            },
        }
    
    # =========================================================================
    # Scout-Specific Methods (not in GoatAgent)
    # =========================================================================
    
    def _check_goal_reached(self) -> bool:
        """
        Check if robot is within 0.75m of goal.
        Scout-specific method for real-world proximity-based stopping.
        """
        # Skip during panorama
        if self.get_subtask_timestep() <= self.panorama_start_steps:
            return False
        
        if not self.tasks or self.current_task_idx >= len(self.tasks):
            return False
        
        task = self.tasks[self.current_task_idx]
        
        # ObjectNav: check proximity to category
        if task.type == "objectnav":
            return self._check_close_to_category(task.goal_semantic_id)
        
        # ImageNav/LanguageNav: check proximity to instance
        if self.inst_goal_id is not None:
            return self._check_close_to_instance(self.inst_goal_id)
        
        return False
    
    def _check_close_to_category(self, category_id: int) -> bool:
        """Check if robot is within 0.75m of target category."""
        if self.verbose:
            print(f"[GOAL_CHECK] Category {category_id}")
        
        # Get robot position in global map
        robot_pose = self.semantic_map.global_pose
        robot_x = int(robot_pose[0].item())
        robot_y = int(robot_pose[1].item())
        
        # Get category map from global map
        semantic_map = self.semantic_map.global_map[0, 4:4+self.num_sem_categories]
        
        if category_id >= semantic_map.shape[0]:
            return False
        
        category_map = semantic_map[category_id]
        total_pixels = (category_map > 0).sum().item()
        
        if self.verbose:
            print(f"[GOAL_CHECK] Robot: ({robot_x},{robot_y}), Cat {category_id} pixels: {total_pixels}")
        
        if total_pixels == 0:
            return False
        
        # Check 75cm radius
        map_resolution = self.config.AGENT.SEMANTIC_MAP.map_resolution
        radius_cells = int(75 / map_resolution)
        
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx*dx + dy*dy > radius_cells*radius_cells:
                    continue
                
                y = robot_y + dy
                x = robot_x + dx
                
                if 0 <= y < category_map.shape[0] and 0 <= x < category_map.shape[1]:
                    if category_map[y, x] > 0:
                        dist_cm = np.sqrt(dx*dx + dy*dy) * map_resolution
                        if self.verbose:
                            print(f"[GOAL_CHECK] ✓ Found at {dist_cm:.1f}cm")
                        return True
        
        return False
    
    def _check_close_to_instance(self, instance_id: int) -> bool:
        """Check if robot is within 0.75m of target instance."""
        if self.instance_memory is None:
            return False
        
        if not self.instance_memory.instance_exists(instance_id):
            return False
        
        instance = self.instance_memory.get_instance(instance_id)
        robot_pose = self.semantic_map.global_pose
        
        if hasattr(instance, 'map_locs') and len(instance.map_locs) > 0:
            locs = torch.stack([loc for loc in instance.map_locs])
            inst_x = locs[:, 0].mean().item()
            inst_y = locs[:, 1].mean().item()
            
            robot_x = robot_pose[0].item()
            robot_y = robot_pose[1].item()
            
            map_resolution = self.config.AGENT.SEMANTIC_MAP.map_resolution
            dist_cells = np.sqrt((robot_x - inst_x)**2 + (robot_y - inst_y)**2)
            dist_cm = dist_cells * map_resolution
            
            if dist_cm <= 75:  # 0.75m
                if self.verbose:
                    print(f"[GOAL_CHECK] ✓ Instance {instance_id} at {dist_cm:.1f}cm")
                return True
        
        return False