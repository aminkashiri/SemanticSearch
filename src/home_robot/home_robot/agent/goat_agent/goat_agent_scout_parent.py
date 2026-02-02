#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Scout GOAT Agent - Real-world multi-task navigation agent
Inherits from GoatAgent, overriding only real-world specific behavior
"""

import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple

from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)
from home_robot.navigation_planner.fixed_discrete_planner import DiscretePlanner

# Import base class
from home_robot.agent.goat_agent.goat_agent import GoatAgent, Task


class ScoutGoatAgent(GoatAgent):
    """
    Real-world GOAT agent for Scout robot.
    Inherits from GoatAgent, overriding methods that need real-world adaptations.
    
    Key differences from GoatAgent:
    - Uses config.ENVIRONMENT instead of config.habitat.simulator for camera params
    - No built-in perception (semantics provided externally via observations)
    - Simplified action return format (raw action, not dict)
    - Additional goal proximity checking for real-world stopping
    """
    
    def __init__(self, config, vocabulary, agent_id=None, device_id: int = 0):
        """
        Initialize ScoutGoatAgent.
        
        Calls parent __init__ but overrides:
        - semantic_map_module (uses ENVIRONMENT config)
        - planner (uses ENVIRONMENT.turn_angle)
        - Skips perception model setup (Detic/YOLO)
        """
        # Store config before parent init (needed for overrides)
        self._scout_config = config
        self.verbose = getattr(config, 'VERBOSE', False)
        
        if self.verbose:
            print(f"\n{'='*60}\n[INIT] ScoutGoatAgent\n{'='*60}")
        
        # === Call parent __init__ ===
        # This sets up: instance_memory, matching, semantic_map, basic state
        # We'll override semantic_map_module and planner after
        super().__init__(
            config=config,
            vocabulary=vocabulary,
            agent_id=agent_id,
            device_id=device_id
        )
        
        # === Override: Force ground_truth_semantics = False ===
        # Real world never has GT semantics from simulator
        self.ground_truth_semantics = False
        
        # === Override: Disable built-in perception ===
        # Scout receives semantics externally (from ROS perception node)
        self.segmentation = None
        self.yolo = None
        self.use_yolo = False
        
        # === Override: semantic_map_module with ENVIRONMENT config ===
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
            print_images=self.visualization_level > 2 if hasattr(self, 'visualization_level') else False
        )
        
        # === Override: planner with ENVIRONMENT.turn_angle ===
        if config.AGENT.panorama_start:
            panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            panorama_start_steps = 0
        
        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualization_level=getattr(config, 'VISUALIZATION_LEVEL', 0),
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=config.AGENT.PLANNER.discrete_actions,
            min_goal_distance_cm=config.AGENT.PLANNER.min_goal_distance_cm,
            panorama_start_steps=panorama_start_steps,
            instance_memory=self.instance_memory,
            goal_filtering=config.AGENT.SEMANTIC_MAP.goal_filtering,
            semantic_map=self.semantic_map,
            frontier_metric=config.AGENT.frontier_metric,
            agent_id=self.agent_id,
            ground_truth_semantics=False,  # Always false for real world
            task_type=self.task_type
        )
        
        # Scout-specific state
        self.panorama_start_steps = panorama_start_steps
        
        if self.verbose:
            print(f"[INIT] Device: {self.device}, Categories: {self.num_sem_categories}")
            print(f"[INIT] Panorama: {panorama_start_steps} steps")
            print(f"[INIT] Complete\n")
    
    # =========================================================================
    # Override: _preprocess_obs - Scout receives pre-computed semantics
    # =========================================================================
    
    def _preprocess_obs(self, obs: Observations):
        """
        Preprocess observations for Scout robot.
        
        Unlike GoatAgent, Scout receives semantics from external perception node,
        so we skip Detic/YOLO inference and directly use obs.semantic.
        """
        if self.verbose:
            print(f"[PREPROCESS] Input depth: min={obs.depth.min():.3f}, max={obs.depth.max():.3f}")
        
        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0  # m to cm
        
        if self.verbose:
            print(f"[PREPROCESS] After *100 (cm): min={depth.min():.1f}, max={depth.max():.1f}")
        
        # One-hot encode semantics (same as GoatAgent)
        semantic = torch.eye(self.num_sem_categories + 1, device=self.device)[
            torch.from_numpy(obs.semantic).to(self.device).long()
        ][:, :, 1:]  # Remove background class
        
        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)
        
        # Instance tracking (same as GoatAgent)
        inst_scores = None
        if self.record_instance_ids and "instance_frame" in obs.task_observations:
            instance_frame = obs.task_observations["instance_frame"]
            unique_ids, new_instance_frame = np.unique(instance_frame, return_inverse=True)
            new_instance_frame = new_instance_frame.reshape(instance_frame.shape)
            new_instance_frame = torch.from_numpy(new_instance_frame).to(self.device).long()
            
            # One-hot encode
            instance_frame_onehot = torch.eye(len(unique_ids), device=self.device)[new_instance_frame]
            if unique_ids[0] == 0:
                instance_frame_onehot = instance_frame_onehot[:, :, 1:]  # Remove background
            
            # Get instance scores if available
            if "instance_scores" in obs.task_observations:
                inst_scores = np.concatenate(([0], obs.task_observations["instance_scores"]))[unique_ids][1:]
            else:
                inst_scores = np.ones(len(unique_ids) - 1)  # Default scores
            
            obs_preprocessed = torch.cat([obs_preprocessed, instance_frame_onehot], dim=-1)
            
            if self.verbose:
                print(f"[PREPROCESS] Instances: {len(unique_ids)}")
        
        obs_preprocessed = obs_preprocessed.permute(2, 0, 1)
        
        # Process tasks (same as GoatAgent)
        self.tasks = self._preprocess_tasks(obs.task_observations["tasks"])
        
        if self.verbose:
            print(f"[PREPROCESS] Output: {obs_preprocessed.shape}")
        
        # Scout doesn't use YOLO, so category_scores is None
        return obs_preprocessed, inst_scores, None
    
    # =========================================================================
    # Override: act - Different return format + goal proximity check
    # =========================================================================
    
    def act(self, **kwargs) -> Tuple[DiscreteNavigationAction, Dict[str, Any], bool]:
        """
        Act for Scout robot.
        
        Differences from GoatAgent:
        - Returns raw DiscreteNavigationAction instead of action dict
        - Adds goal proximity checking via _check_goal_reached()
        """
        if self.verbose:
            print(f"\n[ACT] Step {self.get_subtask_timestep()}")
        
        # Check stuck/timeout (same as GoatAgent)
        stuck = False
        if self.get_subtask_timestep() >= self.max_steps or self.stuck_counter > 30:
            self.log.warning(
                "Reached max number of steps for subgoal, or stuck somewhere, calling STOP"
            )
            stuck = True
        
        # Scout-specific: Check if goal reached by proximity
        if not stuck and self._check_goal_reached():
            if self.verbose:
                print("[ACT] ✓✓✓ GOAL REACHED ✓✓✓")
            return DiscreteNavigationAction.STOP, {}, False
        
        # Get action from planner (using parent's _get_best_action)
        action, vis_inputs = self._get_best_action(**kwargs)
        
        # Process action - extract raw action for Scout
        if isinstance(action, dict):
            raw_action = action.get("action", action)
        else:
            raw_action = action
        
        info = self._get_vis_info(vis_inputs, {"action": raw_action})
        
        if self.verbose:
            print(f"[ACT] Action: {raw_action}\n")
        
        # Handle stop action
        if raw_action == DiscreteNavigationAction.STOP:
            self.handle_stop({"action": raw_action})
        
        return raw_action, info if info else {}, stuck
    
    # =========================================================================
    # Override: reset - Slightly different signature
    # =========================================================================
    
    def reset(self, scene_id, episode_id, current_task_idx=0):
        """
        Reset agent state for new episode.
        
        Extended from GoatAgent to accept current_task_idx parameter.
        """
        if self.verbose:
            print(f"\n[RESET] Episode {episode_id}")
        
        # Call parent reset
        super().reset(scene_id, episode_id)
        
        # Reset visualization directory with task index
        self.reset_vis_dir(scene_id, episode_id, current_task_idx)
        
        if self.verbose:
            print(f"[RESET] Map shape: global={self.semantic_map.global_map.shape}, "
                  f"local={self.semantic_map.local_map.shape}")
    
    # =========================================================================
    # Scout-specific methods (not in GoatAgent)
    # =========================================================================
    
    def _check_goal_reached(self) -> bool:
        """
        Check if robot is within 0.75m of goal.
        Scout-specific method for real-world proximity-based stopping.
        """
        # Skip during panorama
        if self.get_subtask_timestep() <= getattr(self, 'panorama_start_steps', 0):
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
        map_resolution = self._scout_config.AGENT.SEMANTIC_MAP.map_resolution
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
            
            map_resolution = self._scout_config.AGENT.SEMANTIC_MAP.map_resolution
            dist_cells = np.sqrt((robot_x - inst_x)**2 + (robot_y - inst_y)**2)
            dist_cm = dist_cells * map_resolution
            
            if dist_cm <= 75:  # 0.75m
                if self.verbose:
                    print(f"[GOAL_CHECK] ✓ Instance {instance_id} at {dist_cm:.1f}cm")
                return True
        
        return False
    
    # =========================================================================
    # Override: _get_vis_info - Handle missing attributes gracefully
    # =========================================================================
    
    def _get_vis_info(self, vis_inputs, action):
        """
        Get visualization info, with graceful handling for Scout-specific cases.
        """
        if getattr(self, 'visualization_level', 0) < 1:
            return {}
        
        try:
            return super()._get_vis_info(vis_inputs, action)
        except Exception as e:
            if self.verbose:
                print(f"[VIS_INFO] Warning: {e}")
            
            # Fallback minimal info
            is_local = vis_inputs.get("is_local", True) if vis_inputs else True
            return {
                "inst_goal_id": self.inst_goal_id,
                "timestep": self.get_subtask_timestep(),
                "explored_map": self.semantic_map.get_explored_map(is_local),
                "obstacle_map": self.semantic_map.get_obstacle_map(is_local),
                **(vis_inputs or {})
            }