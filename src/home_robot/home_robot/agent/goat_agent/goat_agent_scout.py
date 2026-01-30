#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Scout GOAT Agent - Real-world multi-task navigation agent
Inherits from Agent, mimics GoatAgent methods adapted for real robot
"""

import json
import torch
import psutil
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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

# Try importing GoatMatching
try:
    from home_robot.agent.goat_agent.goat_matching import GoatMatching
    MATCHING_AVAILABLE = True
except ImportError:
    print("Warning: GoatMatching not found. ImageNav and LanguageNav will be disabled.")
    MATCHING_AVAILABLE = False


class ScoutGoatAgent(Agent):
    """
    Real-world GOAT agent for Scout robot
    Based on GoatAgent but adapted for real robot (no Habitat dependencies)
    """
    
    def __init__(self, config, semantic_category_mapping, agent_id=None, device_id: int = 0):
        """Initialize agent (based on GoatAgent.__init__)"""
        
        # Get verbose flag
        self.verbose = getattr(config, 'VERBOSE', False)
        
        if self.verbose:
            print(f"\n{'='*60}\n[INIT] ScoutGoatAgent\n{'='*60}")
        
        self.agent_id = agent_id
        self.log = get_logger(agent_id=agent_id)
        self.config = config
        
        # From GoatAgent: max_steps per sub-task
        self.max_steps = [config.AGENT.max_steps] * config.ENVIRONMENT.max_num_sub_task_episodes
        self.max_num_sub_task_episodes = config.ENVIRONMENT.max_num_sub_task_episodes
        
        # Visualization directories
        self.goal_matching_vis_dir = f"{config.DUMP_LOCATION}/goal_grounding_vis"
        Path(self.goal_matching_vis_dir).mkdir(parents=True, exist_ok=True)
        
        # Device setup (from GoatAgent)
        if config.NO_GPU:
            self.device = torch.device("cpu")
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")
        
        self.semantic_category_mapping = semantic_category_mapping
        self.num_sem_categories = semantic_category_mapping.num_sem_categories
        
        if self.verbose:
            print(f"[INIT] Device: {self.device}, Categories: {self.num_sem_categories}")
        
        # Instance memory setup (from GoatAgent)
        self.record_instance_ids = getattr(config.AGENT.SEMANTIC_MAP, "record_instance_ids", False)
        self.instance_memory = None
        
        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                config.AGENT.SEMANTIC_MAP.du_scale,
                config=config,
                mask_cropped_instances=False,
                padding_cropped_instances=200,
            )
        
        # Goal matching setup (from GoatAgent)
        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None
        self.goal_policy_config = config.AGENT.SUPERGLUE
        
        if MATCHING_AVAILABLE:
            self.matching = GoatMatching(
                device=self.device_id if hasattr(self, 'device_id') else 0,
                score_func=self.goal_policy_config.score_function,
                config=config.AGENT.SUPERGLUE,
                default_vis_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
                print_images=config.PRINT_IMAGES,
                instance_memory=self.instance_memory,
            )
            self.image_matching_function = self.matching.match_image_to_image
            self.matching_fn = {
                "imagenav": self.image_matching_function,
                "languagenav": self.matching.match_language_to_image,
                "objectnav": None,
            }
        else:
            self.matching = None
            self.matching_fn = {"imagenav": None, "languagenav": None, "objectnav": None}
        
        # CHANGED: Real robot camera parameters (not Habitat)
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution))
        
        # CHANGED: Use config.ENVIRONMENT instead of config.habitat.simulator
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
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 500),
            exploration_type=config.AGENT.exploration_type,
            gaze_width=40 if config.AGENT.exploration_type == "raycast" else 30,
            gaze_distance=config.ENVIRONMENT.max_depth if config.AGENT.exploration_type == "raycast" else 3,
            agent_cell_radius=agent_cell_radius,
        )
        
        self.inst_goal_id = None
        self.inst_goal_found = False
        self.visualize = config.VISUALIZE or config.PRINT_IMAGES
        
        # Semantic map state (from GoatAgent)
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_sem_categories=self.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            record_instance_ids=self.record_instance_ids,
            instance_memory=self.instance_memory,
            agent_id=agent_id,
        )
        
        # CHANGED: Use config.ENVIRONMENT.turn_angle instead of habitat.simulator.turn_angle
        if config.AGENT.panorama_start:
            panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            panorama_start_steps = 0
        
        self.panorama_start_steps = panorama_start_steps
        
        if self.verbose:
            print(f"[INIT] Panorama: {panorama_start_steps} steps")
        
        # Planner setup (from GoatAgent)
        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,  # CHANGED
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualize=config.VISUALIZE,
            print_images=config.PRINT_IMAGES,
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
        )
        
        # Episode state (from GoatAgent)
        self.sub_task_timesteps = None
        self.total_timesteps = None
        self.last_pose = None
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.current_task_idx = 0
        self.navigate_to_best = False
        self.stuck_counter = 0
        
        # CHANGED: Always False for real world (no GT semantics)
        self.ground_truth_semantics = False
        
        if self.verbose:
            print(f"[INIT] Complete\n")
    
    # ========================================================================
    # Core Methods (copied from GoatAgent)
    # ========================================================================
    
    def get_subtask_timestep(self) -> int:
        """From GoatAgent.get_subtask_timestep"""
        return self.sub_task_timesteps[self.current_task_idx]
    
    def reset(self, scene_id, episode_id, current_task_idx):
        """From GoatAgent.reset"""
        if self.verbose:
            print(f"\n[RESET] Episode {episode_id}")
        
        self.total_timesteps = 0
        self.sub_task_timesteps = [0] * self.max_num_sub_task_episodes
        self.last_pose = np.zeros(3)
        
        self.semantic_map.init_map_and_pose()
        
        if self.verbose:
            print(f"[RESET] Map shape: global={self.semantic_map.global_map.shape}, local={self.semantic_map.local_map.shape}")
        
        if self.instance_memory is not None:
            self.instance_memory.reset()
        
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.navigate_to_best = False
        self.current_task_idx = -1
        
        self.reset_sub_episode()
        self.planner.reset()
        
        self.inst_goal_found = False
        self.inst_goal_id = None
        self.stuck_counter = 0
        
        self._reset_vis_dir(scene_id, episode_id, current_task_idx)
    
    def reset_sub_episode(self) -> None:
        """From GoatAgent.reset_sub_episode"""
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self.inst_goal_found = False
        self.inst_goal_id = None
        self.current_task_idx += 1
        self.navigate_to_best = False
    
    def update_state(self, obs):
        """From GoatAgent.update_state"""
        if self.verbose:
            print(f"\n{'='*80}\n[UPDATE_STATE] Step {self.get_subtask_timestep() + 1}\n{'='*80}")
        
        self.current_task = obs.task_observations["tasks"][self.current_task_idx]
        self.total_timesteps = self.total_timesteps + 1
        self.sub_task_timesteps[self.current_task_idx] += 1
        self.semantic_map_module.timestep = self.get_subtask_timestep()
        
        if self.verbose:
            print(f"[UPDATE_STATE] Task: {self.current_task['type']}, Timestep: {self.get_subtask_timestep()}")
        
        self.log.info(f"---------------- Subtask step {self.get_subtask_timestep()} ----------------")
        self.log.debug(f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")
        
        # Preprocess
        if self.verbose:
            print("[UPDATE_STATE] Preprocessing obs...")
        obs_preprocessed, pose_delta = self._preprocess_obs(obs)
        
        # Update maps
        if self.verbose:
            print("[UPDATE_STATE] Updating maps...")
        self._update_maps(obs_preprocessed, pose_delta)
        
        # Search for goal (only after panorama)
        if self.get_subtask_timestep() > self.panorama_start_steps:
            if self.verbose:
                print("[UPDATE_STATE] Searching for goal...")
            self._search_for_goal()
        elif self.verbose:
            print(f"[UPDATE_STATE] Panorama: {self.get_subtask_timestep()}/{self.panorama_start_steps}")
        
        # Check stuck
        if torch.norm(pose_delta[:2]).item() < 0.05:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        if self.verbose:
            print(f"{'='*80}\n")
    
    def act(self, other_agents=None) -> Tuple[DiscreteNavigationAction, Dict[str, Any], bool]:
        """
        From GoatAgent.act but with modified signature
        Returns: (action, info, stuck) instead of (action, info)
        """
        if self.verbose:
            print(f"\n[ACT] Step {self.get_subtask_timestep()}")
        
        # CHANGED: No neighbors for single robot (removed communicate call)
        neighbors = []
        
        # Check timeout/stuck (from GoatAgent)
        stuck = False
        if (self.get_subtask_timestep() >= self.max_steps[self.current_task_idx]) or self.stuck_counter > 30:
            self.log.warning("Reached max number of steps for subgoal, or stuck somewhere, calling STOP")
            stuck = True
        
        # ADDED: Check if goal reached (within 0.75m)
        if not stuck and self._check_goal_reached():
            if self.verbose:
                print("[ACT] ✓✓✓ GOAL REACHED ✓✓✓")
            return DiscreteNavigationAction.STOP, {}, False
        
        # Plan action (from GoatAgent)
        action, vis_inputs = self._get_best_action(neighbors)
        info = self._get_vis_info(vis_inputs)
        
        if self.verbose:
            print(f"[ACT] Action: {action}\n")
        
        return action, info, stuck
    
    # ========================================================================
    # Map Update (from GoatAgent)
    # ========================================================================
    
    def _preprocess_obs(self, obs: Observations):
        if self.verbose:
            print(f"[PREPROCESS] Input depth: min={obs.depth.min():.3f}, max={obs.depth.max():.3f}")
        
        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0  # m to cm
        
        if self.verbose:
            print(f"[PREPROCESS] After *100 (cm): min={depth.min():.1f}, max={depth.max():.1f}")
        
        # One-hot encode semantics (EXACT from GoatAgent)
        semantic = torch.eye(self.num_sem_categories + 1, device=self.device)[
            torch.from_numpy(obs.semantic).to(self.device).long()  # ADDED .long()
        ][:, :, 1:]  # Remove background class
        
        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)
        
        # Instance tracking (from GoatAgent)
        if self.record_instance_ids and "instance_frame" in obs.task_observations:
            instances = obs.task_observations["instance_frame"]
            instance_ids, instances_idx = np.unique(instances, return_inverse=True)
            instances_idx = instances_idx.reshape(instances.shape)
            instances = torch.from_numpy(instances_idx).to(self.device).long()  # ADDED .long()
            
            # One-hot encode
            instance_frame_onehot = torch.eye(len(instance_ids), device=self.device)[instances]
            obs_preprocessed = torch.cat([obs_preprocessed, instance_frame_onehot], dim=-1)
            
            if self.verbose:
                print(f"[PREPROCESS] Instances: {len(instance_ids)}")
        
        # Permute (EXACT from GoatAgent)
        obs_preprocessed = obs_preprocessed.permute(2, 0, 1)
        
        # Pose delta (EXACT from GoatAgent)
        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_pose), device=self.device
        )
        self.last_pose = curr_pose
        
        if self.verbose:
            print(f"[PREPROCESS] Output: {obs_preprocessed.shape}, pose_delta: {pose_delta.cpu().numpy()}")
        
        return obs_preprocessed, pose_delta
    
    def _update_maps(self, obs: torch.Tensor, pose_delta: torch.Tensor):
        """From GoatAgent._update_maps - EXACT COPY with verbose added"""
        if self.verbose:
            print(f"[UPDATE_MAPS] BEFORE: global={self.semantic_map.global_map.shape}")
        
        # EXACT from GoatAgent
        (
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
        ) = self.semantic_map_module(
            obs,
            pose_delta,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
        )
        if self.verbose:
            local_map = self.semantic_map.local_map[0]  # Remove batch dim
            print(f"[UPDATE_MAPS] Local map shape: {local_map.shape}")
            print(f"[UPDATE_MAPS] Channel 0 (obstacles): min={local_map[0].min():.3f}, max={local_map[0].max():.3f}, mean={local_map[0].mean():.3f}")
            print(f"[UPDATE_MAPS] Channel 1 (explored): min={local_map[1].min():.3f}, max={local_map[1].max():.3f}, mean={local_map[1].mean():.3f}")
            print(f"[UPDATE_MAPS] Obstacle pixels (>0.5): {(local_map[0] > 0.5).sum().item()}")
            print(f"[UPDATE_MAPS] Explored pixels (>0.5): {(local_map[1] > 0.5).sum().item()}")
            print(f"[UPDATE_MAPS] Free space (explored & !obstacle): {((local_map[1] > 0.5) & (local_map[0] < 0.5)).sum().item()}")
        # DEBUG: Check map channels
        if self.verbose:
            local_map = self.semantic_map.local_map[0]  # Remove batch dim -> [109, 480, 480]
            print(f"[UPDATE_MAPS] Local map shape: {local_map.shape}")
            print(f"[UPDATE_MAPS] Channel 0 (obstacles): min={local_map[0].min():.3f}, max={local_map[0].max():.3f}, mean={local_map[0].mean():.3f}")
            print(f"[UPDATE_MAPS] Channel 1 (explored): min={local_map[1].min():.3f}, max={local_map[1].max():.3f}, mean={local_map[1].mean():.3f}")
    
    # ========================================================================
    # Goal Search (from GoatAgent)
    # ========================================================================
    
    @torch.no_grad()
    def _search_for_goal(self):
        """From GoatAgent._search_for_goal - adapted for real world"""
        if self.inst_goal_found and self.get_subtask_timestep() % 10 != 0:
            self.log.info(f"Already found instance goal, not searching anymore.")
            return
        
        # CHANGED: First step is 1 not 0 (to match after panorama)
        mem_match_confidences, mem_match_instance_ids = (
            self._match_against_memory() if self.get_subtask_timestep() == 1 else ([], [])
        )
        
        obs_match_confidences, obs_match_instance_ids = self._match_against_current_frame()
        
        self.log.debug(
            f"candidate matches in memory: {len(mem_match_confidences)}, "
            f"candidate matches in observation: {len(obs_match_confidences)}"
        )
        
        if len(mem_match_confidences) > 0 or len(obs_match_confidences) > 0:
            self.inst_goal_found, self.inst_goal_id = self.matching.get_best_inst_goal(
                obs_match_confidences,
                obs_match_instance_ids,
                mem_match_confidences=mem_match_confidences,
                mem_match_instance_ids=mem_match_instance_ids,
                score_thresh=self._score_thresh(),
            )
            
            if self.inst_goal_id is not None:
                if self.verbose:
                    print(f"[SEARCH] ✓ Found goal: instance {self.inst_goal_id}")
    
    def _match_against_current_frame(self):
        """From GoatAgent._match_against_current_frame - EXACT COPY"""
        if not MATCHING_AVAILABLE:
            return [], []
        
        image_goal = None
        language_goal = None
        
        if self.current_task["type"] == "imagenav":
            if self.goal_image is None and "image" in self.current_task:  # ADDED safety check
                img_goal = self.current_task["image"]
                self.goal_image, self.goal_image_keypoints = (
                    self.matching.get_goal_image_keypoints(img_goal)
                )
            image_goal = self.goal_image
        elif self.current_task["type"] == "languagenav":
            language_goal = self.current_task.get("description", "")  # ADDED .get()
        
        if self.matching_fn[self.current_task["type"]] is None:
            return [], []
        
        try:
            confidences, frame_matches_instance_ids = self.matching.get_matches_against_current_frame(
                self.matching_fn[self.current_task["type"]],
                self.total_timesteps,
                image_goal=image_goal,
                goal_image_keypoints=self.goal_image_keypoints,
                language_goal=language_goal,
                categories=[self.current_task["semantic_id"]],
                use_full_image=False,
                global_pose=self.semantic_map.global_pose,
            )
            return confidences, frame_matches_instance_ids
        except Exception as e:
            self.log.error(f"Frame matching failed: {e}")
            return [], []
    
    def _match_against_memory(self):
        """From GoatAgent._match_against_memory - EXACT COPY"""
        if not MATCHING_AVAILABLE:
            return [], []
        
        task_type = self.current_task["type"]
        self.log.info("--------Matching against memory!--------")
        
        image_goal = None
        language_goal = None
        goal_image_keypoints = None
        
        if task_type == "languagenav":
            language_goal = self.current_task.get("description", "")  # ADDED .get()
        elif task_type == "imagenav":
            image_goal = self.goal_image
            goal_image_keypoints = self.goal_image_keypoints
        
        try:
            mem_match_confidences, mem_match_instance_ids = self.matching.get_matches_against_memory(
                self.matching_fn[task_type],
                self.total_timesteps,
                language_goal=language_goal,
                image_goal=image_goal,
                goal_image_keypoints=goal_image_keypoints,
                use_full_image=True,
                categories=[self.current_task["semantic_id"]],
                global_pose=self.semantic_map.global_pose,
            )
            
            if len(mem_match_confidences) > 0:
                stats = {
                    i: {
                        "mean": float(scores.mean()),
                        "median": float(np.median(scores)),
                        "max": float(scores.max()),
                        "min": float(scores.min()),
                        "all": scores.flatten().tolist(),
                    }
                    for i, scores in zip(mem_match_instance_ids, mem_match_confidences)
                }
                with open(
                    f"{self.goal_matching_vis_dir}/goal{self.current_task_idx}_{task_type}_stats.json",
                    "w",
                ) as f:
                    json.dump(stats, f, indent=4)
            
            return mem_match_confidences, mem_match_instance_ids
        except Exception as e:
            self.log.error(f"Memory matching failed: {e}")
            return [], []
    
    def _get_best_action(self, neighbors):
        """From GoatAgent._get_best_action - EXACT COPY"""
        if self.verbose:
            print(f"[PLAN] inst_goal_found={self.inst_goal_found}, inst_goal_id={self.inst_goal_id}")
        
        action, vis_input = self.planner.plan(
            self.inst_goal_found,
            self.inst_goal_id,
            self.get_subtask_timestep(),
            self.total_timesteps,
            self.current_task["semantic_id"],
            neighbors=neighbors,
        )
        
        if action is not None:
            if self.verbose:
                print(f"[PLAN] ✓ Planner: {action}")
            return action, vis_input
        
        self.log.info("No reachable goal/frontier.")
        if self.verbose:
            print(f"[PLAN] ✗ No goal/frontier")
        
        if self.navigate_to_best:
            self.log.info("Already tried the best match. Stopping")
            return DiscreteNavigationAction.STOP, {}
        
        self.navigate_to_best = True
        self.log.info("Forcing a match against memory")
        
        mem_match_confidences, mem_match_instance_ids = self._match_against_memory()
        if not len(mem_match_confidences) > 0:
            self.log.info("No match found in memory. Stopping")
            return DiscreteNavigationAction.STOP, {}
        
        prev_inst_goal_id = self.inst_goal_id
        self.inst_goal_found, self.inst_goal_id = self.matching.get_best_inst_goal(
            mem_match_confidences=mem_match_confidences,
            mem_match_instance_ids=mem_match_instance_ids,
            score_thresh=0,
        )
        
        # REMOVED: assert (not needed in real world)
        if self.inst_goal_id == prev_inst_goal_id:
            self.log.info("Best match is the same as the previous one. Stopping")
            return DiscreteNavigationAction.STOP, {}
        
        action, vis_input = self.planner.plan(
            self.inst_goal_found,
            self.inst_goal_id,
            self.get_subtask_timestep(),
            self.total_timesteps,
            self.current_task["semantic_id"],
            fallback_to_frontier=False,
            postfix="_last_shot",
            neighbors=neighbors,
        )
        
        if action is None:
            self.log.info("Fully explored and no path to our best match. Stopping")
            return DiscreteNavigationAction.STOP, {}
        
        self.log.info("Found a path to the last shot goal. Navigating to it.")
        return action, vis_input
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _get_vis_info(self, vis_inputs):
        """From GoatAgent._get_vis_info - with safety"""
        if not self.visualize:
            return {}  # CHANGED: empty dict instead of None
        
        if vis_inputs is None:  # ADDED safety
            vis_inputs = {}
        
        is_local = vis_inputs.get("is_local", True)
        
        info = {
            "inst_goal_id": self.inst_goal_id,
            "timestep": self.get_subtask_timestep(),
            "total_timesteps": self.total_timesteps,
            "instance_memory": self.instance_memory,
        }
        
        # ADDED: Safe map info extraction
        try:
            info.update({
                "explored_map": self.semantic_map.get_explored_map(is_local),
                "obstacle_map": self.semantic_map.get_obstacle_map(is_local),
                "semantic_map_1D": self.semantic_map.get_semantic_map_1D(is_local),
                "frontier_map": self.semantic_map.get_frontier_map(is_local),
                "been_close_map": self.semantic_map.get_been_close_map(is_local),
                "visited_map": self.semantic_map.get_visited_map(is_local),
                "global_pose": self.semantic_map.global_pose,
                "lmb": self.semantic_map.lmb,
            })
        except Exception as e:
            if self.verbose:
                print(f"[VIS_INFO] Warning: {e}")
        
        info.update(vis_inputs)
        return info
    
    def _reset_vis_dir(self, scene_id, episode_id, current_task_idx):
        """From GoatAgent._reset_vis_dir - EXACT COPY"""
        self.planner.set_vis_dir(scene_id, f"{episode_id}_{current_task_idx}")
        if MATCHING_AVAILABLE and self.matching:  # ADDED safety
            self.matching.set_vis_dir(f"{scene_id}_{episode_id}_{current_task_idx}")
        self.semantic_map.vis_dir = self.planner.vis_dir
        self.semantic_map_module.vis_dir = self.planner.vis_dir
    
    def _score_thresh(self):
        """From GoatAgent._score_thresh - EXACT COPY"""
        task_type = self.current_task["type"]
        if task_type == "languagenav":
            return self.goal_policy_config.score_thresh_lang
        elif task_type == "imagenav":
            return self.goal_policy_config.score_thresh_image
        else:
            return 0.0
    
    # ========================================================================
    # NEW: Real-world specific methods
    # ========================================================================
    
    def _check_goal_reached(self) -> bool:
        """Check if within 0.75m of goal (real-world specific)"""
        # Skip during panorama
        if self.get_subtask_timestep() <= self.panorama_start_steps:
            return False
        
        # ObjectNav: check proximity to category
        if self.current_task["type"] == "objectnav":
            return self._check_close_to_category(self.current_task["semantic_id"])
        
        # ImageNav/LanguageNav: check proximity to instance
        if self.inst_goal_found and self.inst_goal_id is not None:
            return self._check_close_to_instance(self.inst_goal_id)
        
        return False
    
    def _check_close_to_category(self, category_id: int) -> bool:
        """Check if within 0.75m of target category"""
        if self.verbose:
            print(f"[GOAL_CHECK] Category {category_id}")
        
        # Get robot position in GLOBAL map
        robot_pose = self.semantic_map.global_pose
        robot_x = int(robot_pose[0].item())
        robot_y = int(robot_pose[1].item())
        
        # Get category map from GLOBAL map
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
        radius_cells = int(75 / self.config.AGENT.SEMANTIC_MAP.map_resolution)  # 15 cells at 5cm
        
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx*dx + dy*dy > radius_cells*radius_cells:
                    continue
                
                y = robot_y + dy
                x = robot_x + dx
                
                if 0 <= y < category_map.shape[0] and 0 <= x < category_map.shape[1]:
                    if category_map[y, x] > 0:
                        dist_cm = np.sqrt(dx*dx + dy*dy) * self.config.AGENT.SEMANTIC_MAP.map_resolution
                        if self.verbose:
                            print(f"[GOAL_CHECK] ✓ Found at {dist_cm:.1f}cm")
                        return True
        
        return False
    
    def _check_close_to_instance(self, instance_id: int) -> bool:
        """Check if within 0.75m of target instance"""
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
            
            dist_cells = np.sqrt((robot_x - inst_x)**2 + (robot_y - inst_y)**2)
            dist_cm = dist_cells * self.config.AGENT.SEMANTIC_MAP.map_resolution
            
            if dist_cm <= 75:  # 0.75m
                if self.verbose:
                    print(f"[GOAL_CHECK] ✓ Instance {instance_id} at {dist_cm:.1f}cm")
                return True
        
        return False