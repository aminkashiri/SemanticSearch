# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified GOAT Agent - Works for both Habitat simulation and real-world Scout robot.
The mode is determined by config.REAL_WORLD flag.
"""

import os
import cv2
import torch
import psutil
import numpy as np
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

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
from .goat_matching import GoatMatching

logger = get_logger()


@dataclass
class Task:
    """Task dataclass for navigation goals."""
    type: str
    goal_semantic_id: int
    goal_image: np.ndarray = None
    goal_image_processed: np.ndarray = None
    goal_image_keypoints: np.ndarray = None
    goal_description: str = None


class GoatAgent(Agent):
    """
    Unified GOAT agent for both simulation and real-world navigation.
    
    Mode is determined by config.REAL_WORLD:
    - False (default): Habitat simulation mode
    - True: Real-world Scout robot mode
    
    Key differences handled internally:
    - Camera parameters source (habitat.simulator vs ENVIRONMENT)
    - Perception (built-in Detic/YOLO vs external)
    - Goal proximity checking (real-world only)
    """

    def __init__(self, config, vocabulary, agent_id=None, device_id: int = 0):
        """
        Initialize GoatAgent.
        
        Args:
            config: Configuration object
            vocabulary: List of semantic category names
            agent_id: Optional agent identifier for multi-agent scenarios
            device_id: CUDA device ID
        """
        # Determine mode
        self.real_world = getattr(config, 'REAL_WORLD', False)
        self.verbose = getattr(config, 'VERBOSE', False)
        
        if self.verbose:
            mode = "REAL WORLD" if self.real_world else "SIMULATION"
            print(f"\n{'='*60}\n[INIT] GoatAgent ({mode})\n{'='*60}")
        
        # Agent identification
        self.is_multiagent = agent_id is not None
        self.agent_id = agent_id
        self.log = get_logger(agent_id=agent_id)
        
        # Configuration
        self.config = config
        self.max_steps = config.AGENT.max_steps
        self.task_type = self._get_task_type(config)
        self.seq_goals = bool(getattr(config, 'SEQ', False))
        self.use_yolo = bool(getattr(config, 'USE_YOLO', False)) and not self.real_world
        self.record_instance_ids = True
        self.visualization_level = getattr(config, 'VISUALIZATION_LEVEL', 0)
        
        # Get max subtasks
        if hasattr(config, 'ENVIRONMENT') and hasattr(config.ENVIRONMENT, 'max_subtasks_per_episode'):
            self.max_subtasks_per_episode = config.ENVIRONMENT.max_subtasks_per_episode
        elif hasattr(config, 'ENVIRONMENT') and hasattr(config.ENVIRONMENT, 'max_num_sub_task_episodes'):
            self.max_subtasks_per_episode = config.ENVIRONMENT.max_num_sub_task_episodes
        else:
            self.max_subtasks_per_episode = 10
        
        # Device setup
        if config.NO_GPU:
            self.device = torch.device("cpu")
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")
        
        self.num_sem_categories = len(vocabulary)
        
        if self.verbose:
            print(f"[INIT] Device: {self.device}, Categories: {self.num_sem_categories}")
        
        # Instance memory
        self.instance_memory = InstanceMemory(
            config=config,
            mask_cropped_instances=False,
            padding_cropped_instances=200,
        ) if self.record_instance_ids else None
        
        # Goal matching
        self.matching = GoatMatching(
            device=device_id if not config.NO_GPU else 0,
            config=config.AGENT.SUPERGLUE,
            default_vis_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
            print_images=self.visualization_level > 1,
            instance_memory=self.instance_memory,
            logger=self.log,
            cat_match_threshold=getattr(config.AGENT, 'cat_match_threshold', 0.5),
        )
        
        # Get camera parameters based on mode
        camera_params = self._get_camera_params(config)
        
        # Semantic map module
        agent_cell_radius = int(
            np.ceil(config.AGENT.radius * 100.0 / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        
        self.semantic_map_module = Categorical2DSemanticMapModule(
            device=self.device,
            frame_height=camera_params['height'],
            frame_width=camera_params['width'],
            camera_height=camera_params['camera_height'],
            hfov=camera_params['hfov'],
            num_sem_categories=self.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            max_depth=camera_params['max_depth'],
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
                camera_params['max_depth']
                if config.AGENT.exploration_type == "raycast"
                else 3
            ),
            agent_cell_radius=agent_cell_radius,
            print_images=self.visualization_level > 2
        )
        
        self.inst_goal_id = None
        
        # Semantic map state
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
        
        # Panorama setup
        turn_angle = camera_params['turn_angle']
        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 / turn_angle)
        else:
            self.panorama_start_steps = 0
        
        if self.verbose:
            print(f"[INIT] Panorama: {self.panorama_start_steps} steps")
        
        # Planner
        self.planner = DiscretePlanner(
            turn_angle=turn_angle,
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
            ground_truth_semantics=getattr(config, 'GROUND_TRUTH_SEMANTICS', False) and not self.real_world,
            task_type=self.task_type
        )
        
        # Episode state
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
        
        # Ground truth semantics (only for simulation)
        self.ground_truth_semantics = getattr(config, 'GROUND_TRUTH_SEMANTICS', False) and not self.real_world
        
        # Setup perception for simulation mode
        self.segmentation = None
        self.yolo = None
        if not self.real_world and not self.ground_truth_semantics:
            self._setup_perception(config, vocabulary)
        
        # Task storage
        self.tasks: List[Task] = []
        self._curr_obs = None
        self.pose_delta = None
        self.frame_yolo = None
        
        if self.verbose:
            print(f"[INIT] Complete\n")
    
    # =========================================================================
    # Configuration Helpers
    # =========================================================================
    
    def _get_task_type(self, config) -> str:
        """Get task type from config."""
        if hasattr(config, 'TASK_TYPE'):
            return config.TASK_TYPE
        if hasattr(config, 'habitat') and hasattr(config.habitat, 'task'):
            return config.habitat.task.type
        return 'Goat-v1'
    
    def _get_camera_params(self, config) -> dict:
        """Get camera parameters based on mode (sim vs real)."""
        if self.real_world:
            # Real world: use ENVIRONMENT config
            return {
                'height': config.ENVIRONMENT.frame_height,
                'width': config.ENVIRONMENT.frame_width,
                'camera_height': config.ENVIRONMENT.camera_height,
                'hfov': config.ENVIRONMENT.hfov,
                'max_depth': config.ENVIRONMENT.max_depth,
                'turn_angle': config.ENVIRONMENT.turn_angle,
            }
        else:
            # Simulation: use habitat.simulator config
            camera = config.habitat.simulator.agents.agent0.sim_sensors.depth_sensor
            return {
                'height': camera.height,
                'width': camera.width,
                'camera_height': camera.position[1],
                'hfov': camera.hfov,
                'max_depth': camera.max_depth,
                'turn_angle': config.habitat.simulator.turn_angle,
            }
    
    def _setup_perception(self, config, vocabulary):
        """Setup perception models for simulation mode."""
        if "Goat-v1" in self.task_type:
            from home_robot.perception.detection.detic.detic_perception import DeticPerception
            self.segmentation = DeticPerception(
                vocabulary="custom",
                custom_vocabulary="," + ",".join(vocabulary),
                sem_gpu_id=(-1 if config.NO_GPU else 0),
            )
            if self.use_yolo:
                from ultralytics import YOLOWorld
                self.yolo = YOLOWorld('yolov8s-worldv2.pt')
                self.yolo.set_classes(vocabulary)
        else:
            from home_robot.perception.detection.maskrcnn.maskrcnn_perception import MaskRCNNPerception
            self.segmentation = MaskRCNNPerception(
                sem_pred_prob_thr=0.8,
                sem_gpu_id=(-1 if config.NO_GPU else 0),
            )
            if self.use_yolo:
                from ultralytics import YOLOv10
                self.yolo = YOLOv10.from_pretrained('jameslahm/yolov10n', verbose=False)
    
    # =========================================================================
    # Core Methods
    # =========================================================================
    
    def get_subtask_timestep(self) -> int:
        """Get current subtask timestep."""
        if self.seq_goals:
            return self.subtask_timesteps[self.current_task_idx]
        return self.total_timesteps
    
    def reset(self, scene_id, episode_id, current_task_idx=0):
        """Initialize agent state for new episode."""
        if self.verbose:
            print(f"\n[RESET] Episode {episode_id}")
        
        self.total_timesteps = 0
        if self.seq_goals:
            self.subtask_timesteps = [0] * self.max_subtasks_per_episode
            self.sub_task_timesteps = self.subtask_timesteps
        self.last_pose = np.zeros(3)
        
        self.semantic_map.init_map_and_pose()
        
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
        """Handle stop action."""
        self.reset_for_next_task()
    
    def reset_for_next_task(self) -> None:
        """Reset for a new task within same episode."""
        self.inst_goal_id = None
        if self.seq_goals:
            self.current_task_idx += 1
        self.navigate_to_best = False
        self.match_memory = True
        self.planner.reset_for_next_task()
    
    def _update_steps(self):
        """Update timestep counters."""
        self.total_timesteps += 1
        if self.seq_goals:
            self.subtask_timesteps[self.current_task_idx] += 1
            self.sub_task_timesteps = self.subtask_timesteps
        self.matching.step = self.total_timesteps
        self.semantic_map_module.timestep = self.get_subtask_timestep()
        self.planner.total_timesteps = self.total_timesteps
        self.planner.timestep = self.get_subtask_timestep()
        self.log.info(f"---------------- Updating state - step:{self.get_subtask_timestep()} ----------------")
        self.log.debug(f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")
    
    def update_state(self, obs: Observations):
        """Update agent state with new observation."""
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
        """Generate action based on current state."""
        if self.verbose:
            print(f"\n[ACT] Step {self.get_subtask_timestep()}")
        
        # Check stuck/timeout
        stuck = False
        if self.get_subtask_timestep() >= self.max_steps or self.stuck_counter > 30:
            self.log.warning("Reached max steps or stuck, calling STOP")
            stuck = True
        
        # Real-world: Check goal proximity
        if self.real_world and not stuck and self._check_goal_reached():
            if self.verbose:
                print("[ACT] ✓✓✓ GOAL REACHED ✓✓✓")
            return DiscreteNavigationAction.STOP, {}, False
        
        # Get action from planner
        action, vis_inputs = self._get_best_action(**kwargs)
        action_dict = self._process_action(action)
        info = self._get_vis_info(vis_inputs, action_dict)
        
        if self.verbose:
            print(f"[ACT] Action: {action}\n")
        
        if action_dict["action"] == DiscreteNavigationAction.STOP:
            self.handle_stop(action_dict)
        
        # Return format depends on mode
        if self.real_world:
            return action_dict["action"], info if info else {}, stuck
        else:
            return action_dict, info, stuck
    
    # =========================================================================
    # Map Update
    # =========================================================================
    
    def _update_pose(self):
        """Update pose from observations."""
        obs = self._curr_obs
        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        self.pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_pose), device=self.device
        )
        self.last_pose = curr_pose
        
        if torch.norm(self.pose_delta[:2]).item() < 0.05:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
    
    def _update_maps(self):
        """Update semantic maps with current observation."""
        obs_preprocessed, instance_scores, category_scores = self._preprocess_obs(self._curr_obs)
        
        if self.verbose:
            print(f"[UPDATE_MAPS] obs shape: {obs_preprocessed.shape}")
            print(f"[UPDATE_MAPS] pose_delta: {self.pose_delta.cpu().numpy()}")
        
        self.semantic_map_module(
            obs_preprocessed,
            self.pose_delta,
            self.semantic_map,
            instance_scores,
            category_scores
        )
    
    def _preprocess_tasks(self, tasks_obs) -> List[Task]:
        """Preprocess task observations into Task objects."""
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
        """Preprocess observations for map update."""
        from scipy.ndimage import binary_erosion
        
        category_scores = None
        
        # Run perception if in simulation mode with non-GT semantics
        if not self.real_world and not self.ground_truth_semantics and self.segmentation is not None:
            obs = self._run_perception(obs)
        
        if self.verbose:
            print(f"[PREPROCESS] Depth: min={obs.depth.min():.3f}, max={obs.depth.max():.3f}")
        
        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0  # m to cm
        
        # One-hot encode semantics
        semantic = torch.eye(self.num_sem_categories + 1, device=self.device)[
            torch.from_numpy(obs.semantic).to(self.device).long()
        ][:, :, 1:]  # Remove background class
        
        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)
        
        # Instance tracking
        inst_scores = None
        if self.record_instance_ids and "instance_frame" in obs.task_observations:
            instance_frame = obs.task_observations["instance_frame"]
            unique_ids, new_instance_frame = np.unique(instance_frame, return_inverse=True)
            new_instance_frame = new_instance_frame.reshape(instance_frame.shape)
            new_instance_frame = torch.from_numpy(new_instance_frame).to(self.device).long()
            
            # One-hot encode
            instance_frame_onehot = torch.eye(len(unique_ids), device=self.device)[new_instance_frame]
            
            has_background = len(unique_ids) > 0 and unique_ids[0] == 0
            if has_background:
                instance_frame_onehot = instance_frame_onehot[:, :, 1:]
            
            # Get instance scores
            num_instances = len(unique_ids) - 1 if has_background else len(unique_ids)
            if "instance_scores" in obs.task_observations and num_instances > 0:
                raw_scores = obs.task_observations["instance_scores"]
                if len(raw_scores) >= num_instances:
                    inst_scores = np.array(raw_scores[:num_instances])
                else:
                    inst_scores = np.ones(num_instances)
                    inst_scores[:len(raw_scores)] = raw_scores
            else:
                inst_scores = np.ones(max(0, num_instances))
            
            obs_preprocessed = torch.cat([obs_preprocessed, instance_frame_onehot], dim=-1)
        
        obs_preprocessed = obs_preprocessed.permute(2, 0, 1)
        
        # Process tasks
        self.tasks = self._preprocess_tasks(obs.task_observations["tasks"])
        
        return obs_preprocessed, inst_scores, category_scores
    
    def _run_perception(self, obs: Observations) -> Observations:
        """Run perception models (simulation mode only)."""
        from home_robot.perception.constants import coco_categories_mapping
        
        if "Goat-v1" in self.task_type:
            obs = self.segmentation.predict(obs, draw_instance_predictions=True)
        else:
            obs = self.segmentation.predict(obs)
        
        # Filter instances by depth
        self._filter_instances_by_depth(obs)
        
        # Run YOLO if enabled
        if self.use_yolo and self.yolo is not None:
            yolo_output = self.yolo(source=obs.rgb, conf=0.2, verbose=False)
            category_scores = {i: [] for i in range(self.num_sem_categories + 1)}
            for box in yolo_output[0].boxes:
                cls = int(box.cls[0])
                if "Goat-v1" in self.task_type:
                    category_scores[cls + 1].append(box.conf[0].item())
                else:
                    category_scores[coco_categories_mapping[cls] + 1].append(box.conf[0].item())
            self.frame_yolo = cv2.cvtColor(yolo_output[0].plot(), cv2.COLOR_BGR2RGB)
            
            for i in range(self.num_sem_categories + 1):
                category_scores[i] = np.max(category_scores[i]) if category_scores[i] else 0
        
        return obs
    
    def _filter_instances_by_depth(self, obs, erosion_iters=1, mad_thresh=3.0):
        """Filter instances by depth consistency."""
        from scipy.ndimage import binary_erosion
        
        if "instance_frame" not in obs.task_observations:
            return
        
        instance_frame = obs.task_observations["instance_frame"]
        depth_frame = obs.depth
        mask_union = np.zeros_like(instance_frame, dtype=bool)
        
        instance_ids = np.unique(instance_frame)
        instance_ids = instance_ids[instance_ids > 0]
        
        for inst_id in instance_ids:
            inst_mask = instance_frame == inst_id
            inst_mask = binary_erosion(inst_mask, iterations=erosion_iters)
            if inst_mask.sum() == 0:
                inst_mask = instance_frame == inst_id
            
            instance_depth = depth_frame[inst_mask]
            median = np.median(instance_depth)
            mad = np.median(np.abs(instance_depth - median))
            
            if mad == 0:
                depth_mask = np.abs(depth_frame - median) < 1e-3
            else:
                depth_mask = np.abs(depth_frame - median) <= mad_thresh * mad
            
            final_inst_mask = inst_mask & depth_mask
            mask_union = mask_union | final_inst_mask
        
        obs.semantic = obs.semantic * mask_union
        obs.task_observations["instance_frame"] = obs.task_observations["instance_frame"] * mask_union
    
    # =========================================================================
    # Goal Search
    # =========================================================================
    
    @torch.no_grad()
    def _search_for_goal(self, select_best=False):
        """Search for goal in current observation and memory."""
        if self.inst_goal_id is not None and self.get_subtask_timestep() % 10 != 0:
            self.log.info("Already found instance goal, not searching anymore.")
        else:
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
        """Get best action from planner."""
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
        
        self.log.info("No reachable frontiers, forcing a match against memory")
        self.match_memory = True
        self._search_for_goal(select_best=True)
        
        if self.inst_goal_id is None:
            self.log.info("No match found in memory, stopping.")
            return DiscreteNavigationAction.STOP, {}
        
        return self._get_best_action(**kwargs)
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def _get_vis_info(self, vis_inputs, action):
        """Get visualization info dict."""
        if self.visualization_level < 1:
            return None
        
        if vis_inputs is None:
            vis_inputs = {}
        
        is_local = vis_inputs.get("is_local", True)
        obs = self._curr_obs
        
        info = {
            "agent_id": self.agent_id,
            "rgb_frame": self.frame_yolo if self.use_yolo and self.frame_yolo is not None else (obs.rgb[:, :, ::-1] if obs else None),
            "depth_frame": obs.depth if obs else None,
            "semantic_frame": obs.task_observations.get("semantic_frame", obs.semantic) if obs else None,
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
        """Add task-specific info to visualization dict."""
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
        """Reset visualization directory."""
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
        """Wrap action in dict format."""
        return {
            "action": action,
            "action_args": {
                "agent_id": 0 if self.agent_id is None else self.agent_id,
                "task_idx": self.current_task_idx,
            },
        }
    
    # =========================================================================
    # Real-World Specific Methods
    # =========================================================================
    
    def _check_goal_reached(self) -> bool:
        """Check if robot is within 0.75m of goal (real-world only)."""
        if not self.real_world:
            return False
        
        if self.get_subtask_timestep() <= self.panorama_start_steps:
            return False
        
        if not self.tasks or self.current_task_idx >= len(self.tasks):
            return False
        
        task = self.tasks[self.current_task_idx]
        
        if task.type == "objectnav":
            return self._check_close_to_category(task.goal_semantic_id)
        
        if self.inst_goal_id is not None:
            return self._check_close_to_instance(self.inst_goal_id)
        
        return False
    
    def _check_close_to_category(self, category_id: int) -> bool:
        """Check if robot is within 0.75m of target category."""
        robot_pose = self.semantic_map.global_pose
        robot_x = int(robot_pose[0].item())
        robot_y = int(robot_pose[1].item())
        
        semantic_map = self.semantic_map.global_map[0, 4:4+self.num_sem_categories]
        
        if category_id >= semantic_map.shape[0]:
            return False
        
        category_map = semantic_map[category_id]
        if (category_map > 0).sum().item() == 0:
            return False
        
        map_resolution = self.config.AGENT.SEMANTIC_MAP.map_resolution
        radius_cells = int(75 / map_resolution)
        
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx*dx + dy*dy > radius_cells*radius_cells:
                    continue
                y, x = robot_y + dy, robot_x + dx
                if 0 <= y < category_map.shape[0] and 0 <= x < category_map.shape[1]:
                    if category_map[y, x] > 0:
                        return True
        return False
    
    def _check_close_to_instance(self, instance_id: int) -> bool:
        """Check if robot is within 0.75m of target instance."""
        if self.instance_memory is None or not self.instance_memory.instance_exists(instance_id):
            return False
        
        instance = self.instance_memory.get_instance(instance_id)
        robot_pose = self.semantic_map.global_pose
        
        if hasattr(instance, 'map_locs') and len(instance.map_locs) > 0:
            locs = torch.stack([loc for loc in instance.map_locs])
            inst_x, inst_y = locs[:, 0].mean().item(), locs[:, 1].mean().item()
            robot_x, robot_y = robot_pose[0].item(), robot_pose[1].item()
            
            map_resolution = self.config.AGENT.SEMANTIC_MAP.map_resolution
            dist_cm = np.sqrt((robot_x - inst_x)**2 + (robot_y - inst_y)**2) * map_resolution
            
            if dist_cm <= 75:
                return True
        return False