# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Scout GOAT Environment - Standalone real-world environment for Scout robot.
Does NOT inherit from HabitatGoatEnv to avoid Habitat dependencies.
"""

import json
import cv2
import numpy as np
import rospy
from pathlib import Path
from typing import Any, Dict, List, Optional

import home_robot
from home_robot.core.interfaces import DiscreteNavigationAction, Observations, ContinuousNavigationAction
from home_robot.perception.constants import GoatCategories
from home_robot.utils.geometry import xyt2sophus
from home_robot.utils.logger import get_logger
from home_robot.utils.camera_params import CameraParams
from home_robot_hw.env.visualizer import Visualizer
from home_robot.utils.constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
)

from home_robot_hw.remote import ScoutClient

logger = get_logger()




class ScoutGoatEnv:
    """Standalone GOAT environment for Scout robot."""
    
    def __init__(self, config, task_config_file):
        self.verbose = getattr(config, 'VERBOSE', False)
        self.verbose = True
        
        if self.verbose:
            print(f"\n{'='*60}\n[SCOUT_ENV] Initializing ScoutGoatEnv\n{'='*60}")
        
        self.config = config
        self.visualization_level = getattr(config, 'VISUALIZATION_LEVEL', 0)
        self.ground_truth_semantics = False
        self.task_type = getattr(config, 'TASK_TYPE', 'Goat-v1')
        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth
        self._camera_params: CameraParams = None
        
        if self.verbose:
            print(f"[SCOUT_ENV] Depth range: {self.min_depth}m - {self.max_depth}m")
        
        self.task_config_file = task_config_file
        self._load_task_config()
        
        if self.verbose:
            print(f"[SCOUT_ENV] Loaded {len(self.episodes)} episodes")
            print(f"[SCOUT_ENV] Vocabulary: {len(self.vocabulary)} categories")
        
        self.semantic_category_mapping = GoatCategories(self.vocabulary)
        
        
        
        # Setup visualizer - FIX: only pass config
        if self.visualization_level > 0:
            self.visualizer = Visualizer(config, self.semantic_category_mapping)
        
        if self.verbose:
            print(f"[SCOUT_ENV] Connecting to ScoutClient...")
        
        self.robot = ScoutClient(camera_overrides={"depth_buffer_size": 3})
        
        if self.verbose:
            print(f"[SCOUT_ENV] ✓ Connected to robot")
        
        self.current_episode = None
        self.current_task_idx = 0
        self.episode_over = False
        self.timestep = 0
        self.episode_id = -1
        self.scene_id = "real_world"
        self._episode_start_pose = None
        self._last_obs = None
        
        self.forward_step = config.ENVIRONMENT.forward_step_size
        self.turn_angle = np.radians(config.ENVIRONMENT.turn_angle)
        
        if self.verbose:
            print(f"[SCOUT_ENV] Forward step: {self.forward_step}m")
            print(f"[SCOUT_ENV] Turn angle: {np.degrees(self.turn_angle)}°")
            print(f"[SCOUT_ENV] ✓ Initialization complete\n{'='*60}\n")
    
    def _load_task_config(self):
        if self.verbose:
            print(f"[SCOUT_ENV] Loading task config: {self.task_config_file}")
        
        config_path = Path(self.task_config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Task config not found: {self.task_config_file}")
        
        with open(config_path, 'r') as f:
            task_data = json.load(f)
        
        self.episodes = task_data.get("episodes", [])
        if not self.episodes:
            raise ValueError("No episodes defined in task config")

        vocab_set = set() 
        for episode in self.episodes:
            for task in episode["tasks"]:
                if "category" in task:
                    vocab_set.add(task["category"])
        self.vocabulary = sorted(list(vocab_set))
    
    def reset(self):
        if self.verbose:
            print(f"\n[SCOUT_ENV] ========== RESET ==========")
        
        self.episode_id += 1
        if self.verbose:
            print(f"[SCOUT_ENV] Advancing to episode {self.episode_id}")
        if self.episode_id >= len(self.episodes):
            if self.verbose:
                print(f"[SCOUT_ENV] No more episodes (total: {len(self.episodes)})")
            raise RuntimeError("No more episodes available")

        if not self.episodes:
            raise ValueError("No episodes available")
        self.current_episode = self.episodes[self.episode_id]
        
        if self.current_episode is None:
            self.current_episode = self.episodes[0]
        
        self.current_task_idx = 0
        self.episode_over = False
        self.timestep = 0
        self._last_obs = None
        self._episode_start_pose = xyt2sophus(self.robot.get_base_pose())
        
        if self.verbose:
            start_pose = self.robot.get_base_pose()
            print(f"[SCOUT_ENV] Start pose: x={start_pose[0]:.2f}, y={start_pose[1]:.2f}, θ={np.degrees(start_pose[2]):.1f}°")
        
        if self.visualizer is not None:
            self.visualizer.reset()
        
        self.scene_id = "real_world"
        
        if self.verbose:
            print(f"[SCOUT_ENV] Episode {self.episode_id}: {self.current_episode.get('description', 'GOAT Episode')}")
            print(f"[SCOUT_ENV] Tasks ({len(self.current_episode['tasks'])}):")
            print(f"[SCOUT_ENV] ================================\n")
    
    
    def reset_vis_dir(self):
        if self.visualization_level > 0:
            dir_name = f"{self.scene_id}_{self.episode_id}"
            if self.config.SEQ:
                dir_name = f"{dir_name}_{self.current_task_idx}"
            self.visualizer.set_vis_dir(
                dir_name
            )
    
    def get_observation(self) -> Observations:
        if self._last_obs is not None:
            return self._last_obs

        if self.verbose:
            print(f"\n[SCOUT_ENV] ----- get_observation (step {self.timestep}) -----")
        
        if self.current_episode is None:
            raise RuntimeError("No episode loaded. Call reset() first.")
        
        if self.verbose:
            print(f"[SCOUT_ENV] Getting RGB-D from robot...")
        
        rospy.sleep(0.2)
        rgb, depth, _ = self.robot.get_images(compute_xyz=True, rotate_images=False)
        
        if self.verbose:
            print(f"[SCOUT_ENV] RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
            print(f"[SCOUT_ENV] Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]m")
        
        current_pose = xyt2sophus(self.robot.get_base_pose())
        relative_pose = self._episode_start_pose.inverse() * current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]
        gps = relative_pose.translation()[:2]
        
        if self.verbose:
            print(f"[SCOUT_ENV] Relative pose: x={gps[0]:.3f}, y={gps[1]:.3f}, θ={np.degrees(theta):.1f}°")
        
        depth = self._preprocess_depth(depth)
        rgb = self._preprocess_rgb(rgb)
        if self.verbose:
            print(f"[SCOUT_ENV] RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
            print(f"[SCOUT_ENV] Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]m")
        
        tasks = self._preprocess_goals(self.current_episode["tasks"])
        
        if self.verbose:
            current_task = tasks[self.current_task_idx]
            print(f"[SCOUT_ENV] Current task: {current_task['type']} -> '{current_task['category']}' (sem_id={current_task['semantic_id']})")
        
        obs = home_robot.core.interfaces.Observations(
            rgb=rgb.copy(),
            depth=depth.copy(),
            gps=gps,
            compass=np.array([theta]),
            task_observations={"tasks": tasks},
            camera_pose=None,
            third_person_image=None,
        )
        if self.verbose:
            print(f"[SCOUT_ENV] Running Detic perception...")
        
        
        self._last_obs = obs
        
        if self.verbose:
            print(f"[SCOUT_ENV] ----- observation complete -----\n")
        
        return obs
    
    def _preprocess_rgb(self, rgb: np.ndarray) -> np.ndarray:
        # rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR) 
        return rgb

    def _preprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        # Neighbor consistency filter: reject isolated valid pixels
        valid = (depth > self.min_depth) & (depth < self.max_depth) & ~np.isnan(depth) & ~np.isinf(depth)
        kernel = np.ones((5, 5), dtype=np.float32)
        neighbor_count = cv2.filter2D(valid.astype(np.float32), -1, kernel)
        # Reject pixels where fewer than 40% of neighbors are valid
        isolated = valid & (neighbor_count < 0.4 * kernel.size)
        depth[~valid] = MAX_DEPTH_REPLACEMENT_VALUE
        depth[isolated] = MAX_DEPTH_REPLACEMENT_VALUE
        return depth
    
    
    
    def _preprocess_goals(self, goals):
        for goal_v in goals:
            goal_v["semantic_id"] = self.semantic_category_mapping.goal_name_to_cat_id[
                "_".join(goal_v["category"].split(" "))
            ]
            if goal_v.get("image") is not None:
                goal_v["type"] = "imagenav"
            elif goal_v.get("description") is not None:
                goal_v["type"] = "languagenav"
            else:
                goal_v["type"] = "objectnav"

            if goal_v["type"] == "imagenav":
                try:
                    if not isinstance(goal_v["image"], str):
                        continue
                    image_path = Path(goal_v["image"])
                    if image_path.exists():
                        img = cv2.imread(str(image_path))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        goal_v["image"] = img
                    else:
                        del goal_v["image"]
                except Exception as e:
                    logger.warning(f"Error occurred while processing image: {e}")
                    
        return goals
 
    def apply_action(self, action: Any, info: Optional[Dict[str, Any]] = None, prev_obs: Optional[Observations] = None) -> bool:
        self.timestep += 1
        if info is not None:
            self._process_info(info)
        self._last_obs = None
        action_enum = self._preprocess_action(action)
        if self.verbose:
            print(f"\n[SCOUT_ENV] ----- apply_action -----")
        
        
        if self.verbose:
            print(f"[SCOUT_ENV] Action: {action_enum.name if hasattr(action_enum, 'name') else action_enum}")
        
        
        
        if action_enum == DiscreteNavigationAction.STOP:
            if self.verbose:
                print(f"[SCOUT_ENV] STOP received - Task {self.current_task_idx + 1} complete")
            self.current_task_idx += 1
            if self.current_task_idx >= len(self.current_episode["tasks"]):
                if self.verbose:
                    print(f"[SCOUT_ENV] ✓ All tasks complete - Episode over")
                self.episode_over = True
            else:
                if self.verbose:
                    next_task = self.current_episode["tasks"][self.current_task_idx]
                    print(f"[SCOUT_ENV] Starting task {self.current_task_idx + 1}: {next_task.get('category', 'unknown')}")
                # self.reset_vis_dir()
        
        continuous_action = np.zeros(3)
        if action_enum == DiscreteNavigationAction.MOVE_FORWARD:
            continuous_action[0] = self.forward_step
            if self.verbose:
                print(f"[SCOUT_ENV] FORWARD: {self.forward_step}m")
        elif action_enum == DiscreteNavigationAction.TURN_RIGHT:
            continuous_action[2] = -self.turn_angle
            if self.verbose:
                print(f"[SCOUT_ENV] TURN RIGHT: {np.degrees(self.turn_angle):.1f}°")
        elif action_enum == DiscreteNavigationAction.TURN_LEFT:
            continuous_action[2] = self.turn_angle
            if self.verbose:
                print(f"[SCOUT_ENV] TURN LEFT: {np.degrees(self.turn_angle):.1f}°")
        elif action_enum is None:
            print(f"[SCOUT_ENV] Too close to obstacles. moving backward to avoid collision")
            continuous_action[0] = -self.forward_step / 2
        elif isinstance(action_enum, ContinuousNavigationAction):
            continuous_action[2] = np.radians(action_enum.xyt[2].item())
            if self.verbose:
                print(f"[SCOUT_ENV] Continuous TURN : {np.degrees(continuous_action[2]):.1f}°")

        if np.any(continuous_action != 0):
            try:
                if self.verbose:
                    print(f"[SCOUT_ENV] Sending to robot: {continuous_action}")
                self.robot.nav.navigate_to(continuous_action, relative=True, blocking=True)
                # input("Press a button")

            except Exception as e:
                logger.error(f"Navigation failed: {e}")
                if self.verbose:
                    print(f"[SCOUT_ENV] Navigation error: {e}")
        
        
        if self.verbose:
            new_pose = self.robot.get_base_pose()
            print(f"[SCOUT_ENV] New pose: x={new_pose[0]:.2f}, y={new_pose[1]:.2f}, θ={np.degrees(new_pose[2]):.1f}°")
            print(f"[SCOUT_ENV] ----- action complete (timestep={self.timestep}) -----\n")
        
    
    def add_subepisode_metrics(self, all_metrics: Dict, action: Any) -> None:
        task_idx = action["action_args"]["task_idx"]
        all_metrics[task_idx] = {
            "timesteps": self.timestep,
            "success": True,
        }
        if self.verbose:
            print(f"[SCOUT_ENV] Metrics for task {task_idx}: {all_metrics[task_idx]}")
        logger.info(f"{self.scene_id}_{self.episode_id}_{task_idx} complete")
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "tasks_completed": self.current_task_idx,
            "total_tasks": len(self.current_episode["tasks"]) if self.current_episode else 0,
            "timesteps": self.timestep,
        }
    
    def get_robot(self):
        return self.robot
    
    @property
    def episode(self):
        class EpisodeWrapper:
            def __init__(self, env):
                self.episode_id = str(env.episode_id)
        return EpisodeWrapper(self)

    def _preprocess_action(self, action) -> int:
        action_enum = action["action"]
        return action_enum

    def _process_info(self, info: Dict[str, Any]) -> Any:
        if self.visualization_level > 0:
            self.visualizer.visualize(**info)

    def sync_camera_params(self) -> CameraParams:
        info = self.robot._ros_client.dpt_cam.get_info()
        self._camera_params = CameraParams.from_ros_camera(
            camera_info=info,
            camera_height=self.config.ENVIRONMENT.camera_height,  # mount height stays in config
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )
        return self._camera_params

    @property
    def camera_params(self) -> CameraParams:
        assert self._camera_params is not None, "Call sync_camera_params() first"
        return self._camera_params
