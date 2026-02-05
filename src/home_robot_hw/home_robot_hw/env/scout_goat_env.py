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
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.perception.constants import SemanticCategoryMapping
from home_robot.utils.geometry import xyt2sophus
from home_robot.utils.logger import get_logger

from home_robot_hw.remote import ScoutClient

logger = get_logger()


# =============================================================================
# Semantic Category Mapping for Scout
# =============================================================================

class ScoutSemanticCategoryMapping(SemanticCategoryMapping):
    """Semantic category mapping for Scout real-world environment."""
    
    def __init__(self, vocabulary: List[str]):
        self.vocabulary = ['unknown'] + list(vocabulary)
        self._num_sem_categories = len(vocabulary)
        
        self.goal_name_to_goal_id = {name: idx for idx, name in enumerate(self.vocabulary)}
        self.goal_id_to_goal_name = {idx: name for idx, name in enumerate(self.vocabulary)}
        self.goal_name_to_cat_id = self.goal_name_to_goal_id
        self.cat_id_to_goal_name = self.goal_id_to_goal_name
        
        np.random.seed(42)
        self._map_color_palette = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(self.vocabulary))]
        self._frame_color_palette = self._map_color_palette.copy()
        self._instance_id_to_category_id = np.zeros(10000, dtype=np.int32)
    
    @property
    def num_sem_categories(self) -> int:
        return self._num_sem_categories
    
    @property
    def map_color_palette(self) -> List[tuple]:
        return self._map_color_palette
    
    @property
    def frame_color_palette(self) -> List[tuple]:
        return self._frame_color_palette
    
    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id
    
    @property
    def categories_legend_path(self) -> str:
        return ""
    
    @property
    def map_goal_id(self) -> int:
        return 0
    
    def reset_instance_id_to_category_id(self, env=None):
        self._instance_id_to_category_id.fill(0)
    
    def get_category_id(self, category_name: str) -> int:
        if category_name in self.goal_name_to_goal_id:
            return self.goal_name_to_goal_id[category_name]
        underscore_name = "_".join(category_name.split(" "))
        if underscore_name in self.goal_name_to_goal_id:
            return self.goal_name_to_goal_id[underscore_name]
        for name, idx in self.goal_name_to_goal_id.items():
            if name.lower() == category_name.lower():
                return idx
        logger.warning(f"Category '{category_name}' not found")
        return 0
    
    def get_category_name(self, category_id: int) -> str:
        return self.goal_id_to_goal_name.get(category_id, "unknown")


# =============================================================================
# Visualizer (optional)
# =============================================================================

try:
    from home_robot_hw.env.visualizer import Visualizer
except ImportError:
    logger.warning("Visualizer not available, using dummy")
    class Visualizer:
        def __init__(self, config):
            self.vis_dir = None
        def reset(self):
            pass
        def set_vis_dir(self, scene_id, episode_id):
            self.vis_dir = f"{scene_id}_{episode_id}"
        def visualize(self, **kwargs):
            pass


# =============================================================================
# Scout GOAT Environment
# =============================================================================

class ScoutGoatEnv:
    """Standalone GOAT environment for Scout robot."""
    
    def __init__(self, config, task_config_file: Optional[str] = None, *args, **kwargs):
        self.verbose = getattr(config, 'VERBOSE', False)
        
        if self.verbose:
            print(f"\n{'='*60}\n[SCOUT_ENV] Initializing ScoutGoatEnv\n{'='*60}")
        
        self.config = config
        self.visualization_level = getattr(config, 'VISUALIZATION_LEVEL', 0)
        self.ground_truth_semantics = False
        self.task_type = getattr(config, 'TASK_TYPE', 'Goat-v1')
        
        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth
        
        if self.verbose:
            print(f"[SCOUT_ENV] Depth range: {self.min_depth}m - {self.max_depth}m")
        
        self.task_config_file = task_config_file or "configs/example_tasks.json"
        self._load_task_config()
        
        if self.verbose:
            print(f"[SCOUT_ENV] Loaded {len(self.episodes)} episodes")
            print(f"[SCOUT_ENV] Vocabulary: {len(self.vocabulary)} categories")
        
        self.semantic_category_mapping = ScoutSemanticCategoryMapping(self.vocabulary)
        
        if self.verbose:
            print(f"[SCOUT_ENV] Semantic categories: {self.semantic_category_mapping.num_sem_categories}")
            for cat in list(self.vocabulary)[:5]:
                cat_id = self.semantic_category_mapping.get_category_id(cat)
                print(f"[SCOUT_ENV]   '{cat}' -> ID {cat_id}")
        
        detic_vocab = ",".join(self.vocabulary)
        if self.verbose:
            print(f"[SCOUT_ENV] Initializing Detic perception...")
        
        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary=detic_vocab,
            sem_gpu_id=(0 if not config.NO_GPU else -1),
        )
        
        if self.verbose:
            print(f"[SCOUT_ENV] ✓ Detic initialized")
        
        # Setup visualizer - FIX: only pass config
        if self.visualization_level > 0:
            self.visualizer = Visualizer(config)
            if self.verbose:
                print(f"[SCOUT_ENV] ✓ Visualizer initialized")
        else:
            self.visualizer = None
        
        if self.verbose:
            print(f"[SCOUT_ENV] Connecting to ScoutClient...")
        
        self.robot = ScoutClient()
        
        if self.verbose:
            print(f"[SCOUT_ENV] ✓ Connected to robot")
        
        self.current_episode = None
        self.current_task_idx = 0
        self.episode_over = False
        self.timestep = 0
        self.episode_id = 0
        self.scene_id = "real_world"
        self._episode_start_pose = None
        self._last_obs = None
        
        self.forward_step = getattr(config.ENVIRONMENT, 'forward', 0.25)
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
        
        cat_map_file = getattr(self.config.ENVIRONMENT, 'category_map_file', None)
        
        if cat_map_file and Path(cat_map_file).exists():
            if self.verbose:
                print(f"[SCOUT_ENV] Loading categories from: {cat_map_file}")
            with open(cat_map_file, 'r') as f:
                cat_map = json.load(f)
            if 'categories' in cat_map:
                self.vocabulary = sorted(cat_map['categories'])
            elif 'obj_category_to_obj_category_id' in cat_map:
                vocab_set = set()
                vocab_set.update(cat_map.get('obj_category_to_obj_category_id', {}).keys())
                vocab_set.update(cat_map.get('recep_category_to_recep_category_id', {}).keys())
                self.vocabulary = sorted(list(vocab_set))
            else:
                raise ValueError(f"Unknown category map format in {cat_map_file}")
        else:
            if self.verbose:
                print(f"[SCOUT_ENV] Extracting vocabulary from tasks")
            vocab_set = set()
            for episode in self.episodes:
                for task in episode.get("tasks", []):
                    if "category" in task:
                        vocab_set.add(task["category"])
            self.vocabulary = sorted(list(vocab_set))
    
    def reset(self):
        if self.verbose:
            print(f"\n[SCOUT_ENV] ========== RESET ==========")
        
        if not self.episodes:
            raise ValueError("No episodes available")
        
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
            for i, task in enumerate(self.current_episode['tasks']):
                cat = task.get('category', task.get('description', 'N/A'))
                task_type = task.get('type', 'objectnav')
                sem_id = self.semantic_category_mapping.get_category_id(cat)
                print(f"[SCOUT_ENV]   {i+1}. {task_type.upper()}: '{cat}' (semantic_id={sem_id})")
            print(f"[SCOUT_ENV] ================================\n")
    
    def next_episode(self) -> bool:
        self.episode_id += 1
        if self.verbose:
            print(f"[SCOUT_ENV] Advancing to episode {self.episode_id}")
        if self.episode_id >= len(self.episodes):
            if self.verbose:
                print(f"[SCOUT_ENV] No more episodes (total: {len(self.episodes)})")
            return False
        self.current_episode = self.episodes[self.episode_id]
        self.reset()
        return True
    
    def reset_vis_dir(self):
        if self.visualizer is not None:
            if getattr(self.config, 'SEQ', True):
                episode_id_str = f"{self.episode_id}_{self.current_task_idx}"
            else:
                episode_id_str = str(self.episode_id)
            self.visualizer.set_vis_dir(self.scene_id, episode_id_str)
            if self.verbose:
                print(f"[SCOUT_ENV] Vis dir: {self.scene_id}_{episode_id_str}")
    
    def get_observation(self) -> Observations:
        if self.verbose:
            print(f"\n[SCOUT_ENV] ----- get_observation (step {self.timestep}) -----")
        
        if self.current_episode is None:
            raise RuntimeError("No episode loaded. Call reset() first.")
        
        if self.verbose:
            print(f"[SCOUT_ENV] Getting RGB-D from robot...")
        
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
        
        if self.verbose:
            print(f"[SCOUT_ENV] Depth after preprocess: range: [{depth.min():.3f}, {depth.max():.3f}]m")
            valid_depth = (depth > 0) & (depth < self.max_depth)
            print(f"[SCOUT_ENV] Valid depth pixels: {valid_depth.sum()} / {depth.size} ({100*valid_depth.sum()/depth.size:.1f}%)")
        
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
        
        obs = self.segmentation.predict(obs, depth_threshold=self.max_depth)
        
        if self.verbose:
            unique_sem_ids = np.unique(obs.semantic)
            print(f"[SCOUT_ENV] Detic output: {len(unique_sem_ids)} unique IDs")
            for sem_id in unique_sem_ids[:10]:
                if sem_id > 0:
                    cat_name = self.semantic_category_mapping.get_category_name(int(sem_id))
                    pixel_count = (obs.semantic == sem_id).sum()
                    print(f"[SCOUT_ENV]   ID {sem_id} -> '{cat_name}' ({pixel_count} px)")
        
        obs = self._postprocess_semantics(obs)
        self._last_obs = obs
        
        if self.verbose:
            print(f"[SCOUT_ENV] ----- observation complete -----\n")
        
        return obs
    
    def _preprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = np.where(depth > 10.0, 0.0, depth)
        depth = np.where(depth < 0.0, 0.0, depth)
        depth = np.where(np.isnan(depth), 0.0, depth)
        depth = np.where(np.isinf(depth), 0.0, depth)
        depth_clean = depth.copy()
        depth_clean[depth < self.min_depth] = 0.0
        depth_clean[depth > self.max_depth] = 0.0
        return depth_clean
    
    def _postprocess_semantics(self, obs: Observations) -> Observations:
        semantic = obs.semantic
        if semantic.ndim == 3:
            semantic = semantic[:, :, -1]
        max_valid_id = len(self.semantic_category_mapping.vocabulary) - 1
        semantic = np.clip(semantic, 0, max_valid_id)
        obs.semantic = semantic.astype(np.int32)
        
        instance_frame = self._generate_instance_ids(obs.semantic)
        obs.task_observations["instance_frame"] = instance_frame
        num_instances = len(np.unique(instance_frame)) - 1
        obs.task_observations["instance_scores"] = np.ones(max(0, num_instances))
        
        if self.verbose:
            print(f"[SCOUT_ENV] Instances detected: {num_instances}")
        
        return obs
    
    def _generate_instance_ids(self, semantic: np.ndarray) -> np.ndarray:
        instance_frame = np.zeros_like(semantic, dtype=np.int32)
        instance_id = 1
        for sem_id in np.unique(semantic):
            if sem_id == 0:
                continue
            mask = (semantic == sem_id).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(mask)
            for label_id in range(1, num_labels):
                instance_frame[labels == label_id] = instance_id
                instance_id += 1
        return instance_frame
    
    def _preprocess_goals(self, goals: List[Dict]) -> List[Dict]:
        processed = []
        for goal in goals:
            category = goal.get("category", "unknown")
            category_key = "_".join(category.split(" "))
            task_obs = {
                "type": goal.get("type", "objectnav"),
                "category": category,
                "semantic_id": self.semantic_category_mapping.get_category_id(category_key),
                "description": goal.get("description", f"Navigate to {category}"),
            }
            if goal.get("type") == "imagenav" and goal.get("image_path"):
                image_path = Path(goal["image_path"])
                if image_path.exists():
                    img = cv2.imread(str(image_path))
                    task_obs["image"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if self.verbose:
                        print(f"[SCOUT_ENV] Loaded goal image: {image_path}")
            processed.append(task_obs)
        return processed
    
    def apply_action(self, action: Any, info: Optional[Dict[str, Any]] = None, prev_obs: Optional[Observations] = None) -> bool:
        if self.verbose:
            print(f"\n[SCOUT_ENV] ----- apply_action -----")
        
        if isinstance(action, dict):
            action_enum = action.get("action", action)
        else:
            action_enum = action
        
        if self.verbose:
            print(f"[SCOUT_ENV] Action: {action_enum.name if hasattr(action_enum, 'name') else action_enum}")
        
        if self.visualizer is not None and info is not None:
            try:
                vis_kwargs = {k: v for k, v in info.items() if v is not None}
                if 'goal_name' not in vis_kwargs and self.current_task_idx < len(self.current_episode["tasks"]):
                    task = self.current_episode["tasks"][self.current_task_idx]
                    vis_kwargs['goal_name'] = task.get("category", task.get("description", "unknown"))
                if vis_kwargs.get('goal_name'):
                    self.visualizer.visualize(**vis_kwargs)
            except Exception as e:
                if self.verbose:
                    print(f"[SCOUT_ENV] Visualization warning: {e}")
        
        if action_enum == DiscreteNavigationAction.STOP:
            if self.verbose:
                print(f"[SCOUT_ENV] STOP received - Task {self.current_task_idx + 1} complete")
            self.current_task_idx += 1
            if self.current_task_idx >= len(self.current_episode["tasks"]):
                if self.verbose:
                    print(f"[SCOUT_ENV] ✓ All tasks complete - Episode over")
                self.episode_over = True
                return True
            else:
                if self.verbose:
                    next_task = self.current_episode["tasks"][self.current_task_idx]
                    print(f"[SCOUT_ENV] Starting task {self.current_task_idx + 1}: {next_task.get('category', 'unknown')}")
                self.reset_vis_dir()
                return False
        
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
        
        if np.any(continuous_action != 0):
            try:
                if self.verbose:
                    print(f"[SCOUT_ENV] Sending to robot: {continuous_action}")
                self.robot.nav.navigate_to(continuous_action, relative=True, blocking=True)
                if self.verbose:
                    new_pose = self.robot.get_base_pose()
                    print(f"[SCOUT_ENV] New pose: x={new_pose[0]:.2f}, y={new_pose[1]:.2f}, θ={np.degrees(new_pose[2]):.1f}°")
            except Exception as e:
                logger.error(f"Navigation failed: {e}")
                if self.verbose:
                    print(f"[SCOUT_ENV] ✗ Navigation error: {e}")
        
        self.timestep += 1
        rospy.sleep(0.1)
        
        if self.verbose:
            print(f"[SCOUT_ENV] ----- action complete (timestep={self.timestep}) -----\n")
        
        return False
    
    def add_subepisode_metrics(self, all_metrics: Dict, action: Any) -> None:
        if isinstance(action, dict):
            task_idx = action.get("action_args", {}).get("task_idx", self.current_task_idx - 1)
        else:
            task_idx = self.current_task_idx - 1
        if task_idx < 0:
            task_idx = 0
        
        task_info = {}
        if task_idx < len(self.current_episode["tasks"]):
            task = self.current_episode["tasks"][task_idx]
            task_info = {"task_type": task.get("type", "objectnav"), "category": task.get("category", "unknown")}
        
        all_metrics[task_idx] = {
            "timesteps": self.timestep,
            "success": True,
            "spl": np.nan,
            "distance_to_goal": np.nan,
            **task_info,
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