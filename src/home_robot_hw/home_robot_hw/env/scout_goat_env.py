#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Scout GOAT Environment - Real-world multi-task navigation environment
Supports: ObjectNav, ImageNav, LanguageNav
Simplified for ObjectNav with merged category vocabulary.
"""

from typing import Any, Dict, List, Optional
import json
import numpy as np
import rospy
import cv2
from pathlib import Path

import home_robot
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.utils.geometry import xyt2sophus
from home_robot.perception.constants import SemanticCategoryMapping
from home_robot.utils.constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
)
from home_robot.utils.logger import get_logger

from home_robot_hw.remote import ScoutClient

# Try to import visualizer
try:
    from home_robot_hw.env.visualizer import Visualizer
except ImportError:
    print("[WARNING] Visualizer not available, using dummy")
    class Visualizer:
        def __init__(self, config):
            pass
        def reset(self):
            pass
        def set_vis_dir(self, scene_id, episode_str):
            pass
        def visualize(self, **kwargs):
            pass

logger = get_logger()


# =============================================================================
# SIMPLIFIED GoatSemanticCategoryMapping
# =============================================================================

class GoatSemanticCategoryMapping(SemanticCategoryMapping):
    """
    Simple semantic category mapping for ObjectNav.
    
    Categories are 1-indexed to match Detic output:
    - ID 0 = background/unknown
    - ID 1 = first category in vocabulary (alphabetically)
    - ID 2 = second category, etc.
    """
    
    def __init__(self, vocabulary: List[str]):
        """
        Args:
            vocabulary: List of category names (MUST be sorted alphabetically!)
        """
        # Vocabulary: index 0 = 'unknown', index 1+ = categories
        self.vocabulary = ['unknown'] + list(vocabulary)
        self._num_sem_categories = len(vocabulary)  # Don't count 'unknown'
        
        # Bidirectional mapping (1-indexed for categories)
        self.goal_name_to_goal_id = {name: idx for idx, name in enumerate(self.vocabulary)}
        self.goal_id_to_goal_name = {idx: name for idx, name in enumerate(self.vocabulary)}
        
        # Debug output
        print(f"[CATEGORY MAP] Initialized with {self._num_sem_categories} categories")
        print(f"[CATEGORY MAP] Sample mappings:")
        for cat in ['chair', 'cup', 'table', 'bottle', 'laptop', 'bed', 'couch']:
            if cat in self.goal_name_to_goal_id:
                print(f"    '{cat}' -> ID {self.goal_name_to_goal_id[cat]}")
        
        # Color palette for visualization
        np.random.seed(42)
        self._map_color_palette = [
            tuple(np.random.randint(0, 255, 3).tolist()) 
            for _ in range(len(self.vocabulary))
        ]
        self._frame_color_palette = self._map_color_palette.copy()
        
        # Instance tracking buffer
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
        """Get category ID from name. Returns 0 if not found."""
        if category_name in self.goal_name_to_goal_id:
            return self.goal_name_to_goal_id[category_name]
        
        # Try lowercase match
        for name, idx in self.goal_name_to_goal_id.items():
            if name.lower() == category_name.lower():
                return idx
        
        print(f"[WARNING] Category '{category_name}' not in vocabulary!")
        return 0
        
    def get_category_name(self, category_id: int) -> str:
        """Get category name from ID. Returns 'unknown' if not found."""
        return self.goal_id_to_goal_name.get(category_id, "unknown")


# =============================================================================
# ScoutGoatEnv - Main Environment Class
# =============================================================================

class ScoutGoatEnv:
    """
    GOAT environment for Scout robot - supports multi-task episodes.
    Simplified for ObjectNav with unified category vocabulary.
    """
    
    def __init__(
        self, 
        config=None,
        task_config_file: Optional[str] = None,
        forward_step: float = 0.25, 
        rotate_step: float = 30.0,
        *args, 
        **kwargs
    ):
        """
        Args:
            config: Configuration object
            task_config_file: Path to JSON file with task definitions
            forward_step: Forward movement distance in meters
            rotate_step: Rotation angle in degrees
        """
        self.verbose = getattr(config, 'VERBOSE', False)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("ScoutGoatEnv.__init__() - Initializing environment")
            print(f"{'='*60}\n")
        
        self.config = config
        self.forward_step = forward_step
        self.rotate_step = np.radians(rotate_step)
        
        # Load task configuration and vocabulary
        self.task_config_file = task_config_file or "projects/real_world_ovmm/configs/example_tasks.json"
        self.load_task_config()
        
        # Setup semantic category mapping
        self.semantic_category_mapping = GoatSemanticCategoryMapping(self.vocabulary)
        
        # Update config with actual num_sem_categories
        from omegaconf import open_dict
        with open_dict(config):
            config.AGENT.SEMANTIC_MAP.num_sem_categories = self.semantic_category_mapping.num_sem_categories + 1
        
        logger.info(f"Semantic categories: {self.semantic_category_mapping.num_sem_categories}")
        
        # Initialize Detic perception
        # IMPORTANT: Detic vocabulary must match our vocabulary order (sorted alphabetically)
        detic_vocab = ",".join(self.vocabulary)
        if self.verbose:
            print(f"[ENV] Initializing Detic with {len(self.vocabulary)} categories")
        
        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary=detic_vocab,
            sem_gpu_id=(0 if not config.NO_GPU else -1),
        )
        
        # Initialize visualizer
        if config is not None and (config.VISUALIZE or config.PRINT_IMAGES):
            self.visualizer = Visualizer(config)
        else:
            self.visualizer = None
            
        # Create robot client
        if self.verbose:
            print("[ENV] Connecting to ScoutClient...")
        self.robot = ScoutClient()
        if self.verbose:
            print("[ENV] ✓ Connected to robot")
        
        # Episode state
        self.current_episode = None
        self.current_task_idx = 0
        self.episode_over = False
        self.timestep = 0
        self.episode_id = 0
        self.scene_id = "real_world"
        self._episode_start_pose = None
        
        # Depth limits from config
        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth
        
        # Initialize
        self.reset()
        
        if self.verbose:
            print("[ENV] ScoutGoatEnv.__init__() - Complete\n")
        
    def load_task_config(self):
        """Load task definitions and vocabulary from category map."""
        # Load episodes from task config
        config_path = Path(self.task_config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Task config not found: {self.task_config_file}")
            
        with open(config_path, 'r') as f:
            task_data = json.load(f)
            
        self.episodes = task_data.get("episodes", [])
        if not self.episodes:
            raise ValueError("No episodes defined in task config")
        
        # Load vocabulary from category map file
        cat_map_file = getattr(self.config.ENVIRONMENT, 'category_map_file', None)
        
        if cat_map_file and Path(cat_map_file).exists():
            with open(cat_map_file, 'r') as f:
                cat_map = json.load(f)
            
            # Simple format: just a "categories" list
            if 'categories' in cat_map:
                self.vocabulary = sorted(cat_map['categories'])
                print(f"[ENV] Loaded {len(self.vocabulary)} categories from {cat_map_file}")
            
            # Legacy format: obj + recep dicts
            elif 'obj_category_to_obj_category_id' in cat_map:
                vocab_set = set()
                vocab_set.update(cat_map.get('obj_category_to_obj_category_id', {}).keys())
                vocab_set.update(cat_map.get('recep_category_to_recep_category_id', {}).keys())
                self.vocabulary = sorted(list(vocab_set))
                print(f"[ENV] Loaded {len(self.vocabulary)} categories from legacy format")
            
            else:
                raise ValueError(f"Unknown category map format in {cat_map_file}")
        
        else:
            # Fallback: extract from tasks
            vocab_set = set()
            for episode in self.episodes:
                for task in episode.get("tasks", []):
                    if "category" in task:
                        vocab_set.add(task["category"])
            self.vocabulary = sorted(list(vocab_set))
            print(f"[ENV] Loaded {len(self.vocabulary)} categories from tasks")
        
        print(f"[ENV] Loaded {len(self.episodes)} episodes")
        
    def reset(self):
        """Reset environment for new episode."""
        if self.current_episode is None:
            self.current_episode = self.episodes[0] if self.episodes else None
        
        if self.current_episode is None:
            raise ValueError("No episode available")
            
        self.current_task_idx = 0
        self.episode_over = False
        self.timestep = 0
        
        self._episode_start_pose = xyt2sophus(self.robot.get_base_pose())
        
        if self.visualizer is not None:
            self.visualizer.reset()
            
        self.scene_id = "real_world"
        
        print(f"\n{'='*60}")
        print(f"Episode {self.episode_id} - {self.current_episode.get('description', 'GOAT Episode')}")
        print(f"Tasks: {len(self.current_episode['tasks'])}")
        for i, task in enumerate(self.current_episode['tasks']):
            cat = task.get('category', task.get('description', 'N/A'))
            sem_id = self.semantic_category_mapping.get_category_id(cat) if hasattr(self, 'semantic_category_mapping') else '?'
            print(f"  {i+1}. {task['type'].upper()}: {cat} (semantic_id={sem_id})")
        print(f"{'='*60}\n")
        
    def next_episode(self) -> bool:
        """Move to next episode. Returns False if no more episodes."""
        self.episode_id += 1
        if self.episode_id >= len(self.episodes):
            return False
        self.current_episode = self.episodes[self.episode_id]
        self.reset()
        return True
        
    def reset_visualization(self):
        """Reset visualization for new task."""
        if self.visualizer is not None:
            self.visualizer.set_vis_dir(
                self.scene_id, 
                f"{self.episode_id}_{self.current_task_idx}"
            )
            
    def get_observation(self) -> Observations:
        """Get current observation from robot sensors."""
        if self.verbose:
            print(f"\n[OBS] Getting observation (step {self.timestep})...")
        
        # Get RGB-D from robot
        rgb, depth, _ = self.robot.get_images(compute_xyz=True, rotate_images=False)
        
        if self.verbose:
            print(f"[OBS] RGB: {rgb.shape}, Depth: {depth.shape} (min={depth.min():.2f}, max={depth.max():.2f})")
        
        # Get current pose
        current_pose = xyt2sophus(self.robot.get_base_pose())
        
        # Compute relative pose from episode start
        relative_pose = self._episode_start_pose.inverse() * current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]
        gps = relative_pose.translation()[:2]
        
        # Preprocess depth
        depth = self._preprocess_depth(depth)
        
        # DEBUG: Verify depth is clean before creating obs
        print(f"[OBS DEBUG] Depth after preprocess: min={depth.min():.3f}, max={depth.max():.3f}")
    
        
        # Create observation
        obs = home_robot.core.interfaces.Observations(
            rgb=rgb.copy(),
            depth=depth.copy(),
            gps=gps,
            compass=np.array([theta]),
            task_observations={
                "tasks": self._get_task_observations(),
            },
            camera_pose=None,
            third_person_image=None,
        )
        
        # Run Detic perception
        if self.verbose:
            print("[OBS] Running Detic...")
        obs = self.segmentation.predict(obs, depth_threshold=self.max_depth)
        
        if self.verbose:
            unique_ids = np.unique(obs.semantic)
            print(f"[OBS] Detic detected {len(unique_ids)} unique IDs: {unique_ids[:10].tolist()}...")
        
        # Post-process semantics
        obs = self._postprocess_semantics(obs)
        
        return obs
        
    def _preprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        """Preprocess depth map."""
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        
        print(f"[DEPTH DEBUG] Raw: min={depth.min():.3f}, max={depth.max():.3f}")
        
        # Replace invalid values with 0 (NOT sentinel values!)
        depth = np.where(depth > 10.0, 0.0, depth)  # >10m is invalid
        depth = np.where(depth < 0.0, 0.0, depth)   # <0 is invalid
        
        # Clip to valid sensor range and set out-of-range to 0
        depth_clean = depth.copy()
        depth_clean[depth < self.min_depth] = 0.0
        depth_clean[depth > self.max_depth] = 0.0
        
        print(f"[DEPTH DEBUG] Final: min={depth_clean.min():.3f}, max={depth_clean.max():.3f}")
        
        return depth_clean
            
    def _postprocess_semantics(self, obs: Observations) -> Observations:
        """Post-process semantic segmentation."""
        
        # DEBUG: Check Detic raw output
        unique_ids = np.unique(obs.semantic)
        print(f"🔴 DETIC RAW IDs: {unique_ids[:15].tolist()}")
        
        for uid in unique_ids:
            if uid > 0:
                pixel_count = (obs.semantic == uid).sum()
                # Use semantic_category_mapping to get the name (it has 'unknown' at 0)
                cat_name = self.semantic_category_mapping.get_category_name(uid)
                print(f"   Detic ID {uid} -> '{cat_name}' ({pixel_count}px)")
                if "cup" in cat_name.lower():
                    print(f"   🎯 CUP FOUND AT DETIC ID {uid}")
        
        # NO SHIFT - Detic IDs already match semantic_category_mapping
        # Just clamp invalid IDs to 0
        max_valid_id = len(self.semantic_category_mapping.vocabulary) - 1
        obs.semantic[obs.semantic > max_valid_id] = 0
        
        obs.task_observations["instance_frame"] = self._generate_instance_ids(obs.semantic)
        return obs
    
    def _generate_instance_ids(self, semantic: np.ndarray) -> np.ndarray:
        """Generate pseudo-instance IDs from semantic map using connected components."""
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
        
    def _get_task_observations(self) -> List[Dict[str, Any]]:
        """Get task observations for current episode."""
        tasks = []
        
        for task_def in self.current_episode["tasks"]:
            category = task_def.get("category", "unknown")
            semantic_id = self.semantic_category_mapping.get_category_id(category)
            
            # DEBUG: Print what ID the task expects
            print(f"🔵 Task expects: category='{category}' -> semantic_id={semantic_id}")
            
            task_obs = {
                "type": task_def["type"],
                "category": category,
                "semantic_id": semantic_id,
                "description": task_def.get("description", f"Navigate to {category}"),
            }
            
            # For imagenav, load the goal image
            if task_def["type"] == "imagenav":
                image_path = task_def.get("image_path")
                if image_path and Path(image_path).exists():
                    goal_image = cv2.imread(str(image_path))
                    goal_image = cv2.cvtColor(goal_image, cv2.COLOR_BGR2RGB)
                    task_obs["image"] = goal_image
            
            tasks.append(task_obs)
        
        return tasks
        
    def apply_action(
        self,
        action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[Observations] = None,
    ):
        """Apply discrete action to the robot."""
        
        # Visualization with robust error handling
        if self.visualizer is not None and info is not None:
            try:
                vis_kwargs = {}
                
                # Only copy safe, non-None values
                for key, value in info.items():
                    if value is not None:
                        vis_kwargs[key] = value
                
                # Ensure goal_name exists
                if 'goal_name' not in vis_kwargs:
                    if self.current_episode and self.current_task_idx < len(self.current_episode["tasks"]):
                        task = self.current_episode["tasks"][self.current_task_idx]
                        vis_kwargs['goal_name'] = task.get("category", task.get("description", "unknown"))
                    else:
                        vis_kwargs['goal_name'] = "unknown"
                
                # Ensure semantic_frame exists
                if 'semantic_frame' not in vis_kwargs:
                    if prev_obs is not None and hasattr(prev_obs, 'task_observations'):
                        vis_kwargs['semantic_frame'] = prev_obs.task_observations.get('semantic_frame')
                    else:
                        vis_kwargs['semantic_frame'] = None
                
                if vis_kwargs.get('goal_name'):
                    self.visualizer.visualize(**vis_kwargs)
                    
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] Visualization failed: {type(e).__name__}: {e}")
            
        # Handle STOP action
        if action == DiscreteNavigationAction.STOP:
            if self.verbose:
                print(f"[ACTION] STOP - Task {self.current_task_idx + 1} complete")
                
            print(f"\n[TASK {self.current_task_idx + 1} COMPLETE]")
            self.current_task_idx += 1
            
            if self.current_task_idx >= len(self.current_episode["tasks"]):
                if self.verbose:
                    print("[ACTION] All tasks complete - Episode over")
                print(f"\n{'='*60}")
                print("ALL TASKS IN EPISODE COMPLETE!")
                print(f"{'='*60}\n")
                self.episode_over = True
                return True
            else:
                if self.verbose:
                    print(f"[ACTION] Starting next task: {self.current_task_idx + 1}")
                print(f"\n[STARTING TASK {self.current_task_idx + 1}]")
                self.reset_visualization()
                return False
                
        # Execute movement
        continuous_action = np.zeros(3)
        action_name = ""
        
        if action == DiscreteNavigationAction.MOVE_FORWARD:
            action_name = "FORWARD"
            continuous_action[0] = self.forward_step
        elif action == DiscreteNavigationAction.TURN_RIGHT:
            action_name = "TURN RIGHT"
            continuous_action[2] = -self.rotate_step
        elif action == DiscreteNavigationAction.TURN_LEFT:
            action_name = "TURN LEFT"
            continuous_action[2] = self.rotate_step
            
        if np.any(continuous_action != 0):
            print(f"→ {action_name}")
            
            try:
                self.robot.nav.navigate_to(
                    continuous_action, 
                    relative=True, 
                    blocking=True
                )
            except Exception as e:
                logger.error(f"Navigation failed: {e}")
            
        self.timestep += 1
        rospy.sleep(0.1)
        
        return False
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
        
    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get metrics for current episode."""
        return {
            "episode_id": self.episode_id,
            "tasks_completed": self.current_task_idx,
            "total_tasks": len(self.current_episode["tasks"]),
            "timesteps": self.timestep,
        }
        
    def get_robot(self):
        """Get robot client."""
        return self.robot


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Scout GOAT Environment Test")
    print("="*60)
    
    rospy.init_node("scout_goat_env_test")
    
    # Create minimal config for testing
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        'NO_GPU': False,
        'VISUALIZE': False,
        'PRINT_IMAGES': False,
        'VERBOSE': True,
        'ENVIRONMENT': {
            'min_depth': 0.2,
            'max_depth': 5.0,
            'category_map_file': 'projects/real_world_ovmm/configs/example_cat_map.json',
        },
        'AGENT': {
            'SEMANTIC_MAP': {
                'num_sem_categories': 150,
            }
        }
    })
    
    env = ScoutGoatEnv(
        config=config,
        task_config_file="projects/real_world_ovmm/configs/example_tasks.json"
    )
    
    print(f"\n[TEST] Vocabulary size: {len(env.vocabulary)}")
    print(f"[TEST] Sample: {env.vocabulary[:10]}")
    
    # Test category mapping
    print("\n[TEST] Category ID mapping:")
    for cat in ['cup', 'chair', 'table', 'bottle', 'laptop']:
        cat_id = env.semantic_category_mapping.get_category_id(cat)
        print(f"  '{cat}' -> {cat_id}")
    
    print("\n[TEST] Getting observation...")
    obs = env.get_observation()
    print(f"[TEST] RGB: {obs.rgb.shape}")
    print(f"[TEST] Semantic unique: {np.unique(obs.semantic)[:10]}")
    
    print("\n[TEST] Complete!")
