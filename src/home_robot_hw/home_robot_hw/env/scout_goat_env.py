#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Scout GOAT Environment - Real-world multi-task navigation environment
Supports: ObjectNav, ImageNav, LanguageNav
Inherits from HabitatGoatEnv for compatibility
"""

from typing import Any, Dict, List, Optional
import json
import numpy as np
import rospy
import cv2
from pathlib import Path
import sys

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

# Import parent class
try:
    # Try to import from home_robot_sim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src/home_robot_sim"))
    from home_robot_sim.env.habitat_goat_env.habitat_goat_env import HabitatGoatEnv
    PARENT_CLASS_AVAILABLE = True
except ImportError:
    print("Warning: HabitatGoatEnv not available, creating minimal base class")
    PARENT_CLASS_AVAILABLE = False
    # Create minimal base class
    class HabitatGoatEnv:
        def __init__(self, *args, **kwargs):
            pass

# Try to import visualizer, create dummy if not available
try:
    from home_robot_hw.env.visualizer import Visualizer
except ImportError:
    print("Warning: Visualizer not available, using dummy")
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


class GoatSemanticCategoryMapping(SemanticCategoryMapping):
    """Custom semantic category mapping for GOAT tasks"""
    
    def __init__(self, vocabulary: List[str]):
        """
        Args:
            vocabulary: List of category names from category map
        """
        # Store vocabulary with 0 as 'unknown' (matching Habitat convention)
        self.vocabulary = ['unknown'] + vocabulary
        self._num_sem_categories = len(self.vocabulary) - 1  # Don't count 'unknown'
        
        # Create bidirectional mapping (1-indexed to match Habitat)
        self.goal_name_to_goal_id = {name: idx for idx, name in enumerate(self.vocabulary)}
        self.goal_id_to_goal_name = {idx: name for idx, name in enumerate(self.vocabulary)}
        
        # Create color palette for visualization
        np.random.seed(42)
        self._map_color_palette = [
            tuple(np.random.randint(0, 255, 3).tolist()) 
            for _ in range(len(self.vocabulary))
        ]
        self._frame_color_palette = self._map_color_palette.copy()
        
        # Instance tracking (empty initially, gets populated by environment)
        self._instance_id_to_category_id = np.zeros(10000, dtype=np.int32)  # Large enough buffer
        
    @property
    def num_sem_categories(self) -> int:
        """Number of semantic categories (excluding 'unknown')"""
        return self._num_sem_categories
    
    @property
    def map_color_palette(self) -> List[tuple]:
        """Color palette for map visualization"""
        return self._map_color_palette
    
    @property
    def frame_color_palette(self) -> List[tuple]:
        """Color palette for frame visualization"""
        return self._frame_color_palette
    
    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        """Mapping from instance IDs to category IDs"""
        return self._instance_id_to_category_id
    
    @property
    def categories_legend_path(self) -> str:
        """Path to categories legend file (not used in real world)"""
        return ""
    
    @property
    def map_goal_id(self) -> int:
        """Goal ID for map visualization (not used)"""
        return 0
    
    def reset_instance_id_to_category_id(self, env=None):
        """
        Reset instance ID to category ID mapping
        In Habitat, this reads from the simulator. For real world, we generate on-the-fly.
        """
        # Reset to zeros - will be populated by environment as instances are discovered
        self._instance_id_to_category_id.fill(0)
        
    def get_category_id(self, category_name: str) -> int:
        """Get category ID from name"""
        # Handle variations in naming
        normalized_name = "_".join(category_name.lower().split())
        for name, idx in self.goal_name_to_goal_id.items():
            if "_".join(name.lower().split()) == normalized_name:
                return idx
        return 0  # Unknown category
        
    def get_category_name(self, category_id: int) -> str:
        """Get category name from ID"""
        return self.goal_id_to_goal_name.get(category_id, "unknown")


class ScoutGoatEnv(HabitatGoatEnv):
    """
    GOAT environment for Scout robot - supports multi-task episodes
    Inherits from HabitatGoatEnv but replaces Habitat simulator with real robot
    """
    
    def __init__(
        self, 
        config=None,
        habitat_env=None,  # Not used but kept for compatibility with parent signature
        task_config_file: Optional[str] = None,
        forward_step: float = 0.25, 
        rotate_step: float = 30.0,
        *args, 
        **kwargs
    ):
        """
        Args:
            config: Configuration object
            habitat_env: Unused (kept for parent compatibility)
            task_config_file: Path to JSON file with task definitions
            forward_step: Forward movement distance in meters
            rotate_step: Rotation angle in degrees
        """
        # Store verbose flag
        self.verbose = getattr(config, 'VERBOSE', False)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("ScoutGoatEnv.__init__() - Initializing environment")
            print(f"  task_config_file: {task_config_file}")
            print(f"  forward_step: {forward_step}m")
            print(f"  rotate_step: {rotate_step}°")
            print(f"  verbose: {self.verbose}")
            print(f"{'='*60}\n")
        
        # Don't call parent __init__ since we're not using Habitat
        # Instead, initialize our own attributes
        
        self.config = config
        self.forward_step = forward_step
        self.rotate_step = np.radians(rotate_step)
        
        # Load task configuration
        self.task_config_file = task_config_file or "projects/real_world_goat/configs/example_tasks.json"
        if self.verbose:
            print(f"[VERBOSE] Loading task config from: {self.task_config_file}")
        self.load_task_config()
        
        # Setup semantic category mapping (similar to parent but without Habitat)
        if self.verbose:
            print("[VERBOSE] Creating GoatSemanticCategoryMapping")
        self.semantic_category_mapping = GoatSemanticCategoryMapping(self.vocabulary)
        
        # Update config with actual num_sem_categories (needed by visualizer and agent)
        # This overrides the placeholder value in YAML
        from omegaconf import OmegaConf, open_dict
        with open_dict(config):
            config.AGENT.SEMANTIC_MAP.num_sem_categories = self.semantic_category_mapping.num_sem_categories
        
        logger.info(f"Semantic categories: {self.semantic_category_mapping.num_sem_categories}")
        if self.verbose:
            print(f"[VERBOSE]   num_sem_categories: {self.semantic_category_mapping.num_sem_categories}")
        
        # Initialize attributes that parent class would have
        self.task_type = "Goat-v1"  # Matches parent's task type
        self.ground_truth_semantics = False  # Always False for real world
        
        # Initialize Detic perception for object detection
        # Note: vocabulary[0] is 'unknown', so we skip it for Detic
        detic_vocabulary = self.vocabulary  # Use original vocabulary (no 'unknown' prefix)
        if self.verbose:
            print(f"[VERBOSE] Initializing Detic with vocabulary: {detic_vocabulary}")
        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary="," + ",".join(detic_vocabulary),
            sem_gpu_id=(0 if not config.NO_GPU else -1),
        )
        
        # Initialize visualizer
        if config is not None and (config.VISUALIZE or config.PRINT_IMAGES):
            if self.verbose:
                print("[VERBOSE] Initializing Visualizer")
            self.visualizer = Visualizer(config)
        else:
            self.visualizer = None
            if self.verbose:
                print("[VERBOSE] Visualization disabled")
            
        # Create robot client (replaces Habitat simulator)
        if self.verbose:
            print("[VERBOSE] Connecting to ScoutClient...")
        self.robot = ScoutClient()
        if self.verbose:
            print("[VERBOSE]   ✓ Connected to robot")
        
        # Episode state (matching parent class attributes)
        self.current_episode = None
        self.current_task_idx = 0
        self.episode_over = False
        self.timestep = 0
        self.episode_id = 0
        self.scene_id = "real_world"
        self._episode_start_pose = None
        
        # Initialize minimum/maximum depth from config
        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth
        
        if self.verbose:
            print("[VERBOSE] Calling reset()...")
        self.reset()
        if self.verbose:
            print("[VERBOSE] ScoutGoatEnv.__init__() - Complete\n")
        
    def load_task_config(self):
        """Load task definitions from JSON file"""
        config_path = Path(self.task_config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Task config not found: {self.task_config_file}")
            
        with open(config_path, 'r') as f:
            task_data = json.load(f)
            
        self.episodes = task_data.get("episodes", [])
        if not self.episodes:
            raise ValueError("No episodes defined in task config")
            
        # Extract vocabulary from all tasks
        vocab_set = set()
        for episode in self.episodes:
            for task in episode.get("tasks", []):
                if "category" in task:
                    vocab_set.add(task["category"])
        self.vocabulary = sorted(list(vocab_set))
        
        print(f"Loaded {len(self.episodes)} episodes with vocabulary: {self.vocabulary}")
        
    def reset(self):
        """
        Reset environment for new episode
        Overrides parent class method to use real robot instead of Habitat
        """
        # Get next episode
        if self.current_episode is None:
            self.current_episode = self.episodes[0] if self.episodes else None
        
        if self.current_episode is None:
            raise ValueError("No episode available")
            
        # Reset task tracking
        self.current_task_idx = 0
        self.episode_over = False
        self.timestep = 0
        
        # Record starting pose (replaces Habitat's episode start pose)
        self._episode_start_pose = xyt2sophus(self.robot.get_base_pose())
        
        # Reset visualizer
        if self.visualizer is not None:
            self.visualizer.reset()
            
        # Set scene_id and episode info (matching parent class)
        self.scene_id = "real_world"
        
        # Create an episode object that mimics Habitat's episode structure
        class Episode:
            def __init__(self, episode_data):
                self.episode_id = str(episode_data.get('episode_id', 0))
                self.scene_id = "real_world"
                # For compatibility with parent class expectations
                self.object_category = episode_data['tasks'][0].get('category', 'unknown') if episode_data.get('tasks') else 'unknown'
        
        self.episode = Episode(self.current_episode)
            
        print(f"\n{'='*60}")
        print(f"Episode {self.episode_id} - {self.current_episode.get('description', 'GOAT Episode')}")
        print(f"Tasks: {len(self.current_episode['tasks'])}")
        for i, task in enumerate(self.current_episode['tasks']):
            print(f"  {i+1}. {task['type'].upper()}: {task.get('category', task.get('description', 'N/A'))}")
        print(f"{'='*60}\n")
        
    def next_episode(self) -> bool:
        """Move to next episode. Returns True if successful, False if no more episodes."""
        self.episode_id += 1
        if self.episode_id >= len(self.episodes):
            return False
        self.current_episode = self.episodes[self.episode_id]
        self.reset()
        return True
        
    def reset_visualization(self):
        """Reset visualization for new task"""
        if self.visualizer is not None:
            self.visualizer.set_vis_dir(
                self.scene_id, 
                f"{self.episode_id}_{self.current_task_idx}"
            )
            
    def get_observation(self) -> Observations:
        """
        Get current observation from robot sensors
        Overrides parent's _preprocess_obs to use real robot data
        """
        if self.verbose:
            print(f"\n[VERBOSE] get_observation() - Step {self.timestep}")
            print("[VERBOSE]   Getting images from robot...")
        
        # Get RGB-D from robot (replaces Habitat observation)
        rgb, depth, _ = self.robot.get_images(compute_xyz=True, rotate_images=False)
        
        if self.verbose:
            print(f"[VERBOSE]   RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
            print(f"[VERBOSE]   Depth shape: {depth.shape}, min: {depth.min():.3f}, max: {depth.max():.3f}")
        
        # Get current pose (replaces Habitat GPS/compass)
        if self.verbose:
            print("[VERBOSE]   Getting robot pose...")
        current_pose = xyt2sophus(self.robot.get_base_pose())
        
        # Compute relative pose from episode start
        relative_pose = self._episode_start_pose.inverse() * current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]
        gps = relative_pose.translation()[:2]
        
        if self.verbose:
            print(f"[VERBOSE]   Relative GPS: [{gps[0]:.3f}, {gps[1]:.3f}]")
            print(f"[VERBOSE]   Relative heading: {np.degrees(theta):.1f}°")
        
        # Preprocess depth (similar to parent's _preprocess_depth)
        if self.verbose:
            print("[VERBOSE]   Preprocessing depth...")
        depth = self._preprocess_depth(depth)
        
        # Create observation (matching parent class structure)
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
        
        # Run Detic perception (replaces parent's GT semantics)
        if self.verbose:
            print("[VERBOSE]   Running Detic perception...")
        obs = self.segmentation.predict(obs, depth_threshold=self.max_depth)
        
        if self.verbose:
            unique_sem = np.unique(obs.semantic)
            print(f"[VERBOSE]   Detic detected {len(unique_sem)} unique categories: {unique_sem.tolist()}")
            for sem_id in unique_sem:
                if sem_id > 0:
                    pixel_count = (obs.semantic == sem_id).sum()
                    print(f"[VERBOSE]     Detic ID {sem_id}: {pixel_count} pixels")
        
        # Post-process semantics (similar to parent's _preprocess_semantic)
        if self.verbose:
            print("[VERBOSE]   Post-processing semantics...")
        obs = self._postprocess_semantics(obs)
        
        if self.verbose:
            unique_mapped = np.unique(obs.semantic)
            print(f"[VERBOSE]   Mapped to {len(unique_mapped)} categories: {unique_mapped.tolist()}")
            print("[VERBOSE] get_observation() - Complete\n")
        
        return obs
        
    def _preprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Preprocess depth map
        Similar to parent's _preprocess_depth but adapted for real sensor
        """
        # Ensure depth is single channel
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        
        # Apply min/max scaling (parent uses min_depth + depth * (max_depth - min_depth))
        # For real robot, depth is already in meters, just clip
        rescaled_depth = np.clip(depth, self.min_depth, self.max_depth)
        
        # Replace invalid values (matching parent's approach)
        rescaled_depth[depth < self.min_depth] = MIN_DEPTH_REPLACEMENT_VALUE
        rescaled_depth[depth > self.max_depth] = MAX_DEPTH_REPLACEMENT_VALUE
        
        return rescaled_depth
        
    def _postprocess_semantics(self, obs: Observations) -> Observations:
        """Post-process semantic segmentation"""
        # Map Detic predictions to our category IDs
        semantic_mapped = np.zeros_like(obs.semantic)
        
        # Get unique semantic IDs from Detic
        unique_ids = np.unique(obs.semantic)
        
        for detic_id in unique_ids:
            if detic_id == 0:  # Background
                continue
            
            # Detic IDs are 1-indexed (0 is background)
            # Our vocabulary is 0-indexed
            vocab_idx = detic_id - 1
            
            if vocab_idx < len(self.vocabulary):
                category_name = self.vocabulary[vocab_idx]
                # Map to our category mapping
                our_id = self.semantic_category_mapping.get_category_id(category_name)
                semantic_mapped[obs.semantic == detic_id] = our_id
            else:
                # Unknown category - map to 0
                semantic_mapped[obs.semantic == detic_id] = 0
            
        obs.semantic = semantic_mapped
        
        # Store instance frame (use semantic as proxy since we don't have GT instances)
        # Generate unique instance IDs by combining semantic class and spatial location
        obs.task_observations["instance_frame"] = self._generate_instance_ids(obs.semantic)
        
        return obs
    
    def _generate_instance_ids(self, semantic: np.ndarray) -> np.ndarray:
        """Generate pseudo-instance IDs from semantic map using connected components"""
        import cv2
        instance_frame = np.zeros_like(semantic, dtype=np.int32)
        instance_id = 1
        
        # For each semantic category, find connected components
        for sem_id in np.unique(semantic):
            if sem_id == 0:  # Skip background
                continue
            
            # Create mask for this category
            mask = (semantic == sem_id).astype(np.uint8)
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(mask)
            
            # Assign unique instance IDs
            for label_id in range(1, num_labels):  # Skip background (0)
                instance_frame[labels == label_id] = instance_id
                instance_id += 1
        
        return instance_frame
        
    def _get_task_observations(self) -> List[Dict[str, Any]]:
        """
        Get task observations for current episode
        Similar to parent's _preprocess_goals but loads from JSON instead of Habitat dataset
        """
        tasks = []
        
        for task_def in self.current_episode["tasks"]:
            task_obs = {
                "type": task_def["type"],
                "semantic_id": 0,  # Will be set based on type
            }
            
            if task_def["type"] == "objectnav":
                task_obs["category"] = task_def["category"]
                task_obs["semantic_id"] = self.semantic_category_mapping.get_category_id(task_def["category"])
                
            elif task_def["type"] == "imagenav":
                # Load goal image (parent gets this from Habitat dataset)
                image_path = task_def.get("image_path")
                if image_path and Path(image_path).exists():
                    goal_image = cv2.imread(str(image_path))
                    goal_image = cv2.cvtColor(goal_image, cv2.COLOR_BGR2RGB)
                    task_obs["image"] = goal_image
                task_obs["category"] = task_def.get("category", "object")
                task_obs["semantic_id"] = self.semantic_category_mapping.get_category_id(task_obs["category"])
                
            elif task_def["type"] == "languagenav":
                task_obs["description"] = task_def["description"]
                task_obs["category"] = task_def.get("category", "object")
                task_obs["semantic_id"] = self.semantic_category_mapping.get_category_id(task_obs["category"])
                
            tasks.append(task_obs)
            
        return tasks
        
    def apply_action(
        self,
        action: Action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[Observations] = None,
    ) -> bool:
        """
        Apply action to robot
        Overrides parent's apply_action to use real robot instead of Habitat simulator
        
        Returns:
            done: True if episode is complete
        """
        if self.verbose:
            print(f"\n[VERBOSE] apply_action() - Timestep {self.timestep}")
            print(f"[VERBOSE]   Action: {action}")
            print(f"[VERBOSE]   Current task: {self.current_task_idx + 1}/{len(self.current_episode['tasks'])}")
        
        # Visualize if enabled (similar to parent's _process_info)
        if self.visualizer is not None and info is not None:
            if self.verbose:
                print("[VERBOSE]   Calling visualizer...")
            
            # Prepare visualization info
            vis_info = info.copy()
            
            # Add required fields for visualizer
            if prev_obs is not None:
                vis_info['rgb_frame'] = prev_obs.rgb[:, :, ::-1]  # RGB to BGR for OpenCV
                vis_info['semantic_frame'] = prev_obs.semantic
                
            # Add goal information
            current_task = self.current_episode["tasks"][self.current_task_idx]
            vis_info['goal_name'] = current_task.get('category', current_task.get('description', 'Unknown'))
            
            # Add other fields
            vis_info['timestep'] = self.timestep
            vis_info['third_person_image'] = None
            vis_info['top_down_map'] = None
            
            try:
                self.visualizer.visualize(**vis_info)
                if self.verbose:
                    print("[VERBOSE]   ✓ Visualization saved")
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")
                if self.verbose:
                    print(f"[VERBOSE]   ✗ Visualization failed: {e}")
            
        # Handle STOP action (parent updates current_task_idx in Habitat task manager)
        if action == DiscreteNavigationAction.STOP:
            if self.verbose:
                print(f"[VERBOSE]   STOP action received - Task {self.current_task_idx + 1} complete")
                
            print(f"\n[TASK {self.current_task_idx + 1} COMPLETE]")
            self.current_task_idx += 1
            
            # Check if all tasks complete
            if self.current_task_idx >= len(self.current_episode["tasks"]):
                if self.verbose:
                    print("[VERBOSE]   All tasks complete - Episode over")
                print(f"\n{'='*60}")
                print("ALL TASKS IN EPISODE COMPLETE!")
                print(f"{'='*60}\n")
                self.episode_over = True
                return True
            else:
                # Reset for next task
                if self.verbose:
                    print(f"[VERBOSE]   Starting next task: {self.current_task_idx + 1}")
                print(f"\n[STARTING TASK {self.current_task_idx + 1}]")
                self.reset_visualization()
                return False
                
        # Record pre-action pose for feedback checking
        pre_action_pose = self.robot.get_base_pose()
        if self.verbose:
            print(f"[VERBOSE]   Pre-action pose: x={pre_action_pose[0]:.3f}, y={pre_action_pose[1]:.3f}, theta={np.degrees(pre_action_pose[2]):.1f}°")
        
        # Convert discrete action to continuous (parent uses HabitatSimActions)
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
        
        if self.verbose:
            print(f"[VERBOSE]   Continuous action: [{continuous_action[0]:.3f}m, {continuous_action[1]:.3f}m, {np.degrees(continuous_action[2]):.1f}°]")
            
        # Execute on robot (replaces parent's Habitat env.step())
        if np.any(continuous_action != 0):
            print(f"→ {action_name}")
            
            try:
                if self.verbose:
                    print("[VERBOSE]   Sending command to robot...")
                    
                # Send command (non-blocking to allow feedback checking)
                self.robot.nav.navigate_to(
                    continuous_action, 
                    relative=True, 
                    blocking=False  # Non-blocking for feedback loop
                )
                
                if self.verbose:
                    print("[VERBOSE]   Command sent, waiting for completion...")
                
                # Feedback loop: wait for action completion
                self._wait_for_action_completion(
                    pre_action_pose, 
                    continuous_action, 
                    action_name,
                    timeout=10.0  # 10 second timeout
                )
                
                if self.verbose:
                    print("[VERBOSE]   ✓ Action completed")
                
            except Exception as e:
                logger.error(f"Navigation failed: {e}")
                if self.verbose:
                    print(f"[VERBOSE]   ✗ Navigation error: {e}")
            
        self.timestep += 1
        rospy.sleep(0.1)  # Brief pause for stability
        
        if self.verbose:
            print("[VERBOSE] apply_action() - Complete\n")
        
        return False
    
    def _wait_for_action_completion(
        self, 
        pre_action_pose: np.ndarray, 
        target_motion: np.ndarray,
        action_name: str,
        timeout: float = 10.0
    ):
        """
        Wait for robot to complete the commanded action with feedback checking
        
        Args:
            pre_action_pose: Pose before action [x, y, theta]
            target_motion: Commanded motion [dx, dy, dtheta]
            action_name: Name of action for logging
            timeout: Maximum time to wait in seconds
        """
        import time
        
        start_time = time.time()
        check_rate = rospy.Rate(10)  # Check at 10 Hz
        
        # Compute expected final pose
        expected_x = pre_action_pose[0] + target_motion[0] * np.cos(pre_action_pose[2])
        expected_y = pre_action_pose[1] + target_motion[0] * np.sin(pre_action_pose[2])
        expected_theta = pre_action_pose[2] + target_motion[2]
        
        # Thresholds for completion
        position_threshold = 0.05  # 5cm
        angle_threshold = np.radians(5)  # 5 degrees
        
        logger.debug(f"Waiting for {action_name} completion...")
        logger.debug(f"Expected pose: x={expected_x:.3f}, y={expected_y:.3f}, theta={np.degrees(expected_theta):.1f}°")
        
        while not rospy.is_shutdown():
            elapsed = time.time() - start_time
            
            # Check timeout
            if elapsed > timeout:
                logger.warning(f"Action timeout after {elapsed:.1f}s")
                break
            
            # Get current pose
            current_pose = self.robot.get_base_pose()
            
            # Calculate error
            position_error = np.sqrt(
                (current_pose[0] - expected_x)**2 + 
                (current_pose[1] - expected_y)**2
            )
            angle_error = abs(self._normalize_angle(current_pose[2] - expected_theta))
            
            # Check if motion complete
            if target_motion[0] != 0:  # Forward motion
                if position_error < position_threshold:
                    logger.debug(f"Forward motion complete: error={position_error:.3f}m")
                    break
            else:  # Rotation
                if angle_error < angle_threshold:
                    logger.debug(f"Rotation complete: error={np.degrees(angle_error):.1f}°")
                    break
            
            # Check if robot stopped moving (velocity-based check)
            # This is a fallback in case odometry is unreliable
            if elapsed > 1.0:  # After 1 second, check if stationary
                time.sleep(0.2)
                new_pose = self.robot.get_base_pose()
                motion = np.linalg.norm(new_pose[:2] - current_pose[:2])
                
                if motion < 0.01:  # Less than 1cm movement
                    logger.debug(f"Robot stationary, assuming action complete")
                    break
            
            check_rate.sleep()
        
        # Final pose check
        final_pose = self.robot.get_base_pose()
        final_position_error = np.sqrt(
            (final_pose[0] - expected_x)**2 + 
            (final_pose[1] - expected_y)**2
        )
        final_angle_error = abs(self._normalize_angle(final_pose[2] - expected_theta))
        
        logger.info(
            f"{action_name} complete: "
            f"pos_error={final_position_error:.3f}m, "
            f"angle_error={np.degrees(final_angle_error):.1f}°, "
            f"time={time.time()-start_time:.2f}s"
        )
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
        
    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get metrics for current episode"""
        return {
            "episode_id": self.episode_id,
            "tasks_completed": self.current_task_idx,
            "total_tasks": len(self.current_episode["tasks"]),
            "timesteps": self.timestep,
        }
        
    def get_robot(self):
        """Get robot client"""
        return self.robot


if __name__ == "__main__":
    """Test the environment"""
    print("="*60)
    print("Scout GOAT Environment Test")
    print("="*60)
    
    rospy.init_node("scout_goat_env_test")
    
    # Create minimal config
    class TestConfig:
        NO_GPU = False
        VISUALIZE = False
        PRINT_IMAGES = False
        ENVIRONMENT = type('obj', (object,), {
            'min_depth': 0.2,
            'max_depth': 5.0,
        })()
        
    config = TestConfig()
    
    # Create environment
    env = ScoutGoatEnv(
        config=config,
        task_config_file="projects/real_world_goat/configs/example_tasks.json"
    )
    
    print("\nTesting observation collection...")
    obs = env.get_observation()
    print(f"RGB shape: {obs.rgb.shape}")
    print(f"Depth shape: {obs.depth.shape}")
    print(f"Semantic shape: {obs.semantic.shape}")
    print(f"GPS: {obs.gps}")
    print(f"Compass: {obs.compass}")
    print(f"Tasks: {len(obs.task_observations['tasks'])}")
    
    print("\nTest complete!")