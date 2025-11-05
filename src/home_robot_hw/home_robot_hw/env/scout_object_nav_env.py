# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Optional
import json

import numpy as np
import rospy

import home_robot
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.utils.geometry import xyt2sophus, xyt_base_to_global

from home_robot_hw.env.visualizer import Visualizer
from home_robot_hw.remote import ScoutClient


class SimpleSemanticCategoryMapping:
    """Minimal semantic category mapping for Scout environment."""
    
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.num_sem_categories = len(vocabulary)
        
        # Create color palette for visualization
        np.random.seed(42)
        self.map_color_palette = [
            tuple(np.random.randint(0, 255, 3).tolist()) 
            for _ in range(self.num_sem_categories)
        ]
    
    def get_category_name(self, idx):
        """Get category name by index."""
        if 0 <= idx < len(self.vocabulary):
            return self.vocabulary[idx]
        return "unknown"
    
    def get_category_id(self, name):
        """Get category index by name."""
        try:
            return self.vocabulary.index(name)
        except ValueError:
            return -1


class ScoutObjectNavEnv:
    """Create a Detic-based object nav environment for the Scout robot."""

    def __init__(
        self, 
        config=None, 
        cat_map_file=None,
        forward_step=0.25, 
        rotate_step=30.0, 
        *args, 
        **kwargs
    ):
        
        # Store config and category map file
        self.config = config
        self.cat_map_file = cat_map_file
        
        # Load categories from file or use defaults
        if cat_map_file is not None:
            self.goal_options = self._load_categories_from_file(cat_map_file)
        else:
            # Default categories
            REAL_WORLD_CATEGORIES = [
                "other",
                "cup",
                "chair",
                "bottle",
            ]
            self.goal_options = REAL_WORLD_CATEGORIES
        
        # Build semantic category mapping for GoatAgent
        self._semantic_category_mapping = SimpleSemanticCategoryMapping(self.goal_options)
        
        self.forward_step = forward_step  # in meters
        self.rotate_step = np.radians(rotate_step)

        # Initialize Detic perception with vocabulary
        # Use ALL objects from category map for detection
        detic_vocab = ",".join([cat for cat in self.goal_options if cat != "other"])
        print(f"[ENV] Detic vocabulary size: {len(detic_vocab.split(','))}")
        print(f"[ENV] Sample Detic vocab: {detic_vocab.split(',')[:10]}")
        
        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary=detic_vocab,
            sem_gpu_id=0,
        )
        
        if config is not None:
            self.visualizer = Visualizer(config)
        else:
            self.visualizer = None

        # Create a robot client to interface with the Scout
        self.robot = ScoutClient()
        
        # Episode state
        self._episode_over = False
        self._episode_metrics = {}
        
        self.reset()

    def _load_categories_from_file(self, cat_map_file):
        """Load category vocabulary from JSON file, preserving exact ID mapping."""
        try:
            with open(cat_map_file, 'r') as f:
                category_map = json.load(f)
            
            # Check for format with category_to_id mappings
            if 'obj_category_to_obj_category_id' in category_map:
                obj_mapping = category_map['obj_category_to_obj_category_id']
                rec_mapping = category_map.get('recep_category_to_recep_category_id', {})
                
                # In your format, receptacles and objects share the SAME ID space
                # We need to merge them into a single vocabulary array
                # ID 0 = "other" (reserved)
                # IDs 1+ = objects and receptacles at their original IDs
                
                # Find max ID
                all_ids = list(obj_mapping.values()) + list(rec_mapping.values())
                max_id = max(all_ids) if all_ids else 0
                
                # Create vocabulary: index = ID directly (no offset!)
                # Size = max_id + 1 (to include ID 0)
                vocabulary = ["other"] * (max_id + 1)
                vocabulary[0] = "other"
                
                # Fill in objects at their exact IDs
                for obj_name, obj_id in obj_mapping.items():
                    vocabulary[obj_id] = obj_name
                
                # Fill in receptacles at their exact IDs  
                for rec_name, rec_id in rec_mapping.items():
                    vocabulary[rec_id] = rec_name
                
                print(f"[ENV] Loaded vocabulary with {len(vocabulary)} indices")
                print(f"[ENV] Objects: {len(obj_mapping)}, Receptacles: {len(rec_mapping)}")
                
                # Print mapping for "cup" specifically
                if 'cup' in obj_mapping:
                    cup_id = obj_mapping['cup']
                    print(f"[ENV] ⚠️  IMPORTANT: 'cup' has ID {cup_id} in category_map")
                    print(f"[ENV]              'cup' is at vocabulary[{cup_id}] = '{vocabulary[cup_id]}'")
                    if cup_id < len(vocabulary):
                        assert vocabulary[cup_id] == 'cup', f"Mismatch! vocabulary[{cup_id}] = {vocabulary[cup_id]}"
                        print(f"[ENV] ✅ Vocabulary mapping is CORRECT")
                
            elif 'objects' in category_map:
                # Old simple format
                vocabulary = ["other"]
                for obj in category_map['objects']:
                    if obj not in vocabulary:
                        vocabulary.append(obj)
                if 'receptacles' in category_map:
                    for rec in category_map['receptacles']:
                        if rec not in vocabulary:
                            vocabulary.append(rec)
                print(f"[ENV] Loaded {len(vocabulary)} categories (simple list format)")
            
            return vocabulary
            
        except Exception as e:
            print(f"[WARNING] Could not load categories from {cat_map_file}: {e}")
            import traceback
            traceback.print_exc()
            print("[WARNING] Using default categories")
            return ["other", "cup", "chair", "bottle"]

    def get_semantic_category_mapping(self):
        """
        Get the semantic category mapping for the environment.
        Required by GoatAgent-based agents.
        
        Returns:
            SimpleSemanticCategoryMapping: The semantic category mapping object
        """
        return self._semantic_category_mapping

    def reset(self):
        """Reset the environment for a new episode."""
        self.sample_goal()
        self._episode_start_pose = xyt2sophus(self.robot.get_base_pose())
        self._episode_over = False
        self._episode_metrics = {}
        
        if self.visualizer is not None:
            self.visualizer.reset()
        
        print(f"[ENV] Episode reset - Goal: {self.current_goal_name}")

    def apply_action(
        self,
        action: Action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[Observations] = None,
    ):
        """Apply discrete action to the robot."""
        if self.visualizer is not None and info is not None:
            try:
                self.visualizer.visualize(**info)
            except Exception as e:
                print(f"[WARNING] Visualization failed: {e}")
        
        continuous_action = np.zeros(3)
        if action == DiscreteNavigationAction.MOVE_FORWARD:
            print("[ACTION] FORWARD")
            continuous_action[0] = self.forward_step
        elif action == DiscreteNavigationAction.TURN_RIGHT:
            print("[ACTION] TURN RIGHT")
            continuous_action[2] = -self.rotate_step
        elif action == DiscreteNavigationAction.TURN_LEFT:
            print("[ACTION] TURN LEFT")
            continuous_action[2] = self.rotate_step
        elif action == DiscreteNavigationAction.STOP:
            print("[ACTION] STOP")
            self._episode_over = True
            return True
        else:
            print(f"[ACTION] Unknown action: {action}")
            pass

        if not np.allclose(continuous_action, 0):
            # Execute the movement
            try:
                self.robot.nav.navigate_to(
                    continuous_action, relative=True, blocking=True
                )
            except Exception as e:
                print(f"[ERROR] Navigation failed: {e}")
        
        rospy.sleep(0.5)
        return self._episode_over

    def set_goal(self, goal):
        """Set a goal as a string."""
        if goal in self.goal_options:
            self.current_goal_id = self.goal_options.index(goal)
            self.current_goal_name = goal
            print(f"[ENV] Goal set to: {goal} (id: {self.current_goal_id})")
            return True
        else:
            print(f"[WARNING] Goal '{goal}' not in vocabulary")
            return False

    def sample_goal(self):
        """Set a random goal (excluding 'other')."""
        # Skip index 0 which is "other"
        if len(self.goal_options) > 1:
            idx = np.random.randint(1, len(self.goal_options))
        else:
            idx = 0
        self.current_goal_id = idx
        self.current_goal_name = self.goal_options[idx]
        print(f"[ENV] Sampled goal: {self.current_goal_name} (id: {self.current_goal_id})")

    def get_observation(self) -> Observations:
        """Get observation from the robot with all fields required by GoatAgent."""
        # Get sensor data from robot
        rgb, depth, _ = self.robot.get_images(compute_xyz=True, rotate_images=False)
        current_pose = xyt2sophus(self.robot.get_base_pose())

        # Calculate relative pose from episode start
        relative_pose = self._episode_start_pose.inverse() * current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]
        
        # GPS in robot coordinates (relative to episode start)
        gps = relative_pose.translation()[:2]

        # Create the observation with all required fields
        obs = home_robot.core.interfaces.Observations(
            rgb=rgb.copy(),
            depth=depth.copy(),
            gps=gps,
            compass=np.array([theta]),
            task_observations={
                "goal_id": self.current_goal_id,
                "goal_name": self.current_goal_name,
                "object_goal": self.current_goal_id,
                
                # For GoatAgent compatibility - tasks list
                "tasks": [{
                    "type": "objectnav",
                    "semantic_id": self.current_goal_id,
                    "description": self.current_goal_name,
                }],
            },
            camera_pose=None,
            third_person_image=None,
        )
        
        # Run Detic segmentation
        obs = self.segmentation.predict(obs, depth_threshold=0.5)
        
        # Map zero-class to 'other' category (first in vocabulary)
        obs.semantic[obs.semantic == 0] = 0  # Keep as 'other'
        
        # Add instance map for GoatAgent (use semantic as proxy)
        obs.task_observations["instance_map"] = obs.semantic.copy()
        obs.task_observations["instance_frame"] = obs.semantic.copy()
        
        # Add semantic frame visualization
        obs.task_observations["semantic_frame"] = None
        
        return obs

    @property
    def episode_over(self) -> bool:
        """Determines if the episode is over."""
        return self._episode_over

    def get_episode_metrics(self) -> Dict:
        """Returns metrics for the current episode."""
        # Calculate additional metrics if needed
        if hasattr(self, '_episode_start_pose'):
            try:
                current_pose = xyt2sophus(self.robot.get_base_pose())
                relative_pose = self._episode_start_pose.inverse() * current_pose
                distance_traveled = np.linalg.norm(relative_pose.translation()[:2])
                
                self._episode_metrics.update({
                    "distance_traveled": float(distance_traveled),
                    "goal_name": self.current_goal_name,
                    "success": self._episode_over,
                })
            except Exception as e:
                print(f"[WARNING] Could not calculate metrics: {e}")
        
        return self._episode_metrics

    def get_robot(self):
        """Returns the robot client object."""
        return self.robot


if __name__ == "__main__":
    # Test the environment
    print("="*60)
    print("Scout Object Navigation Environment Test")
    print("="*60)
    
    rospy.init_node("scout_object_nav_test")
    
    # Test with category map file
    rob = ScoutObjectNavEnv(
        cat_map_file="projects/real_world_ovmm/configs/example_cat_map.json"
    )
    
    # Test semantic category mapping
    mapping = rob.get_semantic_category_mapping()
    print(f"\n[TEST] Number of categories: {mapping.num_sem_categories}")
    print(f"[TEST] Vocabulary: {mapping.vocabulary}")

    # Test observation
    print("\n[TEST] Getting observation...")
    obs = rob.get_observation()
    print(f"[TEST] RGB shape: {obs.rgb.shape}")
    print(f"[TEST] Depth shape: {obs.depth.shape}")
    print(f"[TEST] Semantic unique values: {np.unique(obs.semantic)}")
    print(f"[TEST] Has instance_map: {'instance_map' in obs.task_observations}")
    print(f"[TEST] Has tasks: {'tasks' in obs.task_observations}")
    print(f"[TEST] Goal: {obs.task_observations['goal_name']}")
    
    print("\n[TEST] Environment test completed successfully!")
    print("="*60)
    
    # Interactive mode
    import matplotlib.pyplot as plt
    
    print("\nEntering interactive mode...")
    print("Commands: 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT, q=quit")
    
    while not rospy.is_shutdown():
        cmd = None
        try:
            user_input = input("\nEnter command: ")
            if user_input.lower() == 'q':
                break
            cmd = DiscreteNavigationAction(int(user_input))
        except (ValueError, KeyError):
            print("Invalid command. Use 0-3 or 'q' to quit.")
            continue
        
        if cmd is not None:
            done = rob.apply_action(cmd)
            if done:
                print("Episode ended!")
                rob.reset()

        obs = rob.get_observation()
        
        # Visualization
        depth_vis = obs.depth.copy()
        depth_vis[depth_vis > 5] = 0
        
        plt.clf()
        plt.subplot(131)
        plt.imshow(obs.rgb)
        plt.title("RGB")
        
        plt.subplot(132)
        plt.imshow(depth_vis)
        plt.title("Depth")
        
        plt.subplot(133)
        plt.imshow(obs.semantic, cmap='tab20')
        plt.title("Semantic")
        
        print(f"Compass: {obs.compass}, GPS: {obs.gps}")
        
        plt.pause(0.1)