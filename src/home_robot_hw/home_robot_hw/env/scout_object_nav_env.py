# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Optional

import numpy as np
import rospy

import home_robot
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.utils.geometry import xyt2sophus, xyt_base_to_global

from home_robot_hw.env.visualizer import Visualizer
# The following Stretch-specific import has been replaced by a generic client:
from home_robot_hw.remote import ScoutClient



class ScoutObjectNavEnv:
    """Create a Detic-based object nav environment for the Scout robot."""

    def __init__(
        self, config=None, forward_step=0.25, rotate_step=30.0, *args, **kwargs
    ):
        
        # TODO: pass this in or load from cfg
        REAL_WORLD_CATEGORIES = [
            "other",
            "cup",
            "other",
        ]
        self.goal_options = REAL_WORLD_CATEGORIES
        self.forward_step = forward_step  # in meters
        self.rotate_step = np.radians(rotate_step)

        # TODO Specify confidence threshold as a parameter
        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary=",".join(self.goal_options),
            sem_gpu_id=0,
        )
        if config is not None:
            self.visualizer = Visualizer(config)
        else:
            self.visualizer = None

        # Create a robot client to interface with the Scout.
        self.robot = ScoutClient()
        self.reset()

    def reset(self):
        """Reset the environment for a new episode."""
        self.sample_goal()
        self._episode_start_pose = xyt2sophus(self.robot.get_base_pose())
        if self.visualizer is not None:
            self.visualizer.reset()

    def apply_action(
        self,
        action: Action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[Observations] = None,
    ):
        """Discrete action space. make predictions for where the robot should go, move by a fixed
        amount forward or rotationally."""
        if self.visualizer is not None:
            self.visualizer.visualize(**info)
        continuous_action = np.zeros(3)
        if action == DiscreteNavigationAction.MOVE_FORWARD:
            print("FORWARD")
            continuous_action[0] = self.forward_step
        elif action == DiscreteNavigationAction.TURN_RIGHT:
            print("TURN RIGHT")
            continuous_action[2] = -self.rotate_step
        elif action == DiscreteNavigationAction.TURN_LEFT:
            print("TURN LEFT")
            continuous_action[2] = self.rotate_step
        else:
            # Do nothing if "stop"
            pass

        if continuous_action is not None:
            # In a real implementation, this would publish to /cmd_vel
            self.robot.nav.navigate_to(
                continuous_action, relative=True, blocking=True
            )
        rospy.sleep(0.5)

    def set_goal(self, goal):
        """set a goal as a string"""
        if goal in self.goal_options:
            self.current_goal_id = self.goal_options.index(goal)
            self.current_goal_name = goal
            return True
        else:
            return False

    def sample_goal(self):
        """set a random goal"""
        # idx = np.random.randint(len(self.goal_options) - 2) + 1
        idx = 1
        self.current_goal_id = idx
        self.current_goal_name = self.goal_options[idx]

    def get_observation(self) -> Observations:
        """Get Detic and rgb/xyz/theta from the robot."""
        # NOTE: get_images returns (rgb, depth, xyz) now due to previous patches
        rgb, depth, _ = self.robot.get_images(compute_xyz=True, rotate_images=False)
        current_pose = xyt2sophus(self.robot.get_base_pose())

        # use sophus to get the relative translation
        relative_pose = self._episode_start_pose.inverse() * current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]
        
        # GPS in robot coordinates
        gps = relative_pose.translation()[:2]

        # Create the observation
        obs = home_robot.core.interfaces.Observations(
            rgb=rgb.copy(),
            depth=depth.copy(),
            gps=gps,
            compass=np.array([theta]),
            task_observations={
                "goal_id": self.current_goal_id,
                "goal_name": self.current_goal_name,
                "object_goal": self.current_goal_id, # Redundant, but kept for compatibility
                "recep_goal": self.current_goal_id, # Redundant, but kept for compatibility
                
                # --- PATCH: ADD/CORRECT KEYS EXPECTED BY OvmmPerception ---
                # The object name is mapped to BOTH "goal_name" (for agent) and 
                # "object_name" (for ovmm_perception's _process_obs)
                "object_name": self.current_goal_name, 
                
                # Receptacles are not part of pure ObjectNav, so set to None
                "start_recep_name": None,
                "place_recep_name": None, # OvmmPerception uses "place_recep_name" for end goal
                # --- END PATCH ---
            },
            # camera_pose=self.get_camera_pose_matrix(rotated=True),
        )
        # Run the segmentation model here
        obs = self.segmentation.predict(obs, depth_threshold=0.5)
        obs.semantic[obs.semantic == 0] = len(self.goal_options) - 1
        return obs

    @property
    def episode_over(self) -> bool:
        """Determines if the episode is over."""
        # For a nav environment, you might check if the goal is within range.
        # This implementation is a placeholder.
        return False

    def get_episode_metrics(self) -> Dict:
        """Returns metrics for the current episode."""
        return {}

    def get_robot(self):
        """Returns the robot client object."""
        return self.robot


if __name__ == "__main__":
    # Create the robot
    print("----------------")
    print("Start example - Scout robot")
    rospy.init_node("scout_object_nav_test")
    print("Create ROS interface")
    rob = ScoutObjectNavEnv()

    # Debug the observation space
    import matplotlib.pyplot as plt

    while not rospy.is_shutdown():
        cmd = None
        try:
            cmd = input("Enter a number 0-3 for action:")
            cmd = DiscreteNavigationAction(int(cmd))
        except ValueError:
            cmd = None
        if cmd is not None:
            rob.apply_action(cmd)

        obs = rob.get_observation()
        rgb, depth = obs.rgb, obs.depth

        # Add a visualiztion for debugging
        depth[depth > 5] = 0
        plt.subplot(121)
        plt.imshow(rgb)
        plt.subplot(122)
        plt.imshow(depth)
        
        print("----------------")
        print("Observation values:")
        print("RGB =", np.unique(rgb))
        print("Depth =", np.unique(depth))
        print("Compass =", obs.compass)
        print("Gps =", obs.gps)
        plt.show()