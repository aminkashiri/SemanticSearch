# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy

    
class ControlMode:
    NAVIGATION = "navigation"
    IDLE = "idle"

from .modules.nav import ScoutNavigationClient
from .ros import ScoutRosInterface


class ScoutClient:
    """Defines a ROS-based interface to a Scout mobile robot. Collects observations and commands the robot for navigation."""

    def __init__(
        self,
        init_node: bool = True,
        camera_overrides: Optional[Dict] = None,
    ):
        """
        Create an interface into ROS execution here. This one needs to connect to:
        - sensor topics to read current data
        - tf for SLAM
        - /cmd_vel for navigation commands
        """
        # Ros
        if init_node and not rospy.core.is_initialized():
            rospy.init_node("scout_user_client")

        if camera_overrides is None:
            camera_overrides = {}
            
        self._ros_client = ScoutRosInterface(**camera_overrides)

        # Interface modules - only need navigation
        self.nav = ScoutNavigationClient(self._ros_client, None)
        
        # Init control mode
        self._base_control_mode = ControlMode.NAVIGATION
        
        # Initially start in navigation mode
        self.switch_to_navigation_mode()

    @property
    def model(self):
        # Scout does not have a complex kinematic model like Stretch
        return None

    # Mode interfaces - only navigation is supported
    def switch_to_navigation_mode(self):
        """Switch Scout to navigation control."""
        result = self.nav.enable()
        self._base_control_mode = ControlMode.NAVIGATION
        return result

    def in_manipulation_mode(self):
        # Scout has no manipulation capabilities
        return False

    def in_navigation_mode(self):
        return self._base_control_mode == ControlMode.NAVIGATION

    # General control methods
    def wait(self):
        self.nav.wait()

    def reset(self):
        self.stop()
        self.nav.home()
        self.stop()

    def stop(self):
        self.nav.disable()
        self._base_control_mode = ControlMode.IDLE

    # Observation interfaces
    def get_base_pose(self):
        """Get the current base pose of the robot from ROS TF."""
        # return self._ros_client.get_base_pose()
        return self._ros_client.get_robot_center_pose()

    def get_images(self, compute_xyz: bool, rotate_images: bool) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get RGB, depth, and optional XYZ point cloud from the robot's camera."""
        
        rgb, depth, xyz_full = self._ros_client.get_images()
        
        xyz = None
        if compute_xyz:
            # Only return the XYZ image if compute_xyz is True
            xyz = xyz_full
            
        return rgb, depth, xyz
    
    def get_joint_state(self):
        return None, None, None