# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Iterable, Tuple

import rospy
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBoolRequest, TriggerRequest

from home_robot.motion.robot import Robot
from home_robot.utils.geometry import sophus2xyt, xyt2sophus, xyt_base_to_global
from home_robot_hw.constants import T_LOC_STABILIZE
from home_robot_hw.ros.utils import matrix_to_pose_msg

from .abstract import AbstractControlModule, enforce_enabled


class ScoutNavigationClient(AbstractControlModule):
    """
    A simplified navigation client for the Scout mobile robot.
    This class handles basic navigation commands by publishing to the /cmd_vel topic.
    It does not rely on a sophisticated "goto" controller like the Stretch, as the Scout's
    kinematics are much simpler.
    """

    def __init__(self, ros_client, robot_model: Robot = None):
        super().__init__()

        self._ros_client = ros_client
        # Scout does not have a complex kinematic model, so robot_model is not used here.
        # self._robot_model = robot_model
        self._wait_for_pose()

    # Enable / disable
    def _enable_hook(self) -> bool:
        """Called when interface is enabled. For Scout, this is a no-op."""
        return True

    def _disable_hook(self) -> bool:
        """Called when interface is disabled. For Scout, this is a no-op."""
        return True

    # Interface methods
    def get_base_pose(self):
        """get the latest base pose from sensors"""
        return self._ros_client.get_base_pose()

    def at_goal(self) -> bool:
        """
        For a simplified navigation client, this would be handled by the agent.
        The client does not have a "goal reached" flag.
        """
        # This functionality should be handled by the agent.
        return False

    @enforce_enabled
    def set_velocity(self, v, w):
        """
        Directly sets the linear and angular velocity of robot base.
        This is kept for completeness, but the main agent control loop 
        uses navigate_to.
        """
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w

        # For Scout, we directly publish to the velocity topic.
        print(f"Setting velocity: linear={v:.2f} m/s, angular={w:.2f} rad/s")
        self._ros_client.velocity_pub.publish(msg)

    @enforce_enabled
    def navigate_to(
        self,
        xyt: Iterable[float],
        relative: bool = False,
        position_only: bool = False,
        avoid_obstacles: bool = False,
        blocking: bool = True,
    ):
        """
        FIX: For this simplified velocity-control model, we treat the input XYT 
        as the immediate (v, w) velocity command to execute.
        """
        # Parse inputs
        assert len(xyt) == 3, "Input goal location must be of length 3."

        if avoid_obstacles:
            raise NotImplementedError("Obstacle avoidance unavailable.")

        # Extract linear velocity (v) from x and angular velocity (w) from theta
        v = xyt[0]  # Linear velocity (x)
        w = xyt[2]  # Angular velocity (theta)
        
        # Create the Twist message
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w

        # Publish the velocity command and print the debug statement
        print(f"Setting velocity: linear={v:.2f} m/s, angular={w:.2f} rad/s")
        self._ros_client.velocity_pub.publish(msg)
        rospy.sleep(0.5) 

        rospy.loginfo(f"Navigation to {xyt} requested.")
        rospy.loginfo("Note: ScoutNavigationClient does not block. Agent must handle navigation loop.")

    @enforce_enabled
    def home(self):
        """Sends a command to navigate to the origin [0, 0, 0]."""
        # Sending [0, 0, 0] is a STOP command in this velocity control context
        self.navigate_to([0.0, 0.0, 0.0], blocking=True)

    # Helper methods
    def _wait_for_pose(self):
        """Wait until we have an accurate pose estimate."""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self._ros_client.se3_base_filtered is not None:
                break
            rate.sleep()