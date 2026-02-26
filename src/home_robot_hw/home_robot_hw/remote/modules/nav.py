# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Iterable

import rospy
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBoolRequest, TriggerRequest

from home_robot.motion.robot import Robot
from home_robot.utils.geometry import sophus2xyt, xyt2sophus, xyt_base_to_global
from home_robot_hw.constants import T_LOC_STABILIZE
from home_robot_hw.ros.utils import matrix_to_pose_msg

from .abstract import AbstractControlModule, enforce_enabled
import numpy as np

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
        # return self._ros_client.get_base_pose()
        return self._ros_client.get_robot_center_pose()

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
    def navigate_to(self, xyt, relative=True, blocking=True):
        # 1. Capture Initial State
        start_pose = self.get_base_pose()
        x0, y0, th0 = start_pose[0], start_pose[1], start_pose[2]
        
        target_dist = abs(xyt[0])  # Linear goal (0.25m)
        target_ang = abs(xyt[2])   # Angular goal (rad)
        
        # 2. PID/P Gains (Tune these based on Scout's responsiveness)
        Kp_linear = 2   
        Kp_angular = 2.0
        
        # Velocity Limits 
        min_v, max_v = 0.025, 0.5  # m/s
        min_w, max_w = 0.025, 1.0 # rad/s
        
        # Tolerance: Stop when within 1cm or 1 degree
        linear_tol = 0.0005 
        angular_tol = np.deg2rad(0.3)
        
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            curr_pose = self.get_base_pose()
            curr_x, curr_y, curr_th = curr_pose[0], curr_pose[1], curr_pose[2]

            # 3. Calculate Error (Distance Remaining)
            dist_moved = np.sqrt((curr_x - x0)**2 + (curr_y - y0)**2)
            ang_moved = abs(np.arctan2(np.sin(curr_th - th0), np.cos(curr_th - th0)))
            
            error_v = target_dist - dist_moved
            error_w = target_ang - ang_moved

            cmd = Twist()

            # 4. Control Logic
            if target_dist > 0 and error_v > linear_tol:
                # Proportional output
                v_out = error_v * Kp_linear
                # Clamp between min (to prevent stall) and max (to prevent jerking)
                cmd.linear.x = np.clip(v_out, min_v, max_v)
            
            elif target_ang > 0 and error_w > angular_tol:
                w_out = error_w * Kp_angular
                # Scout turns require more torque, so min_w is higher than min_v
                cmd.angular.z = np.clip(w_out, min_w, max_w) * np.sign(xyt[2])
            
            else:
                # Goal Reached
                break

            self._ros_client.velocity_pub.publish(cmd)
            rate.sleep()

        # 5. Final Stop
        self._ros_client.velocity_pub.publish(Twist())
        rospy.loginfo("Navigation goal reached and stopped.")

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