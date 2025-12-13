# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import ros_numpy
import rospy
import sophuspy as sp
import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest

# NOTE: Assuming home_robot.utils.pose and home_robot_hw.ros.utils are correct
from home_robot.utils.pose import to_matrix
from home_robot_hw.constants import ControlMode
from home_robot_hw.ros.camera import RosCamera
from home_robot_hw.ros.utils import matrix_from_pose_msg
from home_robot_hw.ros.visualizer import Visualizer

# Assuming RosCamera class definition is available from home_robot_hw.ros.camera
# Assuming Visualizer class definition is available from home_robot_hw.ros.visualizer

DEFAULT_COLOR_TOPIC = "/camera/camera/color"
DEFAULT_DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color"


def get_xyz_image_from_depth(depth, fx, fy, px, py):
    """Calculates the XYZ image from the depth map and intrinsics."""
    height, width = depth.shape
    
    # Create normalized coordinates
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    u_map, v_map = np.meshgrid(x_coords, y_coords)
    
    # Calculate X, Y, Z components
    Z = depth
    X = (u_map - px) * Z / fx
    Y = (v_map - py) * Z / fy
    
    # Stack to form the (H, W, 3) XYZ image
    xyz = np.stack([X, Y, Z], axis=-1)
    
    return xyz

class ScoutRosInterface:
    """Interface object with ROS topics and services for the Scout robot."""

    # Base of the robot
    base_link = "base_link"
    
    # ROS-specific variables
    goal_time_tolerance = 1.0
    msg_delay_t = 0.1
    dof = 3  # Degrees of freedom for a mobile base: x, y, theta

    def __init__(
        self,
        init_cameras: bool = True,
        color_topic: Optional[str] = None,
        depth_topic: Optional[str] = None,
        depth_buffer_size: Optional[int] = None,
    ):
        # Initialize caches
        self.current_mode: Optional[str] = None

        self.se3_base_filtered: Optional[sp.SE3] = None
        self.se3_base_odom: Optional[sp.SE3] = None

        self.last_odom_update_timestamp = rospy.Time(0)
        self.last_base_update_timestamp = rospy.Time(0)
        self._goal_reset_t = rospy.Time(0)

        # Create visualizers for pose information
        self.goal_visualizer = Visualizer("command_pose", rgba=[1.0, 0.0, 0.0, 0.5])
        self.curr_visualizer = Visualizer("current_pose", rgba=[0.0, 0.0, 1.0, 0.5])

        # Initialize ros communication
        self._create_pubs_subs()

        self._color_topic = DEFAULT_COLOR_TOPIC if color_topic is None else color_topic
        self._depth_topic = DEFAULT_DEPTH_TOPIC if depth_topic is None else depth_topic
        self._depth_buffer_size = depth_buffer_size

        self.rgb_cam, self.dpt_cam = None, None
        if init_cameras:
            self._create_cameras()
            self._wait_for_cameras()

    # Interfaces
    def recent_depth_image(self, seconds: float, print_delay_timers: bool = False) -> bool:
        """Return true if we have up-to-date depth."""
        if print_delay_timers:
            print(
                " - 1",
                (rospy.Time.now() - self._goal_reset_t).to_sec(),
                self.msg_delay_t,
            )
            print(
                " - 2", (self.dpt_cam.get_time() - self._goal_reset_t).to_sec(), seconds
            )
        if (
            self._goal_reset_t is not None
            and (rospy.Time.now() - self._goal_reset_t).to_sec() > self.msg_delay_t
        ):
            return (self.dpt_cam.get_time() - self._goal_reset_t).to_sec() > seconds
        else:
            return False

    def get_base_pose(self) -> np.ndarray:
        """Get the latest filtered base pose as a 3D numpy array [x, y, yaw]."""
        if self.se3_base_filtered is None:
            rospy.logwarn_throttle(
                1, "Base pose not yet received from state_estimator/pose_filtered"
            )
            return np.array([0, 0, 0], dtype=np.float32)
        pose_matrix = self.se3_base_filtered.matrix()
        # Extract yaw from the rotation matrix
        theta = np.arctan2(pose_matrix[1, 0], pose_matrix[0, 0])
        return np.array([pose_matrix[0, 3], pose_matrix[1, 3], theta])

    # --- FIX APPLIED HERE ---
    def get_images(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get RGB, depth, and XYZ point cloud from the robot's cameras.
           FIXED: Now returns all three components (rgb, depth, xyz).
        """
        
        # 1. Get RGB and Depth images
        rgb, depth = self.rgb_cam.get(), self.dpt_cam.get()
        
        # 2. Get camera intrinsics from the depth camera object
        # The self.dpt_cam object is an instance of RosCamera
        info = self.dpt_cam.get_info()
        fx, fy, px, py = info["fx"], info["fy"], info["px"], info["py"]
        
        # 3. Calculate the XYZ image
        xyz = get_xyz_image_from_depth(depth, fx, fy, px, py)
        
        return rgb, depth, xyz # <--- Corrected to return all three
    # --- END FIX ---

    # Helper functions
    def _create_pubs_subs(self):
        """Create ROS publishers and subscribers."""
        # Create the tf2 buffer
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

        # Create command publishers
        self.velocity_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # Create subscribers
        self._odom_sub = rospy.Subscriber(
            "/dlio/odom_node/odom",
            Odometry,
            self._odom_callback,
            queue_size=1,
        )
        self._base_state_sub = rospy.Subscriber(
            "/dlio/odom_node/pose",
            PoseStamped,
            self._base_state_callback,
            queue_size=1,
        )
        print("Waiting for ROS topics...")
        rospy.wait_for_message("/dlio/odom_node/odom", Odometry, timeout=10.0)
        rospy.wait_for_message("/dlio/odom_node/pose", PoseStamped, timeout=10.0)
        print("...connected to ROS topics.")

    def _create_cameras(self):
        if self.rgb_cam is not None or self.dpt_cam is not None:
            raise RuntimeError("Already created cameras")
        print("Creating cameras...")
        self.rgb_cam = RosCamera(self._color_topic)
        self.dpt_cam = RosCamera(
            self._depth_topic,
            buffer_size=self._depth_buffer_size,
        )
        self.filter_depth = self._depth_buffer_size is not None

    def _wait_for_cameras(self):
        if self.rgb_cam is None or self.dpt_cam is None:
            raise RuntimeError("cameras not initialized")
        print("Waiting for RGB camera images...")
        self.rgb_cam.wait_for_image()
        print("Waiting for depth camera images...")
        self.dpt_cam.wait_for_image()
        print("..done.")
        print("rgb frame =", self.rgb_cam.get_frame())
        print("dpt frame =", self.dpt_cam.get_frame())
        if self.rgb_cam.get_frame() != self.dpt_cam.get_frame():
            raise RuntimeError("issue with camera setup; depth and rgb not aligned")

    # Rostopic callbacks
    def _odom_callback(self, msg: Odometry):
        """odometry callback"""
        self._last_odom_update_timestamp = msg.header.stamp
        self.se3_base_odom = sp.SE3(matrix_from_pose_msg(msg.pose.pose))

    def _base_state_callback(self, msg: PoseStamped):
        """base state updates from SLAM system"""
        self._last_base_update_timestamp = msg.header.stamp
        self.se3_base_filtered = sp.SE3(matrix_from_pose_msg(msg.pose))
        self.curr_visualizer(self.se3_base_filtered.matrix())

    def get_frame_pose(self, frame, base_frame=None, lookup_time=None, timeout_s=None):
        """look up a particular frame in base coords (or some other coordinate frame)."""
        if lookup_time is None:
            lookup_time = rospy.Time(0)  # return most recent transform
        if timeout_s is None:
            timeout_ros = rospy.Duration(0.1)
        else:
            timeout_ros = rospy.Duration(timeout_s)
        if base_frame is None:
            base_frame = self.base_link
        try:
            stamped_transform = self.tf2_buffer.lookup_transform(
                base_frame, frame, lookup_time, timeout_ros
            )
            pose_mat = ros_numpy.numpify(stamped_transform.transform)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            print("!!! Lookup failed from", base_frame, "to", frame, "!!!")
            return None
        return pose_mat