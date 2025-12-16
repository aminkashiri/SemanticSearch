# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import cv2
import math
import skfmm
import shutil
import numpy as np
import skimage.morphology
from typing import List, Tuple
from scipy.ndimage import label
from bresenham import bresenham
from skimage.draw import polygon
import home_robot.utils.pose as pu
from .fmm_planner import FMMPlanner
from scipy.spatial import ConvexHull
from home_robot.core.interfaces import (
    ContinuousNavigationAction,
    DiscreteNavigationAction,
)
from home_robot.utils.visualization import (
    visualize_map,
    visualize_distance_frontiers,
    visualize_semantic_frontiers,
)
from home_robot.utils.logger import get_logger
from scipy.ndimage import distance_transform_edt
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory


script_dir = os.path.dirname(
    os.path.abspath(__file__)
)  # Directory of the current script
CO_LOCATION_WEIGHTS = np.load(os.path.join(script_dir, f"co_location_matrix.npy"))


def add_boundary(mat: np.ndarray, value=1) -> np.ndarray:
    h, w = mat.shape
    new_mat = np.zeros((h + 2, w + 2)) + value
    new_mat[1 : h + 1, 1 : w + 1] = mat
    return new_mat


def remove_boundary(mat: np.ndarray, value=1) -> np.ndarray:
    return mat[value:-value, value:-value]


class DiscretePlanner:
    """
    This class translates planner inputs into a discrete low-level action
    using an FMM planner.

    This is a wrapper used to navigate to a particular object/goal location.
    """

    def __init__(
        self,
        turn_angle: float,
        collision_threshold: float,
        step_size: int,
        obs_dilation_selem_radius: int,
        map_size_cm: int,
        map_resolution: int,
        visualize: bool,
        print_images: bool,
        dump_location: str,
        exp_name: str,
        min_goal_distance_cm: float = 50.0,
        min_obs_dilation_selem_radius: int = 1,
        map_downsample_factor: float = 1.0,
        map_update_frequency: int = 1,
        goal_tolerance: float = 5,  # for sim
        discrete_actions: bool = True,
        continuous_angle_tolerance: float = 30.0,
        panorama_start_steps: int = 0,
        semantic_map: Categorical2DSemanticMapState = None,
        instance_memory: InstanceMemory = None,
        goal_filtering=False,
        frontier_metric: str = "distance",
        agent_id=None,
    ):
        """
        Similar to old DiscretePlanner, but with changes to:
        - How to go to past pose.
        - How to replan if planning failed.
        Arguments:
            turn_angle (float): agent turn angle (in degrees)
            collision_threshold (float): forward move distance under which we
             consider there's a collision (in meters)
            obs_dilation_selem_radius: radius (in cells) of obstacle dilation
             structuring element
            obs_dilation_selem_radius: radius (in cells) of goal dilation
             structuring element
            map_size_cm: global map size (in centimeters)
            map_resolution: size of map bins (in centimeters)
            print_images: if True, save visualization as images
        """
        self.discrete_actions = discrete_actions
        self.print_images = print_images
        self.default_vis_dir = f"{dump_location}/images/{exp_name}"
        os.makedirs(self.default_vis_dir, exist_ok=True)

        self.map_size_cm = map_size_cm
        self.map_resolution = map_resolution
        self.map_shape = (
            self.map_size_cm // self.map_resolution,
            self.map_size_cm // self.map_resolution,
        )
        self.turn_angle = turn_angle
        self.collision_threshold = collision_threshold
        self.step_size = step_size
        self.start_obs_dilation_selem_radius = obs_dilation_selem_radius
        self.min_obs_dilation_selem_radius = min_obs_dilation_selem_radius
        #! myTODO: I think the unit of goal tolerance is in pixels, so I should do some conversions here. Right now this is hardcoded.
        self.goal_tolerance = goal_tolerance
        self.continuous_angle_tolerance = continuous_angle_tolerance

        self.vis_dir = None
        self.collision_map = None
        self.col_width = None
        self.last_global_pose = None
        self.curr_global_pose = None
        self.last_action = None
        self.timestep = 0
        self.total_timesteps = 0
        self.curr_obs_dilation_selem_radius = None
        self.obs_dilation_selem = None
        self.min_goal_distance_cm = min_goal_distance_cm
        self.dd = None

        self.map_downsample_factor = map_downsample_factor
        self.map_update_frequency = map_update_frequency
        self.panorama_start_steps = panorama_start_steps

        self.semantic_map = semantic_map
        self.instance_memory: InstanceMemory = instance_memory
        self.goal_filtering = goal_filtering
        self.prev_frontier = np.zeros(self.map_shape, dtype=np.uint8)
        self.frontier_metric = frontier_metric

        self.agent_id = agent_id
        self.log = get_logger(agent_id=agent_id)
        self.prefix = ""
        self.moved_forward = False

    def reset(self):
        self.vis_dir = self.default_vis_dir
        self.collision_map = np.zeros(self.map_shape)
        self.col_width = 1
        self.last_global_pose = None
        self.curr_global_pose = [
            self.map_size_cm / 100.0 / 2.0,
            self.map_size_cm / 100.0 / 2.0,
            0.0,
        ]
        self.last_action = None
        self.timestep = 0
        self.total_timesteps = 0
        self.curr_obs_dilation_selem_radius = self.start_obs_dilation_selem_radius
        self.obs_dilation_selem = skimage.morphology.disk(
            self.curr_obs_dilation_selem_radius
        )
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.prev_frontier = np.zeros(self.map_shape, dtype=np.uint8)
        self.moved_forward = False
    
    def reset_for_next_task(self):
        self.moved_forward = False

    def set_vis_dir(self, dir_name):
        self.vis_dir = os.path.join(self.default_vis_dir, dir_name)
        os.makedirs(self.vis_dir, exist_ok=True)

    def disable_print_images(self):
        self.print_images = False

    def plan(
        self,
        inst_goal_id: int = None,
        goal_semantic_id: int = None,
        fallback_to_frontier=True,
        postfix="",
        neighbors=None,
    ) -> Tuple[DiscreteNavigationAction, np.ndarray]:
        """Plan a low-level action.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            sensor_pose: (7,) array denoting global pose (x, y, o)
             and local map boundaries planning window (gx1, gx2, gy1, gy2)

        Returns:
            action: low-level action
        """
        reachable = False
        stop = False
        try_best = fallback_to_frontier == False
        vis_input = {}
        inst_goal_found = not inst_goal_id is None

        if inst_goal_found:
            self.episode_panorama_start_steps = 0

        if self.total_timesteps < self.episode_panorama_start_steps:
            return (
                DiscreteNavigationAction.TURN_RIGHT,
                vis_input,
            )  # If failed to plan, visualize locally.

        self.last_global_pose = self.curr_global_pose
        self.curr_global_pose = self.semantic_map.global_pose

        # Check collisions if we have just moved and are uncertain
        if self.last_action == DiscreteNavigationAction.MOVE_FORWARD:
            self._check_collision(postfix)

        self.log.info(f"---- Starting planning ---- ")
        self.log.info(f"> Global Location: {self.semantic_map.global_loc}")
        self.log.info(f"> Local Location: {self.semantic_map.local_loc}")
        self.log.info(f"> Instance goal found: {inst_goal_found}")

        if inst_goal_found:
            (
                reachable,
                stop,
                short_term_goal,
                is_local,
                vis_input,
            ) = self.plan_to_instance_goal(inst_goal_id, try_best, postfix)
        else:
            self.log.debug("No instance goal provided.")

        if not stop and not reachable:
            if fallback_to_frontier:
                (
                    reachable,
                    stop,
                    short_term_goal,
                    is_local,
                    vis_input,
                ) = self.plan_to_frontier_goal(
                    goal_semantic_id, postfix, neighbors=neighbors
                )

        if not (stop or reachable):
            action = None
        else:
            action = self.get_action(
                stop,
                short_term_goal,
                (
                    self.semantic_map.local_loc
                    if is_local
                    else self.semantic_map.global_loc
                ),
                None, # Not passing viewpoint orientation, because we are not going exactly to the viewpoint
            )

        self.last_action = action
        return action, vis_input

    def get_action(
        self, stop, short_term_goal, location, best_viewpoint=None
    ):
        """
        Gets discrete/continuous action given short-term goal. Agent orients to short term goal if stop=True
        """
        if self.moved_forward:
            self.log.debug("Already toward the goal, stopping.")
            return DiscreteNavigationAction.STOP

        viewpoint_orientation = None
        if best_viewpoint is not None:
            viewpoint_orientation = best_viewpoint.pose[2]
        angle_agent = pu.normalize_angle(self.curr_global_pose[2])

        if stop == False:
            stg_x, stg_y = short_term_goal
            relative_stg_x, relative_stg_y = stg_x - location[0], stg_y - location[1]
            angle_st_goal = math.degrees(math.atan2(relative_stg_x, relative_stg_y))
            relative_angle_to_stg = pu.normalize_angle(angle_agent - angle_st_goal)

            if self.discrete_actions:
                if relative_angle_to_stg > self.turn_angle / 2.0:
                    action = DiscreteNavigationAction.TURN_RIGHT
                elif relative_angle_to_stg < -self.turn_angle / 2.0:
                    action = DiscreteNavigationAction.TURN_LEFT
                else:
                    action = DiscreteNavigationAction.MOVE_FORWARD
        else:
            # Try to orient towards the goal object
            if viewpoint_orientation is None:
                # Compute angle to the final goal
                goal_x, goal_y = short_term_goal
                angle_goal = math.degrees(
                    math.atan2(goal_x - location[0], goal_y - location[1])
                )
                # Compute angle to the final goal
                relative_angle_to_closest_goal = pu.normalize_angle(
                    angle_agent - angle_goal
                )
            else:
                relative_angle_to_closest_goal = pu.normalize_angle(
                    angle_agent - viewpoint_orientation
                )
            self.log.debug(
                f">>> orienting towards the goal: {relative_angle_to_closest_goal}"
            )
            if self.discrete_actions:
                if relative_angle_to_closest_goal > 2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_RIGHT
                elif relative_angle_to_closest_goal < -2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_LEFT
                else:
                    self.log.debug("Already toward the goal, taking a last step toward the goal.")
                    action = DiscreteNavigationAction.MOVE_FORWARD
                    self.moved_forward = True



        # if action == DiscreteNavigationAction.STOP:
        #     self.reset_obs_dilation_selem_radius()
        return action

    def get_traversible(self, obstacles, is_local):
        dilated_obstacles = cv2.dilate(obstacles, self.obs_dilation_selem, iterations=1)
        dilated_obstacles[self.semantic_map.get_visited_map(is_local) == 1] = 0

        if is_local:
            gx1, gx2, gy1, gy2 = self.semantic_map.lmb
            collision_map = self.collision_map[gx1:gx2, gy1:gy2] == 1
        else:
            collision_map = self.collision_map == 1
        dilated_obstacles = np.logical_or(dilated_obstacles, collision_map)
        robot_loc = self.semantic_map.get_loc(is_local)
        dilated_obstacles[robot_loc] = 0

        traversible = 1 - dilated_obstacles
        return traversible

    def get_goal_map_pose(
        self,
        traversible,
        goal_instance_map,
        viewpoint_location,
        is_local,
        pose_idx,
        method,
    ):
        self.log.info(
            f"Creating goal map using viewpoint. Choosing {pose_idx}th traversible viewpoint."
        )

        self.log.debug(f"Viewpoint is : {viewpoint_location}")
        if method == "line_to_com":
            goal_indices = np.argwhere(goal_instance_map == 1)
            center_of_mass = goal_indices.mean(axis=0).astype(int)
            self.log.debug(f"Center of mass of the goal_mapis ({center_of_mass})")
            goal_point = center_of_mass
        else:
            # Find closest goal cell in goal_map to the viewpoint location
            goal_indices = np.argwhere(goal_instance_map == 1)
            dists = np.linalg.norm(
                goal_indices - np.array(viewpoint_location)[None, :], axis=1
            )
            closest_instance_idx = goal_indices[np.argmin(dists)]
            self.log.debug(
                f"Closest goal index in goal_map to goal_pose is ({closest_instance_idx})"
            )
            goal_point = closest_instance_idx

        line_coords = list(
            bresenham(
                goal_point[0],
                goal_point[1],
                viewpoint_location[0],
                viewpoint_location[1],
            )
        )

        first_pixels = []
        in_segment = False

        for x, y in line_coords:
            if not (0 <= x < traversible.shape[0] and 0 <= y < traversible.shape[1]):
                continue

            if traversible[x, y] == 1 and goal_instance_map[x, y] != 1:
                if not in_segment:
                    # Start of a new segment
                    self.log.info(f"Adding traversible pose {x, y}")
                    first_pixels.append((x, y))
                    in_segment = True
            else:
                in_segment = False

        first_pixels.append(viewpoint_location)

        if pose_idx >= len(first_pixels):
            self.log.info(f"No traversible view found for the instance goal.")
            return None

        goal_location = first_pixels[pose_idx]
        goal_map = np.zeros_like(goal_instance_map, dtype=np.uint8)
        goal_map[goal_location[0], goal_location[1]] = 1
        self.visualize_get_goal_map(
            traversible,
            goal_instance_map,
            first_pixels,
            viewpoint_location,
            goal_location,
            pose_idx,
            is_local,
        )
        return goal_map

    def _get_closest_free_cell(self, goal_instance_map, traversible):
        kernel_size = 2
        while kernel_size < 20:
            dilated_goal_instance_map = cv2.dilate(
                goal_instance_map.astype(np.uint8),
                skimage.morphology.disk(kernel_size),
                iterations=1,
            )
            free_goal_cells = np.logical_and(
                dilated_goal_instance_map == 1, traversible == 1
            )
            if np.any(free_goal_cells):
                break
            kernel_size += 1
        return free_goal_cells

    def get_goal_map_closest_to_viewpoint(
        self, traversible, goal_instance_map, viewpoint_location, is_local, try_index
    ):
        """
        Sometimes the free cells are on another side of the wall, and in that cases, the agent will need to continue exploring the map, and sometimes can never go there.
        """
        self.log.info(f"Creating goal map using closest goal to viewpoint.")
        if try_index != 0:
            self.log.info(f"Closest to viewpoint only works for try_index=0")
            return None
        # Here, I have to first dilate the goal instance map a bit, then from all free goal cells, choose closest to the viewpoint
        free_goal_cells = self._get_closest_free_cell(goal_instance_map, traversible)
        planner = FMMPlanner(
            traversible,
            step_size=self.step_size,
            vis_dir=self.vis_dir,
            print_images=self.print_images,
            goal_tolerance=self.goal_tolerance,
            # vis_postfix="_closest_to_viewpoint",
        )
        viewpoint_map = np.zeros_like(goal_instance_map)
        viewpoint_map[viewpoint_location[0], viewpoint_location[1]] = 1.0
        distances_from_viewpoint = planner.set_multi_goal(viewpoint_map, self.timestep)
        distances_from_viewpoint[free_goal_cells == 0] = 100000

        closest_goal_map = distances_from_viewpoint == distances_from_viewpoint.min()
        goal_location = np.unravel_index(
            closest_goal_map.argmax(), closest_goal_map.shape
        )
        goal_map = np.zeros_like(goal_instance_map, dtype=np.uint8)
        goal_map[goal_location[0], goal_location[1]] = 1

        features = [(goal_instance_map, [0, 165, 255])]  # orange - All instance cells
        points = []
        points.append((viewpoint_location, [255, 0, 0]))
        points.append((goal_location, [0, 0, 255]))

        visualize_map(
            traversible.shape,
            self.vis_dir,
            f"{self.prefix}{self.timestep}_6.get_closest_to_viewpoint{'' if is_local else '_global'}.png",
            traversible=traversible,
            features=features,
            points=points,
        )
        return goal_map

    def get_hybrid_goal_map(
        self, traversible, goal_instance_map, viewpoint_location, is_local, try_index
    ):
        """
        This is a hybrid to approach. We get closest cluster of free cells to the viewpoint. We try to dilate a goal a lot, so it can contain cells on different sides of the goal.
        Finally, we choose the closest cell in the cluster to a goal point.
        """
        self.log.info(f"Creating goal map using hybrid method.")
        if try_index != 0:
            self.log.info(f"Hybrid only works for try_index=0")
            return None

        kernel_size = 10
        dilated_goal_instance_map = cv2.dilate(
            goal_instance_map.astype(np.uint8),
            skimage.morphology.disk(kernel_size),
            iterations=1,
        )
        free_goal_cells = np.logical_and(
            dilated_goal_instance_map == 1, traversible == 1
        )
        labeled_map, num_clusters = label(free_goal_cells, structure=np.ones((3, 3)))
        if num_clusters == 0:
            free_goal_cells = self._get_closest_free_cell(
                goal_instance_map, traversible
            )
            labeled_map, num_clusters = label(free_goal_cells, structure=np.ones((3, 3)))
        

        viewpoint_map = np.zeros_like(goal_instance_map)
        viewpoint_map[viewpoint_location[0], viewpoint_location[1]] = 1.0

        traversible_ma = np.ma.masked_values(traversible * 1, 0)
        traversible_ma[viewpoint_location[0], viewpoint_location[1]] = 0
        distances_from_viewpoint = skfmm.distance(traversible_ma)
        distances_from_viewpoint = np.ma.filled(distances_from_viewpoint, np.max(distances_from_viewpoint) + 1)
        distances_from_viewpoint[free_goal_cells != 1] = 100000

        # Get closest cluster to viewpoint
        features = []  # orange - All instance cells
        min_distance_per_cluster = []
        for cluster_id in range(1, num_clusters + 1):
            cluster_mask = labeled_map == cluster_id
            features.append((cluster_mask, [0, int(255/(num_clusters+1) * cluster_id), 0]))
            min_dist_in_cluster = np.min(distances_from_viewpoint[cluster_mask])
            min_distance_per_cluster.append(min_dist_in_cluster)

        closest_cluster_id = np.argmin(min_distance_per_cluster) + 1
        closest_cluster_mask = labeled_map == closest_cluster_id


        # Choose a point in cluster

        # Compute each pixel's distance to the nearest goal cell
        goal_distance_map = distance_transform_edt(goal_instance_map == 0)
        cluster_distances_to_goal = goal_distance_map[closest_cluster_mask]
        min_dist = cluster_distances_to_goal.min()
        close_to_goal_mask = np.logical_and(
            closest_cluster_mask,
            goal_distance_map <= min_dist * 1.2
        )

        closest_idx = np.argmin(distances_from_viewpoint[close_to_goal_mask])
        candidate_coords = np.argwhere(close_to_goal_mask)
        goal_location = tuple(candidate_coords[closest_idx])

        goal_map = np.zeros_like(goal_instance_map, dtype=np.uint8)
        goal_map[goal_location[0], goal_location[1]] = 1

        features.append((goal_instance_map, [0, 165, 255]))  # orange - All instance cells
        points = []
        points.append((viewpoint_location, [255, 0, 0]))
        points.append((goal_location, [0, 0, 255]))

        visualize_map(
            traversible.shape,
            self.vis_dir,
            f"{self.prefix}{self.timestep}_6.get_hybrid_goal{'' if is_local else '_global'}.png",
            traversible=traversible,
            features=features,
            points=points,
        )
        return goal_map

    def get_goal_map(
        self,
        traversible,
        goal_instance_map,
        best_viewpoint,
        is_local,
        try_index,
        method,
        try_best=False,
    ):
        """
        Args:
            goal_map
            view_pose: Global loc that we can see the goal instance.
            method: "line_to_com" or "line_to_closest" or "closest_to_viewpoint"
        """
        if np.sum(goal_instance_map) < 47 and not try_best:
            self.log.info(f"Goal instance map too small ({np.sum(goal_instance_map)} cells). Not planning to it.")
            return None
        viewpoint_location = (
        self.semantic_map.global_pose_to_local_location(best_viewpoint.pose)
            if is_local
            else self.semantic_map.global_pose_to_global_location(best_viewpoint.pose)
        )
        if method in ["line_to_com", "line_to_closest"]:
            goal_map = self.get_goal_map_pose(
                traversible,
                goal_instance_map,
                viewpoint_location,
                is_local,
                try_index,
                method,
            )
        elif method == "closest_to_viewpoint":  # closest_to_viewpoint
            goal_map = self.get_goal_map_closest_to_viewpoint(
                traversible, goal_instance_map, viewpoint_location, is_local, try_index
            )
        elif method == "hybrid":
            goal_map = self.get_hybrid_goal_map(
                traversible, goal_instance_map, viewpoint_location, is_local, try_index
            )

        return goal_map

    def _get_short_term_goal(
        self,
        traversible: np.ndarray,
        goal_map: np.ndarray,
        location: List[int],
        postfix: str = "",
    ) -> Tuple[Tuple[int, int], np.ndarray, bool, bool]:
        """Get short-term goal.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            start: start location (x, y)
            planning_window: local map boundaries (gx1, gx2, gy1, gy2)
            plan_to_dilated_goal: for objectnav; plans to dialted goal points instead of explicitly checking reach.

        Returns:
            short_term_goal: short-term goal position (x, y) in map
            replan: binary flag to indicate we couldn't find a plan to reach
             the goal
            stop: binary flag to indicate we've reached the goal
        """
        # goal_map = add_boundary(goal_map, value=0)
        # traversible = add_boundary(traversible)
        self.log.debug(f"Getting short-term goal")
        planner = FMMPlanner(
            traversible,
            step_size=self.step_size,
            vis_dir=self.vis_dir,
            print_images=self.print_images,
            goal_tolerance=self.goal_tolerance,
            vis_postfix=postfix,
        )

        navigable_goal_map = np.logical_and(goal_map, traversible)
        #! myTODO
        if not np.any(navigable_goal_map):
            self.log.info(
                f"Couldn't find any navigable goal points in the map. Should only happen for frontier."
            )
            return (
                False,
                False,
                None,
                None,
                None,
            )

        # * Previously they had another logic of dilating goal similar to obstacles too (cv2.dilate(sel)). I don't see much difference, but I can think more later
        # dilated_goal_map = planner.dilate_goal(
        #     navigable_goal_map,
        #     self.min_goal_distance_cm / self.map_resolution,
        #     timestep=self.timestep,
        #     prefix=self.prefix,
        # )
        # dilated_goal_map = np.logical_and(dilated_goal_map, traversible)

        self.dd = planner.set_multi_goal(
            navigable_goal_map,
            self.timestep,
            self.dd,
            self.map_downsample_factor,
            self.map_update_frequency,
            # number="10",
        )

        state = [location[0], location[1]]

        # This is where we create the planner to get the trajectory to this state
        stg_x, stg_y, reachable, stop = planner.get_short_term_goal(
            state, timestep=self.timestep, prefix=self.prefix
        )

        short_term_goal = int(stg_x), int(stg_y)

        if self.print_images:
            points = [
                ([location[0], location[1]], [255, 0, 0]),  # start blue
                (
                    [short_term_goal[0], short_term_goal[1]],
                    [0, 255, 0],
                ),  # stg green
            ]
            visualize_map(
                navigable_goal_map.shape,
                self.vis_dir,
                f"{self.prefix}{self.timestep}_12.stg{postfix}.png",
                points=points,
                traversible=traversible,
                goal_map=navigable_goal_map,
            )

        return reachable, stop, short_term_goal, navigable_goal_map

    #! It actually gets closest geometrical goal, not closest traversible goal
    def get_closest_goal(self, goal_map, start):
        """closest goal, avoiding any obstacles."""
        empty = np.ones_like(goal_map)
        empty_planner = FMMPlanner(empty)
        empty_planner.set_goal(start)
        dist_map = empty_planner.fmm_dist * goal_map
        dist_map[dist_map == 0] = 10000
        closest_goal_map = dist_map == dist_map.min()
        closest_goal_map = remove_boundary(closest_goal_map)
        closest_goal_pt = np.unravel_index(
            closest_goal_map.argmax(), closest_goal_map.shape
        )
        return closest_goal_pt

    def _check_collision(self, postfix):
        """Check whether we had a collision and update the collision map."""
        collision_map_change = False
        x1, y1, t1 = self.last_global_pose.cpu().numpy()
        x2, y2, _ = self.curr_global_pose.cpu().numpy()
        # buf = 4
        buf = 2
        length = 2

        # You must move at least 5 cm when doing forward actions
        # Otherwise we assume there has been a collision
        if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
            self.col_width += 2
            # if self.col_width == 7:
            #     length = 4
            # buf = 3
            self.col_width = min(self.col_width, 5)
        else:
            self.col_width = 1

        dist = pu.get_l2_distance(x1, x2, y1, y2)

        if dist < self.collision_threshold:
            # We have a collision
            width = self.col_width

            # Add obstacles to the collision map
            for i in range(length):
                for j in range(width):
                    wx = x1 + self.map_resolution / 100 * (
                        (i + buf) * np.cos(np.deg2rad(t1))
                        + (j - width // 2) * np.sin(np.deg2rad(t1))
                    )
                    wy = y1 + self.map_resolution / 100 * (
                        (i + buf) * np.sin(np.deg2rad(t1))
                        - (j - width // 2) * np.cos(np.deg2rad(t1))
                    )
                    r, c = wy, wx
                    r, c = int(r * 100 / self.map_resolution), int(
                        c * 100 / self.map_resolution
                    )
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    self.collision_map[r, c] = 1
                    collision_map_change = True
        
        robot_loc = self.semantic_map.global_pose_to_global_location(self.curr_global_pose)
        self.collision_map[robot_loc[0], robot_loc[1]] = 0

        if collision_map_change:
            obstacle_map = self.semantic_map.get_obstacle_map(False)
            init_location = self.semantic_map.global_pose_to_global_location(self.last_global_pose)
            visualize_map(
                obstacle_map.shape,
                self.vis_dir,
                f"{self.prefix}{self.timestep}_4.collision_map_update{postfix}.png",
                points=[(init_location, [120, 0, 0]), (robot_loc, [255, 0, 0])],
                traversible=1 - obstacle_map,
                features=[
                    (self.collision_map, [0, 120, 120]),
                    (np.logical_and(self.collision_map, obstacle_map), [0, 255, 255]),
                ],  # light yellow / yellow
            )

    def get_largest_cluster(self, goal_instance_map, is_local) -> None:
        def convex_hull(binary_map):
            coords = np.column_stack(np.nonzero(binary_map))
            if len(coords) < 3:
                return binary_map

            hull = ConvexHull(coords)
            hull_coords = coords[hull.vertices]

            rr, cc = polygon(hull_coords[:, 0], hull_coords[:, 1], binary_map.shape)
            hull_mask = np.zeros_like(binary_map, dtype=np.uint8)
            hull_mask[rr, cc] = 1
            hull_mask = cv2.dilate(hull_mask, np.ones((3, 3)), iterations=1)

            return hull_mask

        if not self.goal_filtering:
            return

        self.log.debug("Clustering Instance map and selecting the largest cluster.")
        # init_goal_map_count = goal_instance_map.sum()

        labeled_map, _ = label(goal_instance_map) # Use deault structure of only vert or hor connection, to not include noises
        component_sizes = np.bincount(labeled_map.ravel())
        component_sizes[0] = 0
        largest_label = component_sizes.argmax()
        clustered_map = labeled_map == largest_label

        clustered_map_convex_hull = None
        if clustered_map.sum() > 0:
            # convex hull
            # Im not sure this was a good idea! so commented for now
            pass
            # try:
            #     clustered_map_convex_hull = convex_hull(clustered_map)
            #     self.log.debug("Choosing largest cluster for instance map.")
            #     self.log.debug(
            #         f"Goal map cells count changed from {init_goal_map_count} to {clustered_map.sum()}"
            #     )
            # except:
            #     self.log.debug(
            #         "Convex hull failed. Using the largest cluster without convex hull."
            #     )
        else:
            self.log.debug(
                "Instance map not changed. Largest cluster is empty for some reason!"
            )
            clustered_map = None

        visualize_map(
            goal_instance_map.shape,
            self.vis_dir,
            f"{self.prefix}{self.timestep}_3.cluster_goal.png",
            goal_map=clustered_map_convex_hull,  # Clustered goal map convex hull is red
            dilated_goal_map=goal_instance_map,  # All init goal points are magenta
            traversible=1 - self.semantic_map.get_obstacle_map(is_local),
            features=[(clustered_map, [0, 255, 0])],  # Largest cluster is green
        )
        if clustered_map is None:
            return goal_instance_map
        if clustered_map_convex_hull is None:
            return clustered_map
        return clustered_map_convex_hull

    def get_frontier_planning_maps(self):
        frontier_map = self.semantic_map.get_frontier_map(
            local=True, timestep=self.timestep
        )
        obstacle_map = self.semantic_map.get_obstacle_map(True)
        traversible = self.get_traversible(obstacle_map, True)
        frontier_map = frontier_map & traversible

        if frontier_map.any():
            return frontier_map, obstacle_map, traversible, True

        frontier_map = self.semantic_map.get_frontier_map(
            local=False, timestep=self.timestep
        )
        obstacle_map = self.semantic_map.get_obstacle_map(False)
        traversible = self.get_traversible(obstacle_map, False)
        frontier_map = frontier_map & traversible

        if frontier_map.any():
            return frontier_map, obstacle_map, traversible, False

        return None, None, None, False

    def plan_to_frontier_goal(self, goal_category, postfix, neighbors=None):

        i = 0
        while True:
            frontier_map, obstacle_map, traversible, is_local = (
                self.get_frontier_planning_maps()
            )
            if frontier_map is None:
                self.log.info("No frontiers remaining.")
                return False, False, None, None, {}

            robot_loc = self.semantic_map.get_loc(is_local)
            if self.prev_frontier.shape != frontier_map.shape or np.all(
                (self.prev_frontier & frontier_map) == 0
            ):
                best_frontier_map = self.get_best_frontier(
                    frontier_map,
                    traversible,
                    robot_loc,
                    goal_category,
                    is_local,
                    metric=self.frontier_metric,
                    neighbors=neighbors,
                )
            else:
                self.log.debug("Using previous frontier map for planning.")
                best_frontier_map = self.prev_frontier & frontier_map

            # visualize_map(
            #     obstacle_map.shape,
            #     self.vis_dir,
            #     f"{self.timestep}_5.visited_map{postfix}.png",
            #     points=[(robot_loc, [255, 0, 0])],
            #     traversible=1 - obstacle_map,
            #     frontier_map=self.semantic_map.get_visited_map(is_local)
            # )
            # visualize_map(
            #     obstacle_map.shape,
            #     self.vis_dir,
            #     f"{self.timestep}_6.unreachable_frontiers{f'_{i}'}{'' if is_local else '_global'}{postfix}.png",
            #     points=[(robot_loc, [255, 0, 0])],
            #     traversible=1 - obstacle_map,
            #     frontier_map=self.semantic_map.get_unreachable_frontiers_map(is_local)
            # )
            # visualize_map(
            #     obstacle_map.shape,
            #     self.vis_dir,
            #     f"{self.timestep}_7.planning_input_frontier{f'_{i}'}{'' if is_local else '_global'}{postfix}.png",
            #     points=[(robot_loc, [255, 0, 0])],
            #     traversible=1 - obstacle_map,
            #     dilated_goal_map=frontier_map,
            #     frontier_map=best_frontier_map,
            # )

            (
                reachable,
                stop,
                short_term_goal,
                dilated_frontier_map,
            ) = self._get_short_term_goal(
                traversible,
                best_frontier_map,
                robot_loc,
                postfix=f"_frontier_{i}",
            )
            if reachable:
                self.log.info("Planning to frontier successfull.")
                self.prev_frontier = best_frontier_map
                break

            self.log.info("Frontier not reachable.")
            traversible, success = self.decrease_obstacle_dilation_radius(
                traversible, obstacle_map, is_local
            )
            if not success:
                self.log.info(
                    f"Obstacle dilation radius is already at minimum. Could not plan to the frontier. Trying another one."
                )
                # self.semantic_map.set_unreachable_frontier(best_frontier_map, is_local)
                self.semantic_map.set_unreachable_frontier(
                    dilated_frontier_map, is_local
                )
                self.reset_obs_dilation_selem_radius()

            i += 1

        vis_input = {}
        if reachable:
            vis_input = {
                "short_term_goal": short_term_goal,
                "is_local": is_local,
            }
        vis_input["dilated_obstacle_map"] = 1 - traversible
        return reachable, stop, short_term_goal, is_local, vis_input

    # def get_frontier_scores_distance():

    def get_best_frontier(
        self,
        frontier_map,
        traversible,
        robot_loc,
        goal_category,
        is_local,
        metric="distance",
        neighbors=None,
    ):
        def distance_to_frontier(frontier, distances, loc):
            # Choose the closest point in the frontier to the robot
            dists = np.linalg.norm(frontier - loc, axis=1)
            closest = frontier[np.argmin(dists)]
            assert (
                traversible[closest[0], closest[1]] == 1
            ), "Closest point is not traversible"
            # self.log.debug(f"Frontier: {k}, closest: {closest}")
            return distances[closest[0], closest[1]]

            # Choose the center. Problem: Sometimes occupide.
            # return distances[center[0], center[1]]
        assert (
            traversible[robot_loc[0], robot_loc[1]] == 1
        ), "Robot location is not traversible"

        structure = np.ones((3, 3))  # 8-connectivity
        labeled_map, num_features = label(frontier_map, structure=structure)
        frontiers = [np.argwhere(labeled_map == i) for i in range(1, num_features + 1)]

        if metric == "semantics":
            sem_weights = CO_LOCATION_WEIGHTS[goal_category]
            sem_layers = self.semantic_map.get_semantic_map(is_local)
            semantic_close_radius = 40

        frontier_scores, frontier_centers, top_k_semantic_classes = [], [], []

        self.log.debug(f"Getting best frontier")
        traversible_ma = np.ma.masked_values(traversible * 1, 0)
        traversible_ma[robot_loc[0], robot_loc[1]] = 0
        distances = skfmm.distance(traversible_ma)
        distances = np.ma.filled(distances, np.max(distances) + 1)

        neighbor_distance_cache = {}
        agent_distances, other_distances = [], []
        for k, frontier in enumerate(frontiers):
            distance = distance_to_frontier(frontier, distances, robot_loc)
            # self.log.debug(f"Frontier: {k}")
            # self.log.debug(f" - distance: {distance}, robot loc: {robot_loc}")
            # if distance == np.max(distances):
                # Should not mask them here, because we are not sure if it is the min dilation
                # Also it interferes with the backup logic of decreasing dilation and trying again, and then masking.
                # self.semantic_map.set_unreachable_frontier(
                #     labeled_map == k+1, is_local
                # )
                # continue


            center = frontier.mean(axis=0).astype(int)
            frontier_centers.append(center)

            if metric == "distance":
                agent_distances.append(distance)
                if neighbors is None or len(neighbors) == 0:
                    frontier_scores.append(1 / (distance + 1))
                    top_k_semantic_classes.append([])
                else:
                    neighbor_distances = []
                    for neighbor in neighbors:
                        neighbor_loc = (
                            self.semantic_map.global_location_to_local_location(
                                neighbor.semantic_map.global_loc
                            )
                            if is_local
                            else neighbor.semantic_map.global_loc
                        )
                        traversible_ma = np.ma.masked_values(traversible * 1, 0)
                        if self.semantic_map.is_location_in_local_map(neighbor_loc):
                            key = tuple(neighbor_loc)
                            if key not in neighbor_distance_cache:
                                traversible_ma[neighbor_loc[0], neighbor_loc[1]] = 0
                                ndistances = skfmm.distance(traversible_ma)
                                ndistances = np.ma.filled(
                                    ndistances, np.max(ndistances) + 1
                                )
                                neighbor_distance_cache[key] = ndistances
                                neighbor_distance = distance_to_frontier(
                                    frontier, ndistances, neighbor_loc
                                )
                            else:
                                ndistances = neighbor_distance_cache[key]
                                neighbor_distance = distance_to_frontier(
                                    frontier, ndistances, neighbor_loc
                                )

                            # self.log.debug(
                            #     f"     - neighbor dis: {neighbor_distance}, neighbor loc: {neighbor_loc}"
                            # )
                        else:
                            neighbor_distance = 100000

                        neighbor_distances.append(neighbor_distance)

                    other_distances.append(neighbor_distances)
                    neighbor_distance = min(neighbor_distances)
                    # The number in denominator removes division by 0. The value in the numinator ensures that if two agents have the same distance, we choose the frontier with smallest distance.
                    frontier_scores.append(
                        (neighbor_distance + 10e-5) / (distance + 10e-6)
                    )
                    # self.log.debug(
                    #     f"min neighbor dis: {neighbor_distance}, frontier score: {frontier_scores[-1]}"
                    # )
                    top_k_semantic_classes.append([])

            elif metric == "semantics":
                local_map = sem_layers[
                    1 : 52 + 1,
                    center[0] - semantic_close_radius : center[0] + semantic_close_radius,
                    center[1] - semantic_close_radius : center[1] + semantic_close_radius,
                ]
                neighbor_classes = (
                    np.where(local_map.any(axis=(1, 2)))[0] + 1
                )  # +1 to match ids

                # self.log.debug(f"frontier {k}")
                if len(neighbor_classes) > 0:
                    scores = sem_weights[neighbor_classes]
                    frontier_sem_score = np.mean(scores)
                    top_classes = neighbor_classes[np.argsort(scores)[-3:]].tolist()
                    # self.log.debug(f"Top classes: {top_classes}, Scores: {scores}, Frontier score: {frontier_sem_score}")
                else:
                    frontier_sem_score = np.mean(sem_weights)
                    # self.log.debug(f"No classes found in the local map. Using mean score: {frontier_sem_score}")
                    top_classes = []

                frontier_sem_score = np.exp(8 * frontier_sem_score)
                frontier_sem_score /= distance
                # self.log.debug(f"distance: {distance}, final score: {frontier_sem_score}")
                frontier_scores.append(frontier_sem_score)
                top_k_semantic_classes.append(top_classes)
            else:
                raise Exception(f"Unknown metric: {metric}")

        if metric == "distance":
            visualize_distance_frontiers(
                self.vis_dir,
                traversible=traversible,
                frontier_map=frontier_map,
                frontier_centers=frontier_centers,
                frontier_scores=frontier_scores,
                robot_loc=robot_loc,
                agent_dists=agent_distances,
                other_agents_dists=other_distances,
                top_k=5,
                save_path=f"{self.prefix}{self.timestep}_14.frontier_scores{'' if is_local else '_global'}.png",
            )
        else:
            visualize_semantic_frontiers(
                self.vis_dir,
                traversible=traversible,
                frontier_map=frontier_map,
                frontier_centers=frontier_centers,
                frontier_scores=frontier_scores,
                top_k_semantic_classes=top_k_semantic_classes,
                robot_loc=robot_loc,
                top_k=5,
                save_path=f"{self.prefix}_14.frontier_scores{'' if is_local else '_global'}.png",
            )
        assert (
            len(frontier_scores) != 0
        ), "No frontiers found, but frontier_map is not empty."

        my_priority = 1
        if not neighbors is None:
            for neighbor in neighbors:
                neighbor_loc = (
                    self.semantic_map.global_location_to_local_location(
                        neighbor.semantic_map.global_loc
                    )
                    if is_local
                    else neighbor.semantic_map.global_loc
                )
                if neighbor_loc == robot_loc and neighbor.agent_id < self.agent_id:
                    my_priority += 1

        if my_priority <= len(frontier_scores):
            self.log.debug(
                f"Getting {my_priority}th best frontier (agent: {self.agent_id})"
            )
            best_frontier = np.argsort(frontier_scores)[-my_priority]
        else:
            # If more agents on the same location than the number of frontiers, go to the best
            best_frontier = np.argsort(frontier_scores)[-1]

        best_frontier_map = np.zeros_like(traversible)
        best_frontier = frontiers[best_frontier]
        best_frontier_map[best_frontier[:, 0], best_frontier[:, 1]] = 1
        return best_frontier_map

    def plan_to_instance_goal(self, instance_goal_id, try_best, postfix, method="hybrid"):
        (
            goal_instance_map,
            best_viewpoint,
            is_local,
            obstacle_map,
            robot_loc,
            traversible,
        ) = self.get_instance_planning_maps(instance_goal_id, method)

        # visualize_map(
        #     obstacle_map.shape,
        #     self.vis_dir,
        #     f"{self.timestep}_5.visited_map{postfix}.png",
        #     points=[(robot_loc, [255, 0, 0])],
        #     traversible=1 - obstacle_map,
        #     frontier_map=self.semantic_map.get_visited_map(is_local)
        # )
        # instance_on_obstacles = np.logical_and(
        #     goal_instance_map == 1, obstacle_map == 1
        # )
        # visualize_map(
        #     obstacle_map.shape,
        #     self.vis_dir,
        #     f"{self.timestep}_7.planning_input_instance{postfix}.png",
        #     points=[(robot_loc, [255, 0, 0]), (viewpoint_loc, [120, 0, 0])],
        #     traversible=1 - obstacle_map,
        #     goal_map=goal_instance_map,
        #     features=[(instance_on_obstacles, [0, 255, 255])],  # yellow
        # )

        i = 0
        try_idx = 0
        stop = False
        reachable = False
        force_global = False
        short_term_goal = None
        while True:
            goal_map = self.get_goal_map(
                traversible,
                goal_instance_map,
                best_viewpoint,
                is_local,
                try_idx,
                method,
                try_best=try_best,
            )
            if goal_map is None:
                # This mean we couldn't find any traversible pose. Even with the minimum dilation radius.
                break

            self.log.info(
                f"Trying to plan to instance goal with\n\t - pose_idx: {try_idx}\n\t - {'local' if is_local else 'global'}\n\t - obs dilation: {self.curr_obs_dilation_selem_radius}"
            )

            (reachable, stop, short_term_goal, navigable_goal_map) = (
                self._get_short_term_goal(
                    traversible,
                    goal_map,
                    robot_loc,
                    postfix=f"_attempt_{i}_pose_{try_idx}{'' if is_local else '_global'}{postfix}",
                )
            )

            if stop or reachable:
                break

            i += 1

            self.log.info("Could not find a path to the high-level goal.")
            if is_local:
                force_global = True
            else:
                force_global = False
                traversible, success = self.decrease_obstacle_dilation_radius(
                    traversible, obstacle_map, is_local
                )
                if not success:
                    try_idx += 1
                    #! myTODO: Important: This might not be lots of heurisitc. Maybe its better to use something like BLACKLISTED_TARGET_MAP, however, that has its own issues.
                    self.semantic_map.merge_map(
                        navigable_goal_map, MC.OBSTACLE_MAP, is_local
                    )
                    self.reset_obs_dilation_selem_radius()
            (
                goal_instance_map,
                best_viewpoint,
                is_local,
                obstacle_map,
                robot_loc,
                traversible,
            ) = self.get_instance_planning_maps(
                instance_goal_id, method, force_global=force_global
            )

        if reachable:
            self.log.debug(f"Planning to instance goal successfull.")
            if stop:
                self.log.debug(f"We need to stop.")
            else:
                self.log.debug(f"Short term goal: {short_term_goal}")
        if stop:
            x_goal, y_goal = np.where(goal_instance_map)
            short_term_goal = int(np.mean(x_goal)), int(np.mean(y_goal))

        vis_input = {}
        if reachable:
            vis_input = {
                "short_term_goal": short_term_goal,
                "is_local": is_local,
                "inst_goal_found": True,
            }
        vis_input["dilated_obstacle_map"] = 1 - traversible
        vis_input["goal_instance_map"] = goal_instance_map

        return (
            reachable,
            stop,
            short_term_goal,
            is_local,
            vis_input,
        )

    def get_goal_instance_map_and_viewpoint(
        self, instance_goal_id, method, force_global=False
    ):
        instance_views = self.instance_memory.instances[instance_goal_id].instance_views

        local_goal_instance_map = self.semantic_map.get_instance_map(
            instance_goal_id, local=True
        )
        global_goal_instance_map = self.semantic_map.get_instance_map(
            instance_goal_id, local=False
        )

        all_views_local_locations = [
            self.semantic_map.global_pose_to_local_location(view.pose)
            for view in instance_views
        ]
        are_views_local = all(
            [
                self.semantic_map.is_location_in_local_map(view_local_location)
                for view_local_location in all_views_local_locations
            ]
        )

        is_local = (
            are_views_local
            and np.any(local_goal_instance_map)
            and np.count_nonzero(local_goal_instance_map)
            == np.count_nonzero(global_goal_instance_map)
            and not force_global
        )

        if is_local:
            goal_instance_map = local_goal_instance_map
            all_views_locations = all_views_local_locations
        else:
            goal_instance_map = global_goal_instance_map
            all_views_locations = [
                self.semantic_map.global_pose_to_global_location(view.pose)
                for view in instance_views
            ]

        goal_instance_map = self.get_largest_cluster(goal_instance_map, is_local)

        if method == "hybrid":
            goal_indices = np.argwhere(goal_instance_map == 1)
            center = goal_indices.mean(axis=0)  # can be float
            all_views_locations = np.stack(all_views_locations)
            best_view = np.argmin(np.linalg.norm(all_views_locations - center, axis=1))
        else:
            best_view = np.argmax([view.object_coverage for view in instance_views])

        best_view_pose = instance_views[best_view].pose
        best_view_loc = (
            self.semantic_map.global_pose_to_local_location(best_view_pose)
            if is_local
            else self.semantic_map.global_pose_to_global_location(best_view_pose)
        )
        self.log.debug(
            f"viewpoint location is: {best_view_loc}, with coverage {instance_views[best_view].object_coverage}."
        )

        self.log.debug(
            f">>> Goal instance {instance_goal_id} {'not' if not is_local else ''} present in local map."
        )
        return (
            goal_instance_map,
            instance_views[best_view],
            is_local,
        )

    def get_instance_planning_maps(self, instance_goal_id, method, force_global=False):
        goal_instance_map, best_viewpoint, is_local = (
            self.get_goal_instance_map_and_viewpoint(
                instance_goal_id, method, force_global=force_global
            )
        )
        obstacle_map = self.semantic_map.get_obstacle_map(is_local)
        robot_loc = self.semantic_map.get_loc(is_local)
        traversible = self.get_traversible(obstacle_map, is_local)
        return (
            goal_instance_map,
            best_viewpoint,
            is_local,
            obstacle_map,
            robot_loc,
            traversible,
        )

    def reset_obs_dilation_selem_radius(self):
        self.curr_obs_dilation_selem_radius = self.start_obs_dilation_selem_radius
        self.obs_dilation_selem = skimage.morphology.disk(
            self.curr_obs_dilation_selem_radius
        )

    def decrease_obstacle_dilation_radius(self, traversible, obstacle_map, is_local):
        # self.collision_map *= 0
        if self.curr_obs_dilation_selem_radius > self.min_obs_dilation_selem_radius:
            self.curr_obs_dilation_selem_radius -= 1
            self.obs_dilation_selem = skimage.morphology.disk(
                self.curr_obs_dilation_selem_radius
            )
            self.log.info(
                f"Decreasing obstacle dilation radius to {self.curr_obs_dilation_selem_radius}. Trying again."
            )
            traversible = self.get_traversible(obstacle_map, is_local)
            return traversible, True
        else:
            return traversible, False

    def visualize_get_goal_map(
        self,
        traversible,
        goal_instance_map,
        first_pixels,
        viewpoint_location,
        goal_location,
        index,
        is_local,
    ):
        features = [(goal_instance_map, [0, 165, 255])]  # orange - All instance cells
        points = []
        for pixel in first_pixels:
            points.append((pixel, [0, 255, 0]))  # green - All poses

        # blue - viewpoint
        points.append((viewpoint_location, [255, 0, 0]))

        # red - final goal
        points.append((goal_location, [0, 0, 255]))

        visualize_map(
            traversible.shape,
            self.vis_dir,
            f"{self.timestep}_6.interpolate_goal_idx{index}{'' if is_local else '_global'}.png",
            traversible=traversible,
            features=features,
            points=points,
        )
