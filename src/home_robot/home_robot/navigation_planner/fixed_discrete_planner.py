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
from home_robot.utils.visualization import visualize_map, visualize_frontier_scores_matplotlib
from home_robot.utils.logger import get_logger
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory


logger = get_logger()


script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
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
        goal_tolerance: float = 0.01,  # for sim
        discrete_actions: bool = True,
        continuous_angle_tolerance: float = 30.0,
        panorama_start_steps: int = 0,
        semantic_map: Categorical2DSemanticMapState = None,
        instance_memory: InstanceMemory = None,
        goal_filtering=False,
        frontier_metric: str = "distance"
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
        self.goal_tolerance = goal_tolerance
        self.continuous_angle_tolerance = continuous_angle_tolerance

        self.vis_dir = None
        self.collision_map = None
        self.col_width = None
        self.last_global_pose = None
        self.curr_global_pose = None
        self.last_action = None
        self.timestep = 0
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
        self.prev_frontier = None
        self.frontier_metric = frontier_metric

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
        self.timestep = 1
        self.curr_obs_dilation_selem_radius = self.start_obs_dilation_selem_radius
        self.obs_dilation_selem = skimage.morphology.disk(
            self.curr_obs_dilation_selem_radius
        )
        self.episode_panorama_start_steps = self.panorama_start_steps

    def set_vis_dir(self, scene_id: str, episode_id: str):
        self.vis_dir = os.path.join(self.default_vis_dir, f"{scene_id}_{episode_id}")
        shutil.rmtree(self.vis_dir, ignore_errors=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def disable_print_images(self):
        self.print_images = False

    def plan(
        self,
        inst_goal_found: bool,
        inst_goal_id: int,
        timestep: int,
        total_timesteps: int,
        goal_semantic_id: int,
        fallback_to_frontier=True,
        postfix="",
    ) -> Tuple[DiscreteNavigationAction, np.ndarray]:
        """Plan a low-level action.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            sensor_pose: (7,) array denoting global pose (x, y, o)
             and local map boundaries planning window (gx1, gx2, gy1, gy2)
            found_goal: whether we found the object goal category

        Returns:
            action: low-level action
            closest_goal_map: (M, M) binary array denoting closest goal
             location in the goal map in geodesic distance
        """
        reachable = False
        stop = False
        viewpoint_orientation = None
        vis_input = {}
        self.timestep = timestep

        if inst_goal_found:
            self.episode_panorama_start_steps = 0

        if total_timesteps < self.episode_panorama_start_steps:
            #! When total_timesteps is less than the panorama start steps, we just turn right. So if turn angle is 30, at 12th step, we don't need to turn anymore.
            return DiscreteNavigationAction.TURN_RIGHT, vis_input # If failed to plan, visualize locally.

        self.last_global_pose = self.curr_global_pose
        self.curr_global_pose = self.semantic_map.global_pose

        # Check collisions if we have just moved and are uncertain
        if self.last_action == DiscreteNavigationAction.MOVE_FORWARD:
            self._check_collision(postfix)

        logger.info(f"---- Starting planning ---- ")
        logger.info(f"> Global Location: {self.semantic_map.global_loc}")
        logger.info(f"> Local Location: {self.semantic_map.local_loc}")
        logger.info(f"> Instance goal found: {inst_goal_found}")

        if inst_goal_found:
            (
                reachable,
                stop,
                short_term_goal,
                closest_goal_pt,
                viewpoint_orientation,
                is_local,
                vis_input,
            ) = self.plan_to_instance_goal(inst_goal_id, postfix=postfix)
        else:
            logger.debug("No instance goal provided.")

        if not stop and not reachable:
            if fallback_to_frontier:
                (
                    reachable,
                    stop,
                    short_term_goal,
                    closest_goal_pt,
                    is_local,
                    vis_input,
                ) = self.plan_to_frontier_goal(goal_semantic_id, postfix)

        if not (stop or reachable):
            action = None
        else:
            action = self.get_action(
                stop,
                short_term_goal,
                closest_goal_pt,
                (
                    self.semantic_map.local_loc
                    if is_local
                    else self.semantic_map.global_loc
                ),
                viewpoint_orientation,
            )

        self.last_action = action
        return action, vis_input

    def get_action(
        self, stop, short_term_goal, closest_goal_pt, location, viewpoint_orientation
    ):
        """
        Gets discrete/continuous action given short-term goal. Agent orients to closest goal if found_goal=True and stop=True
        """
        angle_agent = pu.normalize_angle(self.curr_global_pose[2])

        # stop == True, orient towards goal first, then actually stop.
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
            # Try to orient towards the goal object - or at least any point sampled from the goal
            # object.
            logger.debug("----------------------------")
            logger.debug(
                ">>> orienting towards the goal: {relative_angle_to_closest_goal}"
            )
            if viewpoint_orientation is None:
                # Compute angle to the final goal
                goal_x, goal_y = closest_goal_pt
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
            if self.discrete_actions:
                if relative_angle_to_closest_goal > 2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_RIGHT
                elif relative_angle_to_closest_goal < -2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_LEFT
                else:
                    logger.debug("Already toward the goal, stopping.")
                    action = DiscreteNavigationAction.STOP

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

        traversible = 1 - dilated_obstacles
        return traversible

    def get_goal_map(
        self, traversible, goal_instance_map, viewpoint_location, is_local, pose_idx, method="com"
    ):
        """
        Args:
            goal_map
            view_pose: Global loc that we can see the goal instance.
        """
        logger.info(
            f"Creating goal map using viewpoint. Choosing {pose_idx}th traversible viewpoint."
        )

        logger.debug(f"Viewpoint is : {viewpoint_location}")


        if method == "com":
            goal_indices = np.argwhere(goal_instance_map == 1)
            center_of_mass = goal_indices.mean(axis=0).astype(int)
            logger.debug(
                f"Center of mass of the goal_mapis ({center_of_mass})"
            )
            goal_point = center_of_mass
        else:
            # Find closest goal cell in goal_map to the viewpoint location
            goal_indices = np.argwhere(goal_instance_map == 1)
            dists = np.linalg.norm(
                goal_indices - np.array(viewpoint_location)[None, :], axis=1
            )
            closest_instance_idx = goal_indices[np.argmin(dists)]
            logger.debug(
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
                    logger.info(f"Adding traversible pose {x, y}")
                    first_pixels.append((x, y))
                    in_segment = True
            else:
                in_segment = False

        first_pixels.append(viewpoint_location)

        if pose_idx >= len(first_pixels):
            logger.info(f"No traversible view found for the instance goal.")
            return None

        goal_map = np.zeros_like(goal_instance_map, dtype=np.uint8)
        goal_location = first_pixels[pose_idx]
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
            closest_goal_map: (M, M) binary array denoting closest goal
             location in the goal map in geodesic distance
            replan: binary flag to indicate we couldn't find a plan to reach
             the goal
            stop: binary flag to indicate we've reached the goal
        """
        # goal_map = add_boundary(goal_map, value=0)
        # traversible = add_boundary(traversible)
        logger.debug(f"Getting short-term goal")
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
            logger.info(
                f"Couldn't find any navigable goal points in the map. Should only happned for frontier."
            )
            return (
                False,
                False,
                None,
                None,
            )

        # * Previously they had another logic of dilating goal similar to obstacles too (cv2.dilate(sel)). I don't see much difference, but I can think more later
        dilated_goal_map = planner.dilate_goal(
            navigable_goal_map,
            self.min_goal_distance_cm / self.map_resolution,
            timestep=self.timestep,
        )
        dilated_goal_map = np.logical_and(dilated_goal_map, traversible)

        self.dd = planner.set_multi_goal(
            dilated_goal_map,
            self.timestep,
            self.dd,
            self.map_downsample_factor,
            self.map_update_frequency,
            number="10",
        )

        # goal_distance_map, closest_goal_pt = self.get_closest_goal(navigable_goal_map, local_loc)
        #! myTODO: Make sure if I should use navigable goal map or dilated goal map or original goal map
        closest_goal_pt = self.get_closest_goal(
            navigable_goal_map, location
        )

        #! myTODO: Looks like this is no needed
        # self.timestep += 1

        state = [location[0] + 1, location[1] + 1]

        # This is where we create the planner to get the trajectory to this state
        stg_x, stg_y, reachable, stop = planner.get_short_term_goal(
            state, timestep=self.timestep
        )
        stg_x, stg_y = stg_x - 1, stg_y - 1

        short_term_goal = int(stg_x), int(stg_y)

        if self.print_images:
            points = [
                ([location[0] + 1, location[1] + 1], [255, 0, 0]),  # start blue
                (
                    [short_term_goal[0] + 1, short_term_goal[1] + 1],
                    [0, 255, 0],
                ),  # stg green
            ]
            visualize_map(
                dilated_goal_map.shape,
                self.vis_dir,
                f"{self.timestep}_12.stg{postfix}.png",
                points=points,
                traversible=traversible,
                goal_map=dilated_goal_map,
            )

        return (
            reachable,
            stop,
            short_term_goal,
            closest_goal_pt,
        )

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
            self.col_width = min(self.col_width, 7)
        else:
            self.col_width = 1

        dist = pu.get_l2_distance(x1, x2, y1, y2)

        if dist < self.collision_threshold:
            # We have a collision
            width = self.col_width

            # Add obstacles to the collision map
            for i in range(length):
                for j in range(width):
                    wx = x1 + self.map_resolution/100 * (
                        (i + buf) * np.cos(np.deg2rad(t1))
                        + (j - width // 2) * np.sin(np.deg2rad(t1))
                    )
                    wy = y1 + self.map_resolution/100 * (
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

        robot_loc = int(y2 * 100 / self.map_resolution), int(x2 * 100 / self.map_resolution)
        self.collision_map[robot_loc[0], robot_loc[1]] = 0

        if collision_map_change:
            obstacle_map = self.semantic_map.get_obstacle_map(False)
            init_location = int(y1 * 100 / self.map_resolution), int(x1 * 100 / self.map_resolution)
            visualize_map(
                obstacle_map.shape,
                self.vis_dir,
                f"{self.timestep}_4.collision_map_update{postfix}.png",
                points=[(init_location, [120, 0, 0]), (robot_loc, [255, 0, 0])],
                traversible=1 - obstacle_map,
                features=[(self.collision_map, [0, 120, 120]), (np.logical_and(self.collision_map, obstacle_map), [0, 255, 255])],  # light yellow / yellow
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
            hull_mask = cv2.dilate(hull_mask, np.ones((3,3)), iterations=1)
            
            return hull_mask

        if not self.goal_filtering:
            return

        logger.debug("Clustering Instance map and selecting the largest cluster.")
        init_goal_map_count = goal_instance_map.sum()

        labeled_map, _ = label(goal_instance_map, structure=np.ones((3, 3)))
        component_sizes = np.bincount(labeled_map.ravel())
        component_sizes[0] = 0
        largest_label = component_sizes.argmax()
        clustered_map = labeled_map == largest_label

        if clustered_map.sum() > 0:
            # convex hull
            try:
                clustered_map_convex_hull = convex_hull(clustered_map)
                logger.debug("Choosing largest cluster for instance map.")
                logger.debug(
                    f"Goal map cells count changed from {init_goal_map_count} to {clustered_map.sum()}"
                )
            except:
                logger.debug(
                    "Convex hull failed. Using the largest cluster without convex hull."
                )
                clustered_map_convex_hull = None
        else:
            logger.debug(
                "Instance map not changed. Largest cluster is empty for some reason!"
            )
            clustered_map = None
            clustered_map_convex_hull = None
            
        visualize_map(
            goal_instance_map.shape,
            self.vis_dir,
            f"{self.timestep}_3.cluster_goal.png",
            goal_map=clustered_map_convex_hull, # Clustered goal map convex hull is red
            dilated_goal_map=goal_instance_map, # All init goal points are magenta
            traversible=1 - self.semantic_map.get_obstacle_map(is_local),
            features=[(clustered_map, [0, 255, 0])] # Largest cluster is green
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

        if frontier_map.any():
            return frontier_map, obstacle_map, traversible, False

        return None, None, None, False

    def plan_to_frontier_goal(self, goal_category, postfix):

        i = 0
        while True:
            frontier_map, obstacle_map, traversible, is_local = self.get_frontier_planning_maps()
            if frontier_map is None:
                logger.info("No frontiers remaining.")
                return False, False, None, None, None, {}

            robot_loc = (
                self.semantic_map.local_loc if is_local else self.semantic_map.global_loc
            )
            if self.prev_frontier is None or np.all((self.prev_frontier & frontier_map)==0):
                best_frontier_map = self.get_best_frontier(frontier_map, traversible, robot_loc, goal_category, is_local, metric=self.frontier_metric)
            else:
                logger.debug("Using previous frontier map for planning.")
                best_frontier_map = self.prev_frontier & frontier_map

            visualize_map(
                obstacle_map.shape,
                self.vis_dir,
                f"{self.timestep}_5.visited_map{postfix}.png",
                points=[(robot_loc, [255, 0, 0])],
                traversible=1 - obstacle_map,
                frontier_map=self.semantic_map.get_visited_map(is_local)
            )
            visualize_map(
                obstacle_map.shape,
                self.vis_dir,
                f"{self.timestep}_6.unreachable_frontiers{f'_{i}'}{'' if is_local else '_global'}{postfix}.png",
                points=[(robot_loc, [255, 0, 0])],
                traversible=1 - obstacle_map,
                frontier_map=self.semantic_map.get_unreachable_frontiers_map(is_local)
            )
            visualize_map(
                obstacle_map.shape,
                self.vis_dir,
                f"{self.timestep}_7.planning_input_frontier{f'_{i}'}{'' if is_local else '_global'}{postfix}.png",
                points=[(robot_loc, [255, 0, 0])],
                traversible=1 - obstacle_map,
                dilated_goal_map=frontier_map,
                frontier_map=best_frontier_map,
            )


            (
                reachable,
                stop,
                short_term_goal,
                closest_goal_pt,
            ) = self._get_short_term_goal(
                traversible,
                best_frontier_map,
                robot_loc,
                postfix=f"_frontier_{i}",
            )
            if reachable:
                logger.info("Planning to frontier successfull.")
                self.prev_frontier = best_frontier_map
                break

            logger.info("Frontier not reachable.")
            traversible, success = self.decrease_obstacle_dilation_radius(
                traversible, obstacle_map, is_local
            )
            if not success:
                logger.info(
                    f"Obstacle dilation radius is already at minimum. Could not plan to the frontier. Trying another one."
                )
                self.semantic_map.set_unreachable_frontier(best_frontier_map, is_local)
                self.reset_obs_dilation_selem_radius()

            i += 1

        vis_input = {}
        if reachable:
            vis_input = {
                "closest_goal_pt": closest_goal_pt,
                "short_term_goal": short_term_goal,
                "is_local": is_local,
            }
        vis_input["dilated_obstacle_map"] = 1 - traversible
        return reachable, stop, short_term_goal, closest_goal_pt, is_local, vis_input
    
    def get_best_frontier(self, frontier_map, traversible, robot_loc, goal_category, is_local, metric="distance"):
        frontier_map = frontier_map & traversible
        structure = np.ones((3, 3))  # 8-connectivity
        labeled_map, num_features = label(frontier_map, structure=structure)
        frontiers = [
            np.argwhere(labeled_map == i)
            for i in range(1, num_features + 1)
        ]

        sem_weights = CO_LOCATION_WEIGHTS[goal_category]
        sem_layers = self.semantic_map.get_semantic_map(is_local, full=True)
        r = 40

        frontier_scores = []
        frontier_centers = []
        top_k_semantic_classes = []

        logger.debug(f"Getting best frontier")
        traversible_ma = np.ma.masked_values(traversible * 1, 0)
        assert traversible[robot_loc[0], robot_loc[1]] == 1, "Robot location is not traversible"
        traversible_ma[robot_loc[0], robot_loc[1]] = 0
        distances = skfmm.distance(traversible_ma)
        distances = np.ma.filled(distances, np.max(distances) + 1)
        for k, frontier in enumerate(frontiers):
            center = frontier.mean(axis=0).astype(int)
            frontier_centers.append(center)

            # Choose the closest point in the frontier to the robot
            dists = np.linalg.norm(frontier - robot_loc, axis=1)
            closest = frontier[np.argmin(dists)]
            assert traversible[closest[0], closest[1]] == 1, "Closest point is not traversible"
            logger.debug(f"Frontier: {k}, closest: {closest}")
            distance = distances[closest[0], closest[1]]

            # Choose the center. Problem: Sometimes occupide. 
            # distance = distances[center[0], center[1]]

            if metric == "distance":
                frontier_scores.append(1 / (distance + 1))
                top_k_semantic_classes.append([])

            elif metric == "semantics":
                local_map = sem_layers[
                    1:52+1, center[0] - r : center[0] + r, center[1] - r : center[1] + r
                ]
                neighbor_classes = np.where(local_map.any(axis=(1, 2)))[0] + 1 # +1 to match ids

                logger.debug(f"frontier {k}")
                if len(neighbor_classes) > 0:
                    scores = sem_weights[neighbor_classes]
                    frontier_sem_score = np.mean(scores)
                    top_classes = neighbor_classes[np.argsort(scores)[-3:]].tolist()
                    logger.debug(f"Top classes: {top_classes}, Scores: {scores}, Frontier score: {frontier_sem_score}")
                else:
                    frontier_sem_score = np.mean(sem_weights)
                    logger.debug(f"No classes found in the local map. Using mean score: {frontier_sem_score}")
                    top_classes = []

                frontier_sem_score /= distance
                logger.debug(f"distance: {distance}, final score: {frontier_sem_score*1000}")
                frontier_scores.append(frontier_sem_score*1000)
                top_k_semantic_classes.append(top_classes)
            else:
                raise Exception(f"Unknown metric: {metric}")

        visualize_frontier_scores_matplotlib(
            self.vis_dir,
            traversible=traversible,
            frontier_map=frontier_map,
            frontier_centers=frontier_centers,
            frontier_scores=frontier_scores,
            top_k_semantic_classes=top_k_semantic_classes,
            robot_loc=robot_loc,
            top_k=5,
            save_path=f"{self.timestep}_14.frontier_scores{'' if is_local else '_global'}.png"
        )

        assert len(frontier_scores) != 0, "No frontiers found, but frontier_map is not empty."
        # Select the frontier with the highest score
        best_frontier = np.argmax(frontier_scores)
        best_frontier_map = np.zeros_like(traversible)
        best_frontier = frontiers[best_frontier]
        best_frontier_map[best_frontier[:,0], best_frontier[:,1]] = 1
        return best_frontier_map

    def plan_to_instance_goal(self, instance_goal_id, postfix):
        goal_instance_map, viewpoint_loc, viewpoint_orientation, is_local, obstacle_map, robot_loc, traversible = self.get_instance_planning_maps(instance_goal_id)
        instance_on_obstacles = np.logical_and(
            goal_instance_map == 1, obstacle_map == 1
        )

        visualize_map(
            obstacle_map.shape,
            self.vis_dir,
            f"{self.timestep}_5.visited_map{postfix}.png",
            points=[(robot_loc, [255, 0, 0])],
            traversible=1 - obstacle_map,
            frontier_map=self.semantic_map.get_visited_map(is_local)
        )
        visualize_map(
            obstacle_map.shape,
            self.vis_dir,
            f"{self.timestep}_7.planning_input_instance{postfix}.png",
            points=[(robot_loc, [255, 0, 0]), (viewpoint_loc, [120, 0, 0])],
            traversible=1 - obstacle_map,
            goal_map=goal_instance_map,
            features=[(instance_on_obstacles, [0, 255, 255])],  # yellow
        )


        i = 0
        pose_idx = 0
        stop = False
        reachable = False
        force_global = False
        while True:
            goal_map = self.get_goal_map(
                traversible, goal_instance_map, viewpoint_loc, is_local, pose_idx
            )
            if goal_map is None:
                # This mean we couldn't find any traversible pose. Even with the minimum dilation radius.
                break

            logger.info(
                f"Trying to plan to instance goal with\n\t - pose_idx: {pose_idx}\n\t - {'local' if is_local else 'global'}\n\t - obs dilation: {self.curr_obs_dilation_selem_radius}"
            )

            (
                reachable,
                stop,
                short_term_goal,
                closest_goal_pt,
            ) = self._get_short_term_goal(
                traversible,
                goal_map,
                robot_loc,
                postfix=f"_attempt_{i}_pose_{pose_idx}{'' if is_local else '_global'}{postfix}",
            )

            if stop or reachable:
                break

            i += 1

            logger.info("Could not find a path to the high-level goal.")
            if is_local:
                force_global = True
            else:
                force_global = False
                traversible, success = self.decrease_obstacle_dilation_radius(
                    traversible, obstacle_map, is_local
                    )
                if not success:
                    pose_idx += 1
                    self.reset_obs_dilation_selem_radius()
            goal_instance_map, viewpoint_loc, viewpoint_orientation, is_local, obstacle_map, robot_loc, traversible = self.get_instance_planning_maps(instance_goal_id, force_global=force_global)


        if reachable:
            logger.debug(f"Planning to instance goal successfull.")
            if stop:
                logger.debug(f"We need to stop.")
            else:
                logger.debug(f"Short term goal: {short_term_goal}")

        vis_input = {}
        if reachable:
            vis_input = {
                "closest_goal_pt": closest_goal_pt,
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
            closest_goal_pt,
            viewpoint_orientation,
            is_local,
            vis_input,
        )

    def get_goal_instance_map_and_viewpoint(self, instance_goal_id, force_global=False):
        instance_views = self.instance_memory.instances[instance_goal_id].instance_views
        best_view = np.argmax([view.object_coverage for view in instance_views])
        instance_pose = instance_views[best_view].pose
        viewpoint_global_location = self.semantic_map.global_pose_to_global_location(
            instance_pose
        )
        viewpoint_local_location = self.semantic_map.global_pose_to_local_location(
            instance_pose
        )

        if not force_global:
            is_local = self.semantic_map.is_location_in_local_map(viewpoint_local_location)
        
            instances_map = self.semantic_map.get_instances_map(local=True)
            global_instances_map = self.semantic_map.get_instances_map(local=False)
            inst_map_idx = instances_map == instance_goal_id
            inst_map_idx = np.argmax(np.sum(inst_map_idx, axis=(1, 2)))
            goal_instance_map = (instances_map[inst_map_idx] == instance_goal_id).astype(
                int
            )
            global_goal_instance_map = (global_instances_map[inst_map_idx] == instance_goal_id).astype(
                int
            )
            is_local = is_local and np.any(goal_instance_map) and np.count_nonzero(goal_instance_map) == np.count_nonzero(global_goal_instance_map)
            logger.debug(f"Getting goal instance map for instance {instance_goal_id} with is_local={is_local}, and local nonzero count {np.count_nonzero(goal_instance_map)} and global nonzero count {np.count_nonzero(global_goal_instance_map)}.")

            if is_local:
                logger.debug(f">>> Goal instance {instance_goal_id} present in local map.")
                logger.debug(
                    f">>> viewpoint location is: {viewpoint_local_location}, with coverage {instance_views[best_view].object_coverage}."
                )
                return goal_instance_map, viewpoint_local_location, instance_pose[2], True


        instances_map = self.semantic_map.get_instances_map(local=False)
        inst_map_idx = instances_map == instance_goal_id
        inst_map_idx = np.argmax(np.sum(inst_map_idx, axis=(1, 2)))
        goal_instance_map = (instances_map[inst_map_idx] == instance_goal_id).astype(
            int
        )

        return goal_instance_map, viewpoint_global_location, instance_pose[2], False

    def get_instance_planning_maps(self, instance_goal_id, force_global=False):
        goal_instance_map, viewpoint_loc, viewpoint_orientation, is_local = (
            self.get_goal_instance_map_and_viewpoint(instance_goal_id, force_global=force_global)
        )
        goal_instance_map = self.get_largest_cluster(goal_instance_map, is_local)
        obstacle_map = self.semantic_map.get_obstacle_map(is_local)
        robot_loc = (
            self.semantic_map.local_loc if is_local else self.semantic_map.global_loc
        )
        traversible = self.get_traversible(obstacle_map, is_local)
        return goal_instance_map, viewpoint_loc, viewpoint_orientation, is_local, obstacle_map, robot_loc, traversible

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
            logger.info(
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
