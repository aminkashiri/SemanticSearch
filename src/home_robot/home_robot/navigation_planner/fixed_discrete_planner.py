# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import os
import shutil
import time
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology

import home_robot.utils.pose as pu
from home_robot.core.interfaces import (
    ContinuousNavigationAction,
    DiscreteNavigationAction,
)
from home_robot.utils.geometry import xyt_global_to_base

from .fmm_planner import FMMPlanner

from home_robot.utils.logger import get_logger

logger = get_logger()

CM_TO_METERS = 0.01


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
        goal_dilation_selem_radius: int,
        map_size_cm: int,
        map_resolution: int,
        visualize: bool,
        print_images: bool,
        dump_location: str,
        exp_name: str,
        min_goal_distance_cm: float = 50.0,
        min_obs_dilation_selem_radius: int = 1,
        agent_cell_radius: int = 1,
        map_downsample_factor: float = 1.0,
        map_update_frequency: int = 1,
        goal_tolerance: float = 0.01,  # for sim
        discrete_actions: bool = True,
        continuous_angle_tolerance: float = 30.0,
        geodesic_dilation=False,
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
        self.goal_dilation_selem_radius = goal_dilation_selem_radius
        self.min_obs_dilation_selem_radius = min_obs_dilation_selem_radius
        self.agent_cell_radius = agent_cell_radius
        self.goal_tolerance = goal_tolerance
        self.continuous_angle_tolerance = continuous_angle_tolerance

        self.vis_dir = None
        self.collision_map = None
        self.visited_map = None
        self.col_width = None
        self.last_pose = None
        self.curr_pose = None
        self.last_action = None
        self.timestep = 0
        self.curr_obs_dilation_selem_radius = None
        self.obs_dilation_selem = None
        self.min_goal_distance_cm = min_goal_distance_cm
        self.dd = None
        self.reached_goal_candidate = False  # to keep track of whether goal has been reached – for stacking additional checks to confirm whether goal is correct

        self.map_downsample_factor = map_downsample_factor
        self.map_update_frequency = map_update_frequency

        self.geodesic_dilation = geodesic_dilation

    def reset(self):
        self.vis_dir = self.default_vis_dir
        self.collision_map = np.zeros(self.map_shape)
        self.visited_map = np.zeros(self.map_shape)
        self.col_width = 1
        self.last_pose = None
        self.curr_pose = [
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
        self.goal_dilation_selem = skimage.morphology.disk(
            self.goal_dilation_selem_radius
        )
        self.reached_goal_candidate = False

    def set_vis_dir(self, scene_id: str, episode_id: str):
        self.vis_dir = os.path.join(self.default_vis_dir, f"{scene_id}_{episode_id}")
        shutil.rmtree(self.vis_dir, ignore_errors=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def disable_print_images(self):
        self.print_images = False

    def plan(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        frontier_map: np.ndarray,
        sensor_pose: np.ndarray,
        found_goal: bool,
        goal_pose: List[float] = None,
        debug: bool = False,
        use_dilation_for_stg: bool = False,
        timestep: int = None,
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
        # Reset timestep using argument; useful when there are timesteps where the discrete planner is not invoked
        assert timestep is not None
        self.timestep = timestep

        self.last_pose = self.curr_pose
        obstacle_map = np.rint(obstacle_map)

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = sensor_pose
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        #! This is actually correct as far as I get. lmb is [y1,y2,x1,x2], so this makes sense.
        start = [
            int(start_y * 100.0 / self.map_resolution - gx1),
            int(start_x * 100.0 / self.map_resolution - gy1),
        ]

        logger.info(f"---- Starting planning with start pose: {start} ---- ")
        logger.info(f"> Sensor pose: {sensor_pose.tolist()}")
        logger.info(f"> Found goal: {found_goal}")
        logger.info(f"> Goal points provided: {np.any(goal_map > 0)}")

        start = pu.threshold_poses(start, obstacle_map.shape)
        start = np.array(start)

        if self.print_images:
            self.visualize_input(frontier_map, obstacle_map, goal_map, start)

        self.curr_pose = [start_x, start_y, start_o]
        self.visited_map[gx1:gx2, gy1:gy2][
            start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1
        ] = 1

        # Check collisions if we have just moved and are uncertain
        if self.last_action == DiscreteNavigationAction.MOVE_FORWARD or (
            type(self.last_action) == ContinuousNavigationAction
            and np.linalg.norm(self.last_action.xyt[:2]) > 0
        ):
            self._check_collision()

        (
            short_term_goal,
            closest_goal_map,
            replan,
            stop,
            closest_goal_pt,
            dilated_obstacles,
        ) = self._get_short_term_goal(
            obstacle_map,
            np.copy(goal_map),
            start,
            planning_window,
            frontier_map=frontier_map,
            goal_pose=goal_pose,
        )

        # Short term goal is in cm, start_x and start_y are in m
        logger.debug(f"Current pose: {start}")
        logger.debug(f"Short term goal: {short_term_goal}")
        logger.debug(f"Delta = {short_term_goal[0] - start[0]} , {short_term_goal[1] - start[1]}")
        dist_to_short_term_goal = np.linalg.norm(start - np.array(short_term_goal[:2]))
        logger.debug(f"Distance (m): { dist_to_short_term_goal * self.map_resolution * CM_TO_METERS}")
        logger.debug(f"Replan: {replan}")

        i = 0
        while replan and not stop:
            i += 1
            logger.info(
                "Could not find a path to the high-level goal. Trying to replan"
            )
            # Clean collision map
            self.collision_map *= 0

            # Reduce obstacle dilation
            if self.curr_obs_dilation_selem_radius > self.min_obs_dilation_selem_radius:
                self.curr_obs_dilation_selem_radius -= 1
                self.obs_dilation_selem = skimage.morphology.disk(
                    self.curr_obs_dilation_selem_radius
                )
                logger.debug(
                    f"reduced obs dilation to: {self.curr_obs_dilation_selem_radius}"
                )
                (
                    short_term_goal,
                    closest_goal_map,
                    replan,
                    stop,
                    closest_goal_pt,
                    dilated_obstacles,
                ) = self._get_short_term_goal(
                    obstacle_map,
                    np.copy(goal_map),
                    start,
                    planning_window,
                    frontier_map=frontier_map,
                    goal_pose=goal_pose,
                    postfix=f"_replan_{i}",
                )
            else:
                logger.debug(
                    f"Obstacle dilation already at minimum: {self.min_obs_dilation_selem_radius}, but still couldn't find a path to the goal."
                )
                #! Note that found goal means we have at least found some goal category (not necessarily an instance)
                if found_goal == True:
                    logger.info(
                        "Explore frontier map instead of goal map, as we couldn't find a path to the goal."
                    )
                    (
                        short_term_goal,
                        closest_goal_map,
                        replan,
                        stop,
                        closest_goal_pt,
                        dilated_obstacles,
                    ) = self._get_short_term_goal(
                        obstacle_map,
                        np.copy(frontier_map),
                        start,
                        planning_window,
                        frontier_map=frontier_map,
                        goal_pose=goal_pose,
                        postfix="_frontier",
                    )
                    found_goal = False
                    if replan:
                        logger.info("Could not find a path to the frontier goal either, returning stop.")
                        #  TODO separate out STOP_SUCCESS and STOP_FAILURE actions
                        return (
                            DiscreteNavigationAction.STOP,
                            closest_goal_map,
                            short_term_goal,
                            dilated_obstacles,
                            replan,
                            stop,
                        )
                else:
                    logger.info(f"Could not find a path to any frontier goal either with min obs dilation {self.min_obs_dilation_selem_radius}, returning stop.")
                    return (
                        DiscreteNavigationAction.STOP,
                        closest_goal_map,
                        short_term_goal,
                        dilated_obstacles,
                        replan,
                        stop,
                    )


        # Normalize agent angle
        angle_agent = pu.normalize_angle(start_o)

        # If we found a short term goal worth moving towards...
        stg_x, stg_y = short_term_goal
        relative_stg_x, relative_stg_y = stg_x - start[0], stg_y - start[1]
        angle_st_goal = math.degrees(math.atan2(relative_stg_x, relative_stg_y))
        relative_angle_to_stg = pu.normalize_angle(angle_agent - angle_st_goal)

        # Compute angle to the final goal
        goal_x, goal_y = closest_goal_pt
        angle_goal = math.degrees(math.atan2(goal_x - start[0], goal_y - start[1]))

        if goal_pose is None:
            # Compute angle to the final goal
            relative_angle_to_closest_goal = pu.normalize_angle(
                angle_agent - angle_goal
            )
        else:
            relative_angle_to_closest_goal = pu.normalize_angle(
                angle_agent - goal_pose[0]
            )

        if debug:
            # Actual metric distance to goal
            distance_to_goal = np.linalg.norm(np.array([goal_x, goal_y]) - start)
            distance_to_goal_cm = distance_to_goal * self.map_resolution
            # Display information
            print("Found reachable goal:", found_goal)
            print("Stop:", stop)
            print("Angle to goal:", relative_angle_to_closest_goal)
            print("Distance to goal", distance_to_goal)
            print(
                "Distance in cm:",
                distance_to_goal_cm,
                ">",
                self.min_goal_distance_cm,
            )

            m_relative_stg_x, m_relative_stg_y = [
                CM_TO_METERS * self.map_resolution * d
                for d in [relative_stg_x, relative_stg_y]
            ]
            print("continuous actions for exploring")
            print("agent angle =", angle_agent)
            print("angle stg goal =", angle_st_goal)
            print("angle final goal =", relative_angle_to_closest_goal)
            print(
                m_relative_stg_x, m_relative_stg_y, "rel ang =", relative_angle_to_stg
            )

        action = self.get_action(
            relative_stg_x,
            relative_stg_y,
            relative_angle_to_stg,
            relative_angle_to_closest_goal,
            start_o,
            found_goal,
            stop,
            debug,
        )
        if debug:
            print("Replan: ", replan)
            print("Action:", action)
            print("--- End Planning ---")

        self.last_action = action
        return (
            action,
            closest_goal_map,
            short_term_goal,
            dilated_obstacles,
            replan,
            stop,
        )

    def get_action(
        self,
        relative_stg_x: float,
        relative_stg_y: float,
        relative_angle_to_stg: float,
        relative_angle_to_closest_goal: float,
        start_compass: float,
        found_goal: bool,
        stop: bool,
        debug: bool,
    ):
        """
        Gets discrete/continuous action given short-term goal. Agent orients to closest goal if found_goal=True and stop=True
        """
        # Short-term goal -> deterministic local policy
        if not (found_goal and stop):
            if self.discrete_actions:
                if relative_angle_to_stg > self.turn_angle / 2.0:
                    action = DiscreteNavigationAction.TURN_RIGHT
                elif relative_angle_to_stg < -self.turn_angle / 2.0:
                    action = DiscreteNavigationAction.TURN_LEFT
                else:
                    action = DiscreteNavigationAction.MOVE_FORWARD
            else:
                # Use the short-term goal to set where we should be heading next
                m_relative_stg_x, m_relative_stg_y = [
                    CM_TO_METERS * self.map_resolution * d
                    for d in [relative_stg_x, relative_stg_y]
                ]
                if np.abs(relative_angle_to_stg) > self.turn_angle / 2.0:
                    # Must return commands in radians and meters
                    relative_angle_to_stg = math.radians(relative_angle_to_stg)
                    action = ContinuousNavigationAction([0, 0, -relative_angle_to_stg])
                else:
                    # Must return commands in radians and meters
                    relative_angle_to_stg = math.radians(relative_angle_to_stg)
                    xyt_global = [
                        m_relative_stg_y,
                        m_relative_stg_x,
                        -relative_angle_to_stg,
                    ]

                    xyt_local = xyt_global_to_base(
                        xyt_global, [0, 0, math.radians(start_compass)]
                    )
                    xyt_local[2] = (
                        -relative_angle_to_stg
                    )  # the original angle was already in base frame
                    action = ContinuousNavigationAction(xyt_local)
        else:
            # Try to orient towards the goal object - or at least any point sampled from the goal
            # object.
            if debug:
                print()
                print("----------------------------")
                print(">>> orient towards the goal:", relative_angle_to_closest_goal)
            if self.discrete_actions:
                if relative_angle_to_closest_goal > 2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_RIGHT
                elif relative_angle_to_closest_goal < -2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_LEFT
                else:
                    self.reached_goal_candidate = True
                    action = DiscreteNavigationAction.STOP
            elif (
                np.abs(relative_angle_to_closest_goal) > self.continuous_angle_tolerance
            ):
                if debug:
                    print("Continuous rotation towards goal point")
                relative_angle_to_closest_goal = math.radians(
                    relative_angle_to_closest_goal
                )
                action = ContinuousNavigationAction(
                    [0, 0, -relative_angle_to_closest_goal]
                )
            else:
                action = DiscreteNavigationAction.STOP
                if debug:
                    print("!!! DONE !!!")

        return action

    def _get_short_term_goal(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        start: List[int],
        planning_window: List[int],
        frontier_map=None,
        goal_pose: List[float] = None,
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
        logger.info(f"Getting short-term goal")
        gx1, gx2, gy1, gy2 = planning_window
        (
            x1,
            y1,
        ) = (
            0,
            0,
        )
        x2, y2 = obstacle_map.shape
        obstacles = obstacle_map[x1:x2, y1:y2]

        # Dilate obstacles
        dilated_obstacles = cv2.dilate(obstacles, self.obs_dilation_selem, iterations=1)

        # Create inverse map of obstacles - this is territory we assume is traversible
        # Traversible is now the map
        traversible = 1 - dilated_obstacles
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
        agent_rad = self.agent_cell_radius
        traversible[
            int(start[0] - x1) - agent_rad : int(start[0] - x1) + agent_rad + 1,
            int(start[1] - y1) - agent_rad : int(start[1] - y1) + agent_rad + 1,
        ] = 1
        traversible = add_boundary(traversible)
        goal_map = add_boundary(goal_map, value=0)
        planner = FMMPlanner(
            traversible,
            step_size=self.step_size,
            vis_dir=self.vis_dir,
            print_images=self.print_images,
            goal_tolerance=self.goal_tolerance,
            geodesic_dilation=self.geodesic_dilation,
            vis_postfix=postfix,
        )
        goal_map = planner.change_goal_map_to_closest_traversible_from_past_pose(
            goal_map, goal_pose, planning_window, self.timestep
        )

        #! This is dilation logic.
        navigable_goal_map = planner._dilate_goal(
            goal_map,
            self.min_goal_distance_cm / self.map_resolution,
            timestep=self.timestep
        )
        if not np.any(navigable_goal_map):
            logger.info(
                f"Couldn't find any navigable goal points in the map. Using frontier map instead."
            )
            frontier_map = add_boundary(frontier_map, value=0)
            navigable_goal_map = frontier_map

        self.dd = planner.set_multi_goal(
            navigable_goal_map,
            self.timestep,
            self.dd,
            self.map_downsample_factor,
            self.map_update_frequency,
            number="5"
        )
        goal_distance_map, closest_goal_pt = self.get_closest_goal(goal_map, start)

        #! myTODO: Looks like this is no needed
        # self.timestep += 1

        state = [start[0] - x1 + 1, start[1] - y1 + 1]

        # This is where we create the planner to get the trajectory to this state
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(
            state, continuous=(not self.discrete_actions), timestep=self.timestep
        )
        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        short_term_goal = int(stg_x), int(stg_y)

        if self.print_images:
            _navigable_goal_map = navigable_goal_map.copy()
            _navigable_goal_map = _navigable_goal_map.astype(np.uint8)
            # _traversible = traversible.astype(np.uint8)

            white = np.ones((_navigable_goal_map.shape + (3,)), dtype=np.uint8) * 255
            white[traversible == 0] = [0, 0, 0]
            white[_navigable_goal_map == 1] = [0, 0, 255]
            white[start[0] + 1, start[1] + 1] = [0, 255, 0]
            white[int(stg_x) + 1, int(stg_y) + 1] = [255, 0, 0]

            # logger.debug(f"SAVING 7.stg")
            cv2.imwrite(
                os.path.join(self.vis_dir, f"{self.timestep}_7.stg{postfix}.png"),
                np.flipud(white),
            )

        return (
            short_term_goal,
            goal_distance_map,
            replan,
            stop,
            closest_goal_pt,
            dilated_obstacles,
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
        return closest_goal_map, closest_goal_pt

    def _check_collision(self):
        """Check whether we had a collision and update the collision map."""
        x1, y1, t1 = self.last_pose
        x2, y2, _ = self.curr_pose
        buf = 4
        length = 2

        # You must move at least 5 cm when doing forward actions
        # Otherwise we assume there has been a collision
        if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
            self.col_width += 2
            if self.col_width == 7:
                length = 4
                buf = 3
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
                    wx = x1 + 0.05 * (
                        (i + buf) * np.cos(np.deg2rad(t1))
                        + (j - width // 2) * np.sin(np.deg2rad(t1))
                    )
                    wy = y1 + 0.05 * (
                        (i + buf) * np.sin(np.deg2rad(t1))
                        - (j - width // 2) * np.cos(np.deg2rad(t1))
                    )
                    r, c = wy, wx
                    r, c = int(r * 100 / self.map_resolution), int(
                        c * 100 / self.map_resolution
                    )
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    self.collision_map[r, c] = 1

    def visualize_input(self, frontier_map, obstacle_map, goal_map, start):
        fm = np.flipud(frontier_map.squeeze())
        om = np.flipud(obstacle_map.squeeze())
        gm = np.flipud(goal_map.squeeze())
        H, W = fm.shape
        vis_map = np.ones((H, W, 3), dtype=np.uint8) * 255
        vis_map[om == 1] = [0, 0, 0]
        vis_map[fm == 1] = [
            255,
            0,
            0,
        ]  # Frontier is red, but sometimes replaced by goal which is blue if we have no other goal.
        vis_map[gm == 1] = [0, 0, 255]  # Goal is blue
        vis_map[np.logical_and(gm == 1, om == 1)] = [255, 0, 255]  # purple
        vis_map[start[0], start[1]] = [0, 255, 0]
        # logger.debug(f"SAVING 1.planning_input.png")
        cv2.imwrite(
            os.path.join(self.vis_dir, f"{self.timestep}_1.planning_input.png"),
            vis_map[..., ::-1].astype(int),
        )