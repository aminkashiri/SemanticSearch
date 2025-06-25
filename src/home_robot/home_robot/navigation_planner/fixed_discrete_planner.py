# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import os
import shutil
import torch
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

from bresenham import bresenham
from home_robot.utils.visualization import visualize_map
from home_robot.utils.logger import get_logger
logger = get_logger()



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
        agent_cell_radius: int = 1,
        map_downsample_factor: float = 1.0,
        map_update_frequency: int = 1,
        goal_tolerance: float = 0.01,  # for sim
        discrete_actions: bool = True,
        continuous_angle_tolerance: float = 30.0,
        panorama_start_steps: int = 0,
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
        self.agent_cell_radius = agent_cell_radius
        self.goal_tolerance = goal_tolerance
        self.continuous_angle_tolerance = continuous_angle_tolerance

        self.vis_dir = None
        self.collision_map = None
        self.visited_map = None
        self.col_width = None
        self.last_global_pose = None
        self.curr_global_pose = None
        self.last_action = None
        self.timestep = 0
        self.curr_obs_dilation_selem_radius = None
        self.obs_dilation_selem = None
        self.min_goal_distance_cm = min_goal_distance_cm
        self.dd = None
        self.reached_goal_candidate = False  # to keep track of whether goal has been reached – for stacking additional checks to confirm whether goal is correct

        self.map_downsample_factor = map_downsample_factor
        self.map_update_frequency = map_update_frequency
        self.panorama_start_steps=panorama_start_steps

    def reset(self):
        self.vis_dir = self.default_vis_dir
        self.collision_map = np.zeros(self.map_shape)
        self.visited_map = np.zeros(self.map_shape)
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
        obstacle_map: np.ndarray,
        frontier_map: np.ndarray,
        global_pose: np.ndarray,
        lmb: np.ndarray,
        instance_goal_found: bool,
        instance_map: np.ndarray,
        view_loc: List[float] = None,
        timestep: int = None,
        total_timesteps: int =None,
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
        reset_module = False
        #! Note: All the maps are local maps
        assert timestep is not None
        self.timestep = timestep

        if instance_goal_found:
            self.episode_panorama_start_steps = 0

        if total_timesteps < self.episode_panorama_start_steps:
            #! When total_timesteps is less than the panorama start steps, we just turn right. So if turn angle is 30, at 12th step, we don't need to turn anymore.
            return (
                DiscreteNavigationAction.TURN_RIGHT,
                None,
                None,
                None,
                True, # Means we have found a path, don't stop.
                False, #reset module
            )

        self.last_global_pose = self.curr_global_pose
        self.curr_global_pose = global_pose
        obstacle_map = np.rint(obstacle_map)

        planning_window = lmb.tolist()

        #! This is actually correct as far as I get. lmb is [y1,y2,x1,x2], so this makes sense.
        local_loc = [
            int(self.curr_global_pose[1] * 100.0 / self.map_resolution - lmb[0]),
            int(self.curr_global_pose[0] * 100.0 / self.map_resolution - lmb[2]),
        ]

        logger.info(f"---- Starting planning with local loc (pixels): {local_loc} ---- ")
        logger.info(f"> global pose: {global_pose.tolist()}")
        logger.info(f"> Instance goal found: {instance_goal_found}")
        if instance_map is not None:
            assert np.any(instance_map > 0)



        start_thresholded = pu.threshold_poses(local_loc, obstacle_map.shape)
        if start_thresholded[0] != local_loc[0] or start_thresholded[1] != local_loc[1]:
            logger.warning(f"location is changed because it was out of map. Init location: {local_loc}, new location: {start_thresholded}")
        local_loc = start_thresholded
        local_loc = np.array(local_loc)

        if self.print_images:
            instance_on_obstacles = False
            if not instance_map is None:
                instance_on_obstacles = np.logical_and(instance_map == 1, obstacle_map == 1)
            visualize_map(
                obstacle_map.shape, 
                self.vis_dir,
                f"{self.timestep}_1.planning_input.png",
                points=[(local_loc, [255, 0, 0])],
                traversible=1-obstacle_map,
                goal_map=instance_map,
                frontier_map=frontier_map,
                features=[(instance_on_obstacles,[0,255,255])], # yellow
            )


        self.visited_map[lmb[0]:lmb[1], lmb[2]:lmb[3]][local_loc[0], local_loc[1]] = 1

        # Check collisions if we have just moved and are uncertain
        if self.last_action == DiscreteNavigationAction.MOVE_FORWARD or (
            type(self.last_action) == ContinuousNavigationAction
            and np.linalg.norm(self.last_action.xyt[:2]) > 0
        ):
            self._check_collision()


        reachable = False
        stop = False
        i = 0
        traversible = self.get_traversible(obstacle_map, planning_window, local_loc)
        logger.info(f"Trying to plan with obs dilation: {self.curr_obs_dilation_selem_radius}.")
        if instance_goal_found == True:
            pose_idx = 0
            while True:
                goal_map = self.get_goal_map_from_goal_pose(traversible, instance_map, view_loc, planning_window, pose_idx, self.timestep)
                if goal_map is None:
                    # This mean we couldn't find any traversible pose. We should go to frontiers
                    #! myTODO: Should I remove this instance goal?
                    break

                (
                    short_term_goal,
                    closest_goal_map,
                    reachable,
                    stop,
                    closest_goal_pt,
                ) = self._get_short_term_goal(
                    traversible,
                    goal_map,
                    local_loc,
                    postfix=f"_replan_{i}" if i > 0 else "",
                )

                if stop or reachable:
                    break
                
                i += 1
                logger.info(
                    "Could not find a path to the high-level goal. Trying to replan"
                )
                # self.collision_map *= 0
                if self.curr_obs_dilation_selem_radius > self.min_obs_dilation_selem_radius:
                    self.curr_obs_dilation_selem_radius -= 1
                    self.obs_dilation_selem = skimage.morphology.disk(
                        self.curr_obs_dilation_selem_radius
                    )
                    logger.info(f"Decreasing obstacle dilation radius to {self.curr_obs_dilation_selem_radius}")
                    traversible = self.get_traversible(obstacle_map, planning_window, local_loc)
                else:
                    logger.info(f"Obstacle dilation radius is already at minimum, trying next pose.")
                    pose_idx += 1


        if not stop and reachable:
            logger.debug(f"Found a path to the high-level goal")
            logger.debug(f"Short term goal: {short_term_goal}")
            logger.debug(f"Delta = [{short_term_goal[0] - local_loc[0]}, {short_term_goal[1] - local_loc[1]}]")
            dist_to_short_term_goal = np.linalg.norm(local_loc - np.array(short_term_goal[:2]))
            logger.debug(f"Distance to stg: { dist_to_short_term_goal * self.map_resolution / 100} m")
        elif not (stop or reachable):
            if instance_goal_found:
                logger.debug("Couldn't find any path to instance goal. Resetting module so it doesn't use this instance goal anymore.")
                reset_module = True
            else:
                logger.debug("No instance goal provided.")
            if frontier_map.any():
                logger.debug("Trying frontiers instead.")
                (
                    short_term_goal,
                    closest_goal_map,
                    reachable,
                    stop,
                    closest_goal_pt,
                ) = self._get_short_term_goal(
                    traversible,
                    frontier_map,
                    local_loc,
                    postfix="_frontier",
                )
            else:
                logger.info("No frontier map available.")
            if not reachable:
                logger.info("Could not find a path to the frontier goal either.")

        if not (stop or reachable):
            return (
                None,
                closest_goal_map,
                short_term_goal,
                1- traversible,
                reachable,
                reset_module
            )


        # Normalize agent angle
        angle_agent = pu.normalize_angle(self.curr_global_pose[2])

        # If we found a short term goal worth moving towards...
        stg_x, stg_y = short_term_goal
        relative_stg_x, relative_stg_y = stg_x - local_loc[0], stg_y - local_loc[1]
        angle_st_goal = math.degrees(math.atan2(relative_stg_x, relative_stg_y))
        relative_angle_to_stg = pu.normalize_angle(angle_agent - angle_st_goal)

        # Compute angle to the final goal
        goal_x, goal_y = closest_goal_pt
        angle_goal = math.degrees(math.atan2(goal_x - local_loc[0], goal_y - local_loc[1]))

        if view_loc is None:
            # Compute angle to the final goal
            relative_angle_to_closest_goal = pu.normalize_angle(
                angle_agent - angle_goal
            )
        else:
            relative_angle_to_closest_goal = pu.normalize_angle(
                angle_agent - view_loc[2]
            )

            # # Actual metric distance to goal
            # distance_to_goal = np.linalg.norm(np.array([goal_x, goal_y]) - local_start_pose)
            # distance_to_goal_cm = distance_to_goal * self.map_resolution
            # # Display information
            # print("Angle to goal:", relative_angle_to_closest_goal)
            # print("Distance to goal", distance_to_goal)
            # print(
            #     "Distance in cm:",
            #     distance_to_goal_cm,
            # )

            # m_relative_stg_x, m_relative_stg_y = [
            #     CM_TO_METERS * self.map_resolution * d
            #     for d in [relative_stg_x, relative_stg_y]
            # ]
            # print("continuous actions for exploring")
            # print("agent angle =", angle_agent)
            # print("angle stg goal =", angle_st_goal)
            # print("angle final goal =", relative_angle_to_closest_goal)
            # print(
            #     m_relative_stg_x, m_relative_stg_y, "rel ang =", relative_angle_to_stg
            # )

        action = self.get_action(
            relative_stg_x,
            relative_stg_y,
            relative_angle_to_stg,
            relative_angle_to_closest_goal,
            self.curr_global_pose[2],
            stop,
        )

        self.last_action = action
        return (
            action,
            closest_goal_map,
            short_term_goal,
            1-traversible,
            reachable,
            reset_module
        )

    def get_action(
        self,
        relative_stg_x: float,
        relative_stg_y: float,
        relative_angle_to_stg: float,
        relative_angle_to_closest_goal: float,
        start_compass: float,
        stop: bool,
    ):
        """
        Gets discrete/continuous action given short-term goal. Agent orients to closest goal if found_goal=True and stop=True
        """
        # stop == True, orient towards goal first, then actually stop. 
        if stop == False:
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
                    100 * self.map_resolution * d
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
            logger.debug("----------------------------")
            logger.debug(">>> orienting towards the goal: {relative_angle_to_closest_goal}")
            if self.discrete_actions:
                if relative_angle_to_closest_goal > 2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_RIGHT
                elif relative_angle_to_closest_goal < -2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_LEFT
                else:
                    logger.debug("Already toward the goal, stopping.")
                    action = DiscreteNavigationAction.STOP
            elif (
                np.abs(relative_angle_to_closest_goal) > self.continuous_angle_tolerance
            ):
                relative_angle_to_closest_goal = math.radians(
                    relative_angle_to_closest_goal
                )
                action = ContinuousNavigationAction(
                    [0, 0, -relative_angle_to_closest_goal]
                )
            else:
                action = DiscreteNavigationAction.STOP
                print("!!! DONE !!!")

        return action

    def get_traversible(self, obstacles, planning_window, local_loc):
        gx1, gx2, gy1, gy2 = planning_window
        dilated_obstacles = cv2.dilate(obstacles, self.obs_dilation_selem, iterations=1)

        # Create inverse map of obstacles - this is territory we assume is traversible
        # Traversible is now the map
        traversible = 1 - dilated_obstacles
        traversible[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 0
        traversible[self.visited_map[gx1:gx2, gy1:gy2] == 1] = 1
        agent_rad = self.agent_cell_radius
        traversible[
            local_loc[0] - agent_rad : local_loc[0] + agent_rad + 1,
            local_loc[1] - agent_rad : local_loc[1] + agent_rad + 1,
        ] = 1
        return traversible

    def get_goal_map_from_goal_pose(self, traversible, instance_map, view_loc, planning_window, index, timestep):
        """
        Args:
            goal_map
            view_pose: Global loc that we can see the goal instance.
        """
        logger.info(f"Creating goal map using past view. Choosing {index}th traversible pose.")
        assert not view_loc is None

        assert instance_map.shape == traversible.shape

        # no plus 1. No boundary yet
        local_view_loc = np.array([
            view_loc[0] - planning_window[0],
            view_loc[1] - planning_window[2]
        ])
        logger.debug(f"Global view loc is : {view_loc}, local view loc is : {local_view_loc}")

        # Find closest goal_map cell to goal_pose
        goal_indices = np.argwhere(instance_map == 1)
        if goal_indices.size == 0:
            raise Exception("No instance cells found in goal_map, should not happen.")

        # Find closest goal cell in goal_map to the goal_pose
        dists = np.linalg.norm(goal_indices - local_view_loc[None, :], axis=1)
        closest_instance_idx = goal_indices[np.argmin(dists)]
        logger.debug(f"Closest goal index in goal_map to goal_pose is ({closest_instance_idx})")

        # Generate Bresenham line from closest_goal_idx to goal_pose  
        line_coords = list(bresenham(closest_instance_idx[0], closest_instance_idx[1],local_view_loc[0], local_view_loc[1]))

        first_pixels = []
        in_segment = False

        for x, y in line_coords:
            if not (0 <= x < traversible.shape[0] and 0 <= y < traversible.shape[1]):
                continue

            if traversible[x, y] == 1:
                if not in_segment:
                    # Start of a new segment
                    logger.info(f"Adding traversible pose {x, y}")
                    first_pixels.append((x, y))
                    in_segment = True
            else:
                in_segment = False
        

        if not (0 <= local_view_loc[0] < traversible.shape[0] and 0 <= local_view_loc[1] < traversible.shape[1]):
            first_pixels.append(local_view_loc)

        if index >= len(first_pixels):
            logger.info(f"No traversible view found for the instance goal.")
            return None

        goal_map = np.zeros_like(instance_map, dtype=np.uint8)
        goal_loc = first_pixels[index]
        goal_map[goal_loc[0], goal_loc[1]] = 1
        self.visualize_converting_goal_to_pose(traversible, instance_map, first_pixels, local_view_loc, goal_loc, index, timestep)
        return goal_map

    def visualize_converting_goal_to_pose(self,traversible, instance_map, first_pixels, local_view_loc, goal_loc, index, timestep):
        h, w = traversible.shape
        vis_img = np.ones((h, w, 3), dtype=np.uint8) * 255 
        vis_img[traversible == 0] = [0, 0, 0]

        # orange - All instance cells
        vis_img[instance_map == 1] = [0, 165, 255]

        for pixel in first_pixels:
            # green - All poses
            vis_img[pixel[0], pixel[1]] = [0, 255, 0]

        if not (0 <= local_view_loc[0] < traversible.shape[0] and 0 <= local_view_loc[1] < traversible.shape[1]):
            # blue
            logger.warning(f"Don't visualizing local view loc, because it is out of bound (GOAL is not in local map)")
            vis_img[local_view_loc[0], local_view_loc[1]] = [255, 0,0]

        # red - final goal
        vis_img[goal_loc[0], goal_loc[1]] = [0, 0, 255]

        vis_img = np.flipud(vis_img)

        # logger.debug(f"SAVING 2.interpolate_goal")
        cv2.imwrite(
            os.path.join(self.vis_dir, f"{timestep}_2.interpolate_goal_idx{index}.png"),
            vis_img,
        )

    def _get_short_term_goal(
        self,
        traversible: np.ndarray,
        goal_map: np.ndarray,
        local_loc: List[int],
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
        goal_map = add_boundary(goal_map, value=0)
        traversible = add_boundary(traversible)
        logger.debug(f"Getting short-term goal")
        planner = FMMPlanner(
            traversible,
            step_size=self.step_size,
            vis_dir=self.vis_dir,
            print_images=self.print_images,
            goal_tolerance=self.goal_tolerance,
            vis_postfix=postfix,
        )

        # navigable_goal_map = goal_map
        navigable_goal_map= np.logical_and(goal_map, traversible)
        #! myTODO
        assert np.any(navigable_goal_map)
        # logger.info(
        #     f"Couldn't find any navigable goal points in the map. returning replan=True, stop=False, to try again with next best option (lower dilation, frontier, best goal)."
        # )

        #* Previously they had another logic of dilating goal similar to obstacles too (cv2.dilate(sel)). I don't see much difference, but I can think more later
        dilated_goal_map = planner.dilate_goal(
            navigable_goal_map,
            self.min_goal_distance_cm / self.map_resolution,
            timestep=self.timestep
        )
        dilated_goal_map = np.logical_and(dilated_goal_map, traversible)


        self.dd = planner.set_multi_goal(
            dilated_goal_map,
            self.timestep,
            self.dd,
            self.map_downsample_factor,
            self.map_update_frequency,
            number="5"
        )

        # goal_distance_map, closest_goal_pt = self.get_closest_goal(navigable_goal_map, local_loc)
        #! myTODO: Make sure if I should use navigable goal map or dilated goal map or original goal map
        goal_distance_map, closest_goal_pt = self.get_closest_goal(navigable_goal_map, local_loc)

        #! myTODO: Looks like this is no needed
        # self.timestep += 1

        state = [local_loc[0] + 1, local_loc[1] + 1]

        # This is where we create the planner to get the trajectory to this state
        stg_x, stg_y, reachable, stop = planner.get_short_term_goal(
            state, continuous=(not self.discrete_actions), timestep=self.timestep
        )
        stg_x, stg_y = stg_x - 1, stg_y - 1

        short_term_goal = int(stg_x), int(stg_y)

        if self.print_images:
            points = [
                ([local_loc[0] + 1, local_loc[1] + 1], [255,0,0]), # start blue
                ([short_term_goal[0] + 1, short_term_goal[1] + 1], [0, 255, 0]) # stg green
            ]
            visualize_map(
                dilated_goal_map.shape, 
                self.vis_dir,
                f"{self.timestep}_7.stg{postfix}.png",
                points=points,
                traversible=traversible,
                goal_map=dilated_goal_map,
            )


        return (
            short_term_goal,
            goal_distance_map,
            reachable,
            stop,
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
        return closest_goal_map, closest_goal_pt

    def _check_collision(self):
        """Check whether we had a collision and update the collision map."""
        x1, y1, t1 = self.last_global_pose.cpu().numpy()
        x2, y2, _ = self.curr_global_pose.cpu().numpy()
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
