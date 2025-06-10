# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skfmm
import skimage
from numpy import ma

import matplotlib.cm as cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from bresenham import bresenham

from home_robot.utils.logger import get_logger
logger = get_logger()

def convert_to_cmap(subset):
    final_img = np.flipud(subset.copy()) 
    unique_vals = np.unique(final_img)
    second_max = unique_vals[-2]
    final_img[final_img==np.max(final_img)] = second_max + 1

    vmin, vmax = np.nanmin(final_img), np.nanmax(final_img)
    if vmin < 0:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = cm.get_cmap("plasma")

    rgba_img = cmap(norm(final_img)) 
    rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8) 
    return rgb_img

class FMMPlanner:
    """
    Fast Marching Method Planner.
    This is just the core FMM logic.
    """

    def __init__(
        self,
        traversible: np.ndarray,
        scale: int = 1,
        step_size: int = 5,
        goal_tolerance: float = 2.0,
        vis_dir: str = "data/images/planner",
        print_images=True,
        debug=False,
        geodesic_dilation=False,
        vis_postfix: str = "",
    ):
        """
        Arguments:
            traversible: (M + 1, M + 1) binary map encoding traversible regions
            scale: map scale
            step_size: maximum distance of the short-term goal selected by the
             planner
            vis_dir: folder where to dump visualization
        """
        self.print_images = print_images
        self.vis_dir = vis_dir
        os.makedirs(self.vis_dir, exist_ok=True)

        self.scale = scale
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        if scale != 1.0:
            self.traversible = cv2.resize(
                traversible,
                (traversible.shape[1] // scale, traversible.shape[0] // scale),
                interpolation=cv2.INTER_NEAREST,
            )
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.du = int(self.step_size / (self.scale * 1.0))
        self.fmm_dist = None
        self.debug = debug
        # self.goal_map = None
        self.geodesic_dilation = geodesic_dilation
        self.vis_postfix = vis_postfix

    def set_goal(self, goal, auto_improve: bool = False):
        """Set planner goal. Goal should be of size 2, containing x and y positions."""
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = int(goal[0] / (self.scale * 1.0)), int(
            goal[1] / (self.scale * 1.0)
        )

        if self.traversible[goal_x, goal_y] == 0.0 and auto_improve:
            goal_x, goal_y = self._find_nearest_goal([goal_x, goal_y])

        traversible_ma[goal_x, goal_y] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        return

    def set_multi_goal(
        self,
        goal_map: np.ndarray,
        timestep: int = 0,
        dd: np.ndarray = None,
        map_downsample_factor: float = 1.0,
        map_update_frequency: int = 1,
        number="",
    ):
        """Set long-term goal(s) used to compute distance from a binary
        goal map.
        dd: distance map for when we want to reuse previously computed ones (instead of updating at each step)
        map_update_frequency: skfmm.distance call made every n steps
        map_downsample_factor: 1 for no downsampling, 2 for halving both image dimensions.
        """
        assert map_downsample_factor >= 1.0
        traversible = self.traversible
        if map_downsample_factor > 1.0:
            l, w = self.traversible.shape
            traversible = cv2.resize(
                traversible,
                dsize=(int(l / map_downsample_factor), int(w / map_downsample_factor)),
            )
            print(f"Downsampling goal and traversible maps {map_downsample_factor}x.")
            goal_map_copy = goal_map.copy()
            goal_map = cv2.resize(
                goal_map,
                dsize=(int(l / map_downsample_factor), int(w / map_downsample_factor)),
                interpolation=cv2.INTER_NEAREST,
            )

            if goal_map.sum() == 0:
                # dilating goal map so as to not lose pixels when resizing
                kernel = np.ones((2, 2), np.uint8)
                goal_map = cv2.dilate(goal_map_copy, kernel, iterations=1)
                goal_map = cv2.resize(
                    goal_map,
                    dsize=(
                        int(l / map_downsample_factor),
                        int(w / map_downsample_factor),
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

        traversible_ma = ma.masked_values(traversible * 1, 0)
        traversible_ma[goal_map == 1] = 0

        # This is where we actually call the FMM algorithm!!
        # It will compute the distance from each traversible point to the goal.
        if (timestep - 1) % map_update_frequency == 0 or dd is None:
            dd = skfmm.distance(traversible_ma, dx=1 * map_downsample_factor)
            dd = ma.filled(dd, np.max(dd) + 1)
            if self.debug:
                print(f"Computing skfmm.distance (timestep: {timestep})")
        else:
            if self.debug:
                print(f"Reusing previous skfmm.distance value (timestep: {timestep})")

        if map_downsample_factor > 1.0:
            dd = cv2.resize(dd, (l, w))  # upsampling

        self.fmm_dist = dd
        # self.goal_map = goal_map

        if self.print_images and timestep != 0:
            r, c = traversible.shape  # for visualizing (downsampled) traversible map
            dist_vis = np.zeros((r, c * 3))
            dist_vis[:, :c] = np.flipud(traversible)
            dist_vis[:, c : 2 * c] = np.flipud(goal_map)
            dist_vis[:, 2 * c :] = np.flipud(self.fmm_dist / self.fmm_dist.max())

            # logger.debug(f"SAVING {number}.planner_snapshot")
            output_name = f"{timestep}_{number}.planner_snapshot{self.vis_postfix}.png"
            cv2.imwrite(
                os.path.join(self.vis_dir, output_name),
                (dist_vis * 255).astype(int),
            )
        return dd

    def visualize_get_short_term_goal(self, vis_list, timestep):
        sub_h, sub_w = vis_list[0].shape
        dist_vis = np.zeros((sub_h * 2, sub_w * 2,3))
        dist_vis[:sub_h, :sub_w] = convert_to_cmap(vis_list[0])
        dist_vis[:sub_h, sub_w : 2 * sub_w,:] = vis_list[1]
        dist_vis[sub_h:, :sub_w] = convert_to_cmap(vis_list[2])
        dist_vis[sub_h:, sub_w:] = convert_to_cmap(vis_list[3])

        # logger.debug(f"SAVING 6.get_stg")
        cv2.imwrite(
            os.path.join(self.vis_dir, f"{timestep}_6.get_stg_details{self.vis_postfix}.png"),
            (dist_vis).astype(int),
        )


    def get_short_term_goal(self, state: List[float], continuous=True, timestep=0):
        """Compute the short-term goal closest to the current state.

        Arguments:
            state: current location
        """
        scale = self.scale * 1.0
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        mask = FMMPlanner.get_mask(
            dx, dy, scale, self.step_size, min_radius=0 if continuous else None
        )
        dist_mask = FMMPlanner.get_dist(dx, dy, scale, self.step_size)

        state = [int(x) for x in state]
        # max_value = self.fmm_dist.shape[0] ** 2
        max_value = np.max(self.fmm_dist)

        #! max in self.fmm_dist actually means obstacle. (it is set to: actual_max+1)
        #! Hence, we also set padded cells to this maximum.
        dist = np.pad(
            self.fmm_dist,
            self.du,
            "constant",
            # constant_values=self.fmm_dist.shape[0] ** 2,
            constant_values=max_value
        )
        subset = dist[
            state[0] : state[0] + 2 * self.du + 1, state[1] : state[1] + 2 * self.du + 1
        ]

        assert (
            subset.shape[0] == 2 * self.du + 1 and subset.shape[1] == 2 * self.du + 1
        ), "Planning error: unexpected subset shape {}".format(subset.shape)


        #!myTODO: Input this from env 
        print_images = True

        sub_h, sub_w = subset.shape
        dist_vis = np.zeros((sub_h * 2, sub_w * 2,3))

        vis_list = []
        vis_list.append(subset.copy())

        subset *= mask
        # subset += (1 - mask) * self.fmm_dist.shape[0] ** 2
        subset += (1 - mask) * max_value

        vis_list.append(np.flipud(mask[...,None].copy()*255))
        vis_list.append(subset.copy())

        logger.debug(f"[FMM] Distance to fmm navigable goal pt = {subset[self.du, self.du] * 5}")

        stop = subset[self.du, self.du] < self.goal_tolerance
        logger.debug(f"subset[self.du, self.du] {subset[self.du, self.du]}")
        logger.debug(f"self.goal_tolerance {self.goal_tolerance}")
        logger.debug(f"stop {stop}")

        subset -= subset[self.du, self.du]
        vis_list.append(subset.copy())
        ratio1 = subset / dist_mask
        subset[ratio1 < -1.5] = 1

        if print_images:
            try:
                self.visualize_get_short_term_goal(vis_list, timestep)
            except Exception as e:
                logger.error(f"{timestep}_6.get_stg_details.png, probably because there is no way to goal: {e}")
                logger.debug(f">> Some more info:")
                logger.debug(f">> subset.shape: subset max and min: {np.max(vis_list[0])}, {np.min(vis_list[0])}")

        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        # Subset will contain negative distance to goal
        replan = subset[stg_x, stg_y] > -0.0001

        return (
            (stg_x + state[0] - self.du) * scale,
            (stg_y + state[1] - self.du) * scale,
            replan,
            stop,
        )

    def visualize_converting_goal_to_pose(self, goal_map, traversible, pose_xy, new_goal_map, timestep):
        h, w = traversible.shape
        vis_img = np.ones((h, w, 3), dtype=np.uint8) * 255 
        vis_img[traversible == 0] = [0, 0, 0]

        # red
        vis_img[goal_map == 1] = [0, 0, 255]

        # green
        vis_img[new_goal_map == 1] = [0, 255, 0]

        # blue
        vis_img[pose_xy[0], pose_xy[1]] = [255, 0, 0]

        vis_img = np.flipud(vis_img)

        # logger.debug(f"SAVING 2.interpolate_goal")
        cv2.imwrite(
            os.path.join(self.vis_dir, f"{timestep}_2.interpolate_goal.png"),
            vis_img,
        )

    def change_goal_map_to_closest_traversible_from_past_pose(self, goal_map, goal_pose, planning_window, timestep):
        logger.info(f"Changing goal map to closest traversible from past pose if needed.")
        if goal_pose is None:
            logger.info(f"No goal pose provided, returning original goal_map.")
            return goal_map

        #! myTODO: Commneted this because sometimes, goal point are considered traversible (they are not obstacles, because they have a height more than the robot)
        #! However, you can not easily go to them, and agent starts trying to find a path to the other side of a wall or sth similar.  
        # If there is any goal cell that is also traversible, return goal_map itself
        traversible_goal = np.logical_and(goal_map == 1, self.traversible == 1)
        if np.any(traversible_goal):
            logger.info(f"Goal map already has traversible cells, but doing nothing.")
            # return goal_map

        #! The goal pose is the actual index in global map
        pose_xy = np.array([
            int(goal_pose[1] - planning_window[0] + 1),
            int(goal_pose[2] - planning_window[2])+1]
        )
        logger.info(f"Changing goal map to closest traversible from past pose. Global goal pose is : {goal_pose}, local pose is : {pose_xy}")

        # Find closest goal_map cell to goal_pose
        goal_indices = np.argwhere(goal_map == 1)
        if goal_indices.size == 0:
            raise Exception("No goal cells found in goal_map, should not happen.")

        # Find closest goal cell in goal_map to the goal_pose
        dists = np.linalg.norm(goal_indices - pose_xy[None, :], axis=1)
        closest_goal_idx = goal_indices[np.argmin(dists)]
        gx, gy = closest_goal_idx
        logger.info(f"Closest goal index in goal_map to goal_pose is ({gx}, {gy})")

        # Generate Bresenham line from closest_goal_idx to goal_pose  
        line_coords = list(bresenham(int(gx), int(gy),pose_xy[0], pose_xy[1]))

        # Walk from closest goal_map cell to goal_pose and find first traversible point
        for x, y in line_coords:
            if self.traversible[x, y] == 1:
                new_goal_map = np.zeros_like(goal_map)
                new_goal_map[x, y] = 1
                logger.info(f"Setting traversible goal to {x, y}")
                self.visualize_converting_goal_to_pose(goal_map, self.traversible, pose_xy, new_goal_map, timestep)
                return new_goal_map
        logger.info(f"No traversible point found from closest goal, returning original goal_map.")

        return goal_map
 

    @staticmethod
    def get_mask(sx, sy, scale, step_size, min_radius=None):
        """Set everything in a circle around the agent to 1; else set to zero"""
        if min_radius is None:
            min_radius = (step_size - 1) ** 2
        size = int(step_size // scale) * 2 + 1
        mask = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                cond1 = (
                    ((i + 0.5) - (size // 2 + sx)) ** 2
                    + ((j + 0.5) - (size // 2 + sy)) ** 2
                ) <= step_size**2
                cond2 = (
                    ((i + 0.5) - (size // 2 + sx)) ** 2
                    + ((j + 0.5) - (size // 2 + sy)) ** 2
                ) > min_radius
                if cond1 and cond2:
                    mask[i, j] = 1
        mask[size // 2, size // 2] = 1
        return mask

    @staticmethod
    def get_dist(sx, sy, scale, step_size):
        size = int(step_size // scale) * 2 + 1
        mask = np.zeros((size, size)) + 1e-10
        for i in range(size):
            for j in range(size):
                if (
                    ((i + 0.5) - (size // 2 + sx)) ** 2
                    + ((j + 0.5) - (size // 2 + sy)) ** 2
                ) <= step_size**2:
                    mask[i, j] = max(
                        5,
                        (
                            ((i + 0.5) - (size // 2 + sx)) ** 2
                            + ((j + 0.5) - (size // 2 + sy)) ** 2
                        )
                        ** 0.5,
                    )
        return mask

    def _find_within_distance_to_multi_goal(
        self,
        goal: np.ndarray,
        distance: float,
        min_distance_only=False,
        timestep=0,
    ) -> np.ndarray:
        """
        Find the nearest point to a goal which is traversible
        """
        logger.info(f"Dilating goal map")

        
        #! myTODO: Finish fixing goal dilation
        planner = FMMPlanner(
            self.traversible if self.geodesic_dilation else np.ones_like(self.traversible),
            print_images=self.print_images,
            vis_dir=self.vis_dir,
            vis_postfix=self.vis_postfix
        )
        # Plan to the goal mask
        planner.set_multi_goal(goal, timestep=timestep, number="3")

        # Now mask out anything here based on distance to the goal mask
        mask = self.traversible
        dist_map = planner.fmm_dist * mask
        dist_map[dist_map == 0] = dist_map.max()

        if min_distance_only:
            min_dist_idx = dist_map.argmin()
            goal_pt = np.unravel_index(min_dist_idx, dist_map.shape)
            navigable_goal_map = np.zeros_like(goal)
            navigable_goal_map[goal_pt[0], goal_pt[1]] = 1
        else:
            navigable_goal_map = dist_map < distance

        if self.print_images:
            _navigable_goal_map = navigable_goal_map.copy()
            _navigable_goal_map = _navigable_goal_map.astype(np.uint8)
            _traversible = self.traversible.astype(np.uint8)

            white = np.ones((_navigable_goal_map.shape + (3,)), dtype=np.uint8) * 255
            white[_traversible == 0] = [0, 0, 0]
            white[_navigable_goal_map == 1] = [255, 0, 255] # purple
            white[goal == 1] = [0, 0, 255] # Initial goal in red
            white = np.flipud(white)
            # logger.debug(f"SAVING 4.dilate")
            cv2.imwrite(
                os.path.join(self.vis_dir, f"{timestep}_4.dilate_goal{self.vis_postfix}.png"),
                white,
            )
        return navigable_goal_map
