# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import shutil
from collections import defaultdict
from typing import List, Optional, Tuple, Dict, Union

import cv2
import numpy as np
import skimage.morphology
from PIL import Image
from habitat.utils.visualizations import maps

import home_robot.utils.pose as pu
import home_robot.utils.visualization as vu
from habitat.utils.visualizations.utils import draw_collision
from habitat.utils.render_wrapper import append_text_to_image
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory
from home_robot.perception.constants import LanguageNavCategories
from home_robot.perception.constants import PaletteIndices as PI
from home_robot.perception.constants import RearrangeDETICCategories

from home_robot.utils.logger import get_logger

logger = get_logger()

rgb2bgr = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


class VIS_LAYOUT:
    HEIGHT = 480
    FIRST_PERSON_W = 360
    TOP_DOWN_W = HEIGHT
    LEFT_PADDING = 40
    MIDDLE_PADDING = 15
    TOP_PADDING = 50
    LEGEND_TOP_PADDING = 5
    BOTTOM_PADDING = 120

    Y1 = TOP_PADDING
    Y2 = TOP_PADDING + HEIGHT
    RGB_X1 = LEFT_PADDING
    RGB_X2 = LEFT_PADDING + FIRST_PERSON_W
    SEM_X1 = MIDDLE_PADDING + RGB_X2
    SEM_X2 = SEM_X1 + FIRST_PERSON_W

    TOP_DOWN_Y1 = Y1
    TOP_DOWN_Y2 = Y2
    TOP_DOWN_X1 = SEM_X2 + MIDDLE_PADDING
    TOP_DOWN_X2 = TOP_DOWN_X1 + TOP_DOWN_W
    ORACLE_TOP_DOWN_X1 = TOP_DOWN_X2 + MIDDLE_PADDING
    ORACLE_TOP_DOWN_X2 = ORACLE_TOP_DOWN_X1 + TOP_DOWN_W

    IMAGE_HEIGHT = Y2 + BOTTOM_PADDING
    IMAGE_WIDTH = ORACLE_TOP_DOWN_X2 + LEFT_PADDING


class VIS_LAYOUT_IMAGENAV:
    HEIGHT = 480
    FIRST_PERSON_W = 360
    TOP_DOWN_W = HEIGHT
    LEFT_PADDING = 40
    MIDDLE_PADDING = 15
    TOP_PADDING = 50
    LEGEND_TOP_PADDING = 5
    BOTTOM_PADDING = 120

    Y1 = TOP_PADDING
    Y2 = TOP_PADDING + HEIGHT
    RGB_X1 = LEFT_PADDING
    RGB_X2 = LEFT_PADDING + FIRST_PERSON_W
    SEM_X1 = MIDDLE_PADDING + RGB_X2
    SEM_X2 = SEM_X1 + FIRST_PERSON_W
    GOAL_X1 = MIDDLE_PADDING + SEM_X2
    GOAL_X2 = GOAL_X1 + FIRST_PERSON_W

    TOP_DOWN_Y1 = Y2 + MIDDLE_PADDING
    TOP_DOWN_Y2 = TOP_DOWN_Y1 + HEIGHT
    TOP_DOWN_X1 = LEFT_PADDING + MIDDLE_PADDING
    TOP_DOWN_X2 = TOP_DOWN_X1 + TOP_DOWN_W
    ORACLE_TOP_DOWN_X1 = TOP_DOWN_X2 + MIDDLE_PADDING
    ORACLE_TOP_DOWN_X2 = ORACLE_TOP_DOWN_X1 + TOP_DOWN_W

    IMAGE_HEIGHT = TOP_DOWN_Y2 + BOTTOM_PADDING
    IMAGE_WIDTH = GOAL_X2 + LEFT_PADDING


V = VIS_LAYOUT


class Visualizer:
    """
    This class is intended to visualize a single object goal navigation task.
    """

    def __init__(self, config, semantic_category_mapping, dataset=None):
        self.semantic_category_mapping = semantic_category_mapping
        self.show_images = config.VISUALIZE
        self.print_images = config.PRINT_IMAGES
        self.default_vis_dir = f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}"
        self._dataset = dataset
        os.makedirs(self.default_vis_dir, exist_ok=True)
        self.episodes_data_path = config.habitat.dataset.data_path

        self.num_sem_categories = semantic_category_mapping.num_sem_categories + 1
        self.map_resolution = config.AGENT.SEMANTIC_MAP.map_resolution
        map_size_cm = config.AGENT.SEMANTIC_MAP.map_size_cm
        self.map_shape = (
            map_size_cm // self.map_resolution,
            map_size_cm // self.map_resolution,
        )

        self.vis_dir = None
        self.image_vis = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.text_color = (20, 20, 20)  # BGR
        self.text_thickness = 1
        self.show_rl_obs = config.SHOW_RL_OBS
        self.ind_frame_height = 480

        self.num_agents = config.NUM_AGENTS

    def reset(self):
        self.vis_dir = self.default_vis_dir
        self.image_vis = None

    def set_vis_dir(self, dir_name:str):
        self.vis_dir = os.path.join(self.default_vis_dir, dir_name)
        shutil.rmtree(self.vis_dir, ignore_errors=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        if self.num_agents > 1:
            for i in range(self.num_agents):
                agent_dir = os.path.join(self.vis_dir, f"agent_{i}")
                os.makedirs(agent_dir, exist_ok=True)


    def _add_border(self, frame: np.ndarray, border_size: int) -> np.ndarray:
        """Add a white border to a frame."""
        h, w = frame.shape[:2]
        side = np.ones((h, border_size, 3), dtype=np.uint8) * 255
        frame = np.concatenate([side, frame, side], axis=1)
        top = np.ones((border_size, w + 2 * border_size, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top, frame, top], axis=0)
        return frame

    def color_semantic_frame(self, sem_img):
        semantic_map_vis = Image.new("P", (sem_img.shape[1], sem_img.shape[0]))
        semantic_map_vis.putpalette(self.semantic_category_mapping.map_color_palette)
        semantic_map_vis.putdata(sem_img.flatten().astype(np.uint8))
        semantic_map_vis = semantic_map_vis.convert("RGB")

        semantic_map_vis = rgb2bgr(np.asarray(semantic_map_vis))
        return semantic_map_vis

    def update_semantic_map_with_instances(self, semantic_map, instance_map):
        """
        Update the semantic mapping with instance ids.

        Draws borders around instances in the semantic map.

        Args:
            semantic_map: np.ndarray of shape [H, W] with semantic categories.
            instance_map: np.ndarray of shape [num_sem_categories - 2, H, W] where each channel has instances labeled as 1, 2, ...
        """
        for instance_channel in instance_map:
            if np.sum(instance_channel) == 0:
                continue
            instance_channel = (instance_channel > 0).astype(np.uint8)
            # get the border pixels
            border_pixels = np.logical_and(
                cv2.dilate(instance_channel, self.instance_dilation_selem),
                np.logical_not(instance_channel),
            )
            # update semantic map with instance ids
            semantic_map[border_pixels > 0] = PI.INSTANCE_BORDER

    def get_td_map(self, top_down_map: np.ndarray) -> np.ndarray:
        td_map = maps.colorize_draw_agent_and_fit_to_height(
            top_down_map, output_height=top_down_map["map"].shape[0]
        )
        return self.prepare_for_vis(rgb2bgr(td_map), "TD Map", (V.TOP_DOWN_W, V.HEIGHT))

    def visualize(
        self,
        timestep: int,
        semantic_frame: np.ndarray,
        rgb_frame: np.ndarray,
        caption: str,
        obstacle_map: np.ndarray,
        global_pose: np.ndarray,
        lmb: np.ndarray,
        explored_map: np.ndarray,
        semantic_map_1D: Tuple[np.ndarray, np.ndarray],
        been_close_map: np.ndarray,
        top_down_map,
        agent_id: Optional[int],
        visited_map,
        is_collision,
        task_type = None,
        short_term_goal: np.ndarray = None,
        closest_goal_pt: np.ndarray = None,
        dilated_obstacle_map: np.ndarray = None,
        instances_map:np.ndarray = None,
        inst_goal_found: bool = None,
        goal_instance_map: np.ndarray = None,
        goal_image: np.ndarray = None,
        is_local=True,
        metrics = None, #TODO
        blacklisted_targets_map: np.ndarray = None,
        frontier_map: np.ndarray = None,
        depth_frame: np.ndarray = None,
        **kwargs,
    ):
        """Visualize frame input and semantic map."""
        global V

        if task_type == "imagenav":
            V = VIS_LAYOUT_IMAGENAV
        else:
            V = VIS_LAYOUT

        if not self.show_images and not self.print_images:
            return

        main_frame = self.init_frame(caption)

        if dilated_obstacle_map is not None:
            obstacle_map = dilated_obstacle_map

        self.instance_dilation_selem = skimage.morphology.disk(1)

        semantic_map, no_category_mask = semantic_map_1D
        main_frame[V.TOP_DOWN_Y1 : V.TOP_DOWN_Y2, V.TOP_DOWN_X1 : V.TOP_DOWN_X2] = (
            self.make_sem_map(
                global_pose,
                lmb,
                obstacle_map,
                explored_map,
                semantic_map,
                closest_goal_pt,
                goal_instance_map,
                no_category_mask,
                visited_map,
                instances_map,
                been_close_map,
                short_term_goal,
                inst_goal_found,
                is_local,
            )
        )

        if task_type == "imagenav":
            main_frame[V.Y1 : V.Y2, V.GOAL_X1 : V.GOAL_X2] = self.prepare_for_vis(
                goal_image, "Goal", (V.FIRST_PERSON_W, V.HEIGHT)
            )

        if top_down_map:
            main_frame[
                V.TOP_DOWN_Y1 : V.TOP_DOWN_Y2,
                V.ORACLE_TOP_DOWN_X1 : V.ORACLE_TOP_DOWN_X2,
            ] = self.get_td_map(top_down_map)
        else:
            if not depth_frame is None:
                depth_frame[depth_frame > 5.0] = 0.0
                main_frame[
                    V.TOP_DOWN_Y1 : V.TOP_DOWN_Y2,
                    V.ORACLE_TOP_DOWN_X1 : V.ORACLE_TOP_DOWN_X1 + V.FIRST_PERSON_W,
                ] = self.prepare_for_vis(depth_frame / depth_frame.max() * 255.0, "Depth", (V.FIRST_PERSON_W, V.HEIGHT))

        main_frame[V.Y1 : V.Y2, V.RGB_X1 : V.RGB_X2] = self.prepare_for_vis(
            rgb_frame,
            "Observation",
            (V.FIRST_PERSON_W, V.HEIGHT),
            inst_goal_found,
            is_collision
        )

        sem_frame = self.color_semantic_frame(semantic_frame + PI.SEM_START)
        # sem_frame = semantic_frame
        main_frame[V.Y1 : V.Y2, V.SEM_X1 : V.SEM_X2] = self.prepare_for_vis(
            sem_frame,
            "Semantics",
            (V.FIRST_PERSON_W, V.HEIGHT),
            inst_goal_found,
            is_collision
        )

        # if instance_memory is not None:
        #     image_vis = self._visualize_instance_counts(image_vis, instance_memory)

        if self.show_images:
            cv2.imshow("Visualization", main_frame)
            cv2.waitKey(1)

        if self.print_images:
            if agent_id is None:
                path = os.path.join(self.vis_dir, f"{timestep}_13.snapshot.png")
            else:
                path = os.path.join(self.vis_dir, f"agent_{agent_id}", f"{timestep}_13.snapshot.png")
            success = cv2.imwrite(path, main_frame)

    def _visualize_instance_counts(
        self, image_vis: np.ndarray, instance_memory: InstanceMemory
    ):
        """
        Add instance counts to the panel

        Args:
            instance_memory (InstanceMemory): memory of all instances and views seen so far
            image_vis (np.ndarray): The image panel before adding instances

        Returns:
            image_vis (np.ndarray): The image panel after adding instances
        '"""
        num_instances_per_category = defaultdict(int)
        num_views_per_instance = defaultdict(list)
        for instance_id, instance in instance_memory.instances[0].items():
            num_instances_per_category[instance.category_id] += 1
            num_views_per_instance[instance.category_id].append(
                len(instance.instance_views)
            )
        text = "Instance counts"
        offset = 48
        y_pos = offset

        for index, count in num_instances_per_category.items():
            if count > 0:
                text = f"cat {index}: {num_views_per_instance[index]} views"
                image_vis = self._put_text_on_image(
                    image_vis,
                    text,
                    V.THIRD_PERSON_W,
                    y_pos,
                    V.THIRD_PERSON_W,
                    V.TOP_PADDING,
                )
                y_pos += offset
        return image_vis

    def _put_text_on_image(
        self,
        vis_image,
        text: str,
        bbox_x_start: int,
        bbox_y_start: int,
        bbox_x_len: int,
        bbox_y_len: int,
        font_scale: int = None,
    ):
        """
        Place text at the center of the given bounding box.
        """
        if font_scale is None:
            font_scale = self.font_scale

        textsize = cv2.getTextSize(text, self.font, font_scale, self.text_thickness)[0]
        # The x coordinate at which the left edge of text needs to be placed
        textX = (bbox_x_len - textsize[0]) // 2 + bbox_x_start
        # The height at which base needs to be placed
        textY = (bbox_y_len + textsize[1]) // 2 + bbox_y_start
        return cv2.putText(
            vis_image,
            text,
            (textX, textY),
            self.font,
            font_scale,
            self.text_color,
            self.text_thickness,
            cv2.LINE_AA,
        )

    def init_frame(self, caption):
        width = V.IMAGE_WIDTH

        main_frame = np.ones((V.IMAGE_HEIGHT, width, 3)).astype(np.uint8) * 255

        # TODO
        # Draw outlines
        # color = (100, 100, 100)
        # for y in [V.Y1 - 1, V.Y2]:
        #     for x_start, x_len in [
        #         (V.RGB_X1, V.FIRST_PERSON_W),
        #         (V.SEM_X1, V.FIRST_PERSON_W),
        #         (V.TOP_DOWN_X1, V.TOP_DOWN_W),
        #     ]:
        #         main_frame[y, x_start - 1 : x_start + x_len] = color

        # for x in [
        #     V.RGB_X1 - 1,
        #     V.RGB_X2,
        #     V.SEM_X1 - 1,
        #     V.SEM_X2,
        #     V.TOP_DOWN_X1 - 1,
        #     V.TOP_DOWN_X2,
        # ]:
        #     main_frame[V.Y1 - 1 : V.Y2, x] = color

        # # Draw legend
        # if os.path.exists(self.semantic_category_mapping.categories_legend_path):
        #     legend = cv2.imread(self.semantic_category_mapping.categories_legend_path)
        #     lx, ly, _ = legend.shape
        #     vis_image[
        #         V.Y2 + V.LEGEND_TOP_PADDING : V.Y2 + lx + V.LEGEND_TOP_PADDING, 0:ly, :
        #     ] = legend

        main_frame = self._put_text_on_image(
            main_frame,
            caption,
            0,
            V.TOP_DOWN_Y2 + V.LEGEND_TOP_PADDING,
            V.IMAGE_WIDTH,
            V.TOP_PADDING,
        )
        return main_frame

    def make_sem_map(
        self,
        global_pose: np.ndarray,
        lmb: np.ndarray,
        obstacle_map: np.ndarray,
        explored_map: np.ndarray,
        semantic_map: np.ndarray,
        closest_goal_pt: np.ndarray,
        goal_instance_map: np.ndarray,
        no_category_mask: np.ndarray,
        visited_map: np.ndarray,
        instances_map: np.ndarray,
        been_close_map: np.ndarray,
        short_term_goal,
        inst_goal_found=False,
        is_local=True,
    ) -> np.ndarray:
        if obstacle_map is None:
            return None
        curr_x, curr_y, curr_o = global_pose.cpu().float().numpy()
        if is_local:
            gy1, gy2, gx1, gx2 = lmb
            gy1, gy2, gx1, gx2 = int(gy1), int(gy2), int(gx1), int(gx2)
        else:
            gy1, gy2, gx1, gx2 = 0, obstacle_map.shape[0], 0, obstacle_map.shape[1]

        semantic_map += PI.SEM_START

        # Obstacles, explored, and visited areas
        semantic_map[no_category_mask] = PI.EMPTY_SPACE
        semantic_map[np.logical_and(no_category_mask, explored_map == 1)] = PI.EXPLORED
        semantic_map[np.logical_and(no_category_mask, obstacle_map == 1)] = PI.OBSTACLES
        semantic_map[visited_map == 1] = PI.VISITED

        # Goal
        if inst_goal_found:
            selem = skimage.morphology.disk(4)
            semantic_map[goal_instance_map] = PI.REST_OF_GOAL
            if closest_goal_pt is not None:
                closest_goal_map = np.zeros_like(goal_instance_map)
                closest_goal_map[closest_goal_pt[0], closest_goal_pt[1]] = 1
                closest_goal_mat = (
                    1 - skimage.morphology.binary_dilation(closest_goal_map, selem) != 1
                )
                closest_goal_mask = closest_goal_mat == 1
                semantic_map[closest_goal_mask] = PI.CLOSEST_GOAL

            if short_term_goal is not None:
                short_term_goal_mask = np.zeros(goal_instance_map.shape)
                short_term_goal_mask[short_term_goal[0], short_term_goal[1]] = 1
                short_term_goal_mask = (
                    1 - skimage.morphology.binary_dilation(short_term_goal_mask, selem)
                    != 1
                )
                short_term_goal_mask = short_term_goal_mask == 1
                semantic_map[short_term_goal_mask] = PI.SHORT_TERM_GOAL

        if instances_map is not None:
            self.update_semantic_map_with_instances(semantic_map, instances_map)

        # Semantic categories
        semantic_map_vis = self.color_semantic_frame(semantic_map)
        semantic_map_vis = np.flipud(semantic_map_vis)
        semantic_map_vis = np.ascontiguousarray(semantic_map_vis)


        # overlay the regions the agent has been close to
        been_close_map = np.flipud(been_close_map == 1)
        color_index = PI.BEEN_CLOSE * 3
        color = self.semantic_category_mapping.map_color_palette[
            color_index : color_index + 3
        ][::-1]
        semantic_map_vis[been_close_map] = (
            semantic_map_vis[been_close_map] + color
        ) / 2

        # overlay blacklisted targets
        # blacklisted_targets_map = np.flipud(np.rint(blacklisted_targets_map) == 1)
        # color_index = PI.BLACKLISTED_TARGETS_MAP * 3
        # color = self.semantic_category_mapping.map_color_palette[
        #     color_index : color_index + 3
        # ][::-1]
        # semantic_map_vis[blacklisted_targets_map] = (
        #     semantic_map_vis[blacklisted_targets_map] + color
        # ) / 2

        # Agent arrow
        pos = (
            (curr_x * 100.0 / self.map_resolution - gx1) * 480 / obstacle_map.shape[0],
            (obstacle_map.shape[1] - curr_y * 100.0 / self.map_resolution + gy1)
            * 480
            / obstacle_map.shape[1],
            np.deg2rad(-curr_o),
        )
        # pos = (
        #     pos[0] * V.TOP_DOWN_W / semantic_map.shape[1],
        #     pos[1] * V.HEIGHT / semantic_map.shape[0],
        #     pos[2],
        # )
        # agent_arrow = vu.get_contour_points(pos, origin=(V.TOP_DOWN_X1, V.Y1))
        agent_arrow = vu.get_contour_points(pos, origin=(0, 0))
        color = self.semantic_category_mapping.map_color_palette[9:12][::-1]
        cv2.drawContours(semantic_map_vis, [agent_arrow], 0, color, -1)
        semantic_map_vis = self.prepare_for_vis(semantic_map_vis, "Map", (V.TOP_DOWN_W, V.HEIGHT))
        return semantic_map_vis

    def _found_goal_detection(self, view: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """overlay a green goal detected banner"""
        strip_width = view.shape[0] // 15
        mask = np.ones(view.shape)
        mask[strip_width:-strip_width] = 0
        mask = mask == 1
        view[mask] = (alpha * np.array([0, 255, 0]) + (1.0 - alpha) * view)[mask]
        return append_text_to_image(view, ["Goal Detected"], font_size=0.5)

    def prepare_for_vis(
        self, frame, text, shape, set_found_goal=False, set_collision=False
    ):
        border_size = 0
        text_bar_height = 50 - border_size
        new_h = self.ind_frame_height - text_bar_height - 2 * border_size
        new_w = int(new_h / frame.shape[0] * frame.shape[1])
        frame = cv2.resize(frame, (new_w, new_h))

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if set_found_goal:
            frame = self._found_goal_detection(frame)

        # frame = self._write_metrics(frame, metrics)

        if set_collision:
            frame = draw_collision(frame)

        frame = self._add_border(frame, border_size)

        top_bar = np.ones((text_bar_height, frame.shape[1], 3), dtype=np.uint8) * 255
        frame = np.concatenate([top_bar, frame.astype(np.uint8)], axis=0)

        textsize = cv2.getTextSize(
            text, self.font, self.font_scale, self.text_thickness
        )[0]
        textX = (frame.shape[1] - textsize[0]) // 2
        textY = (text_bar_height + border_size + textsize[1]) // 2
        frame = cv2.putText(
            frame,
            text,
            (textX, textY),
            self.font,
            self.font_scale,
            self.text_color,
            self.text_thickness,
            cv2.LINE_AA,
        )
        return cv2.resize(frame, shape)
