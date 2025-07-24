# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


def show_image(rgb):
    """Simple helper function to show images"""
    plt.figure()
    plt.imshow(rgb)
    plt.show()


def show_image_with_mask(rgb, mask):
    """tool for showing a mask and some other stuff"""
    plt.figure()
    plt.subplot(131)
    plt.imshow(rgb)
    plt.subplot(132)
    plt.imshow(mask)
    plt.subplot(133)
    _mask = mask[:, :, None]
    _mask = np.repeat(_mask, 3, axis=-1)
    plt.imshow(_mask * rgb / 255.0)
    plt.show()


def get_contour_points(
    pos: Tuple[float, float, float],
    origin: Tuple[float, float],
    size: int = 20,
) -> np.ndarray:
    x, y, o = pos
    pt1 = (int(x) + origin[0], int(y) + origin[1])
    pt2 = (
        int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1],
    )
    pt3 = (int(x + size * np.cos(o)) + origin[0], int(y + size * np.sin(o)) + origin[1])
    pt4 = (
        int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1],
    )

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(
    start: Tuple[int, int],
    end: Tuple[int, int],
    mat: np.ndarray,
    steps: int = 25,
    w: int = 1,
) -> np.ndarray:
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w : x + w, y - w : y + w] = 1
    return mat


def visualize_map(input_shape, dir, name, features=None, points=None, traversible=None, goal_map=None, dilated_goal_map=None, frontier_map=None):
    shape = input_shape + (3,)
    white = np.ones(shape, dtype=np.uint8) * 255

    if not traversible is None:
        white[traversible == 0] = [0, 0, 0] # black

    if not dilated_goal_map is None:
        white[dilated_goal_map == 1] = [255, 0, 255] # magenta
    
    if not goal_map is None:
        white[goal_map == 1] = [0, 0, 255] # red

    if not frontier_map is None:
        white[frontier_map == 1] = [255, 255, 0] # cyan

    if not features is None:
        for feature_map, color in features:
            if not feature_map is None:
                white[feature_map == 1] = color
    
    if not points is None:
        for point, color in points:
            white[point[0], point[1]] = color

    white = np.flipud(white)
    # logger.debug(f"SAVING 4.dilate")
    cv2.imwrite(
        os.path.join(dir, name),
        white,
    )


def visualize_frontier_scores_matplotlib(
    dir,
    traversible,
    frontier_map,
    frontier_centers,
    frontier_scores,
    top_k_semantic_classes,
    robot_loc=None,
    top_k=3,
    save_path="frontiers_with_scores.png"
):
    fig, ax = plt.subplots(figsize=(8, 8))
    img = np.ones(traversible.shape + (3,), dtype=np.uint8) * 255
    img[traversible == 0] = [0, 0, 0]  # obstacles as black
    img[frontier_map == 1] = [0, 255, 255]  # frontiers as cyan

    img = np.flipud(img)

    ax.imshow(img)

    # Sort and select top-k frontiers by score
    top_indices = np.argsort(frontier_scores)[-top_k:]

    for i in top_indices:
        center = frontier_centers[i]
        flipped_center = (center[1], traversible.shape[0] - center[0])

        ax.plot(flipped_center[0], flipped_center[1], "o", color="red", markersize=5)

        # annotation text
        score_text = f"{frontier_scores[i]:.2f}"
        sem_classes = top_k_semantic_classes[i]
        sem_text = ", ".join(f"#{cls}" for cls in sem_classes)

        full_text = f"{score_text}\n{sem_text}"

        # annotation with offset
        ax.annotate(
            full_text,
            xy=flipped_center,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=5,
            color="black",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", lw=1),
        )

    if robot_loc is not None:
        flipped_robot = (robot_loc[1], traversible.shape[0] - robot_loc[0])
        ax.plot(flipped_robot[0], flipped_robot[1], "x", color="blue", markersize=8, label="Robot")

    ax.set_title("Top Frontier Scores")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(dir,save_path), dpi=300)
    plt.close()

