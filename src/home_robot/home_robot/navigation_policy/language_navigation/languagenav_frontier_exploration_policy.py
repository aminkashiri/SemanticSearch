# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import scipy
import skimage.morphology
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN

from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.utils.morphology import binary_dilation
from scipy.ndimage import label

from home_robot.utils.logger import get_logger
logger = get_logger()


class LanguageNavSemanticFrontierExplorationPolicy(nn.Module):
    """
    Policy to select high-level goals for Object Goal Navigation:
    go to object goal if it is mapped and explore frontier (closest
    unexplored region) otherwise.
    """

    def __init__(
        self,
        exploration_strategy: str,
        num_of_semantic_categories: int,
        warmup_steps: int,
    ):
        super().__init__()
        assert exploration_strategy in [
            "seen_frontier",
            "been_close_to_frontier",
            "semantic",
        ]
        self.exploration_strategy = exploration_strategy

        self.dilate_explored_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(10))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.select_border_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(1))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.num_of_semantic_categories = num_of_semantic_categories
        self.warmup_steps = warmup_steps

    @property
    def goal_update_steps(self):
        return 1

    def forward(
        self,
        map_features,
        step,
        robot_position,
        object_category=None,
        reject_visited_targets=False,
    ):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 9 + num_sem_categories, M, M)
            object_category: object goal category
        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
            goal category of shape (batch_size,)
        """
        assert object_category is not None

        goal_map, found_goal = self.reach_goal_if_in_map(
            map_features, object_category, reject_visited_targets=reject_visited_targets
        )

        if self.exploration_strategy == "semantic" and step > self.warmup_steps:
            goal_map = self.semantic_exploration(
                map_features, goal_map, found_goal, object_category, robot_position
            )
        else:
            goal_map = self.explore_otherwise(map_features, goal_map, found_goal)
        return goal_map, found_goal

    def semantic_exploration(
        self, map_features, goal_map, found_goal, category, robot_position
    ):
        if found_goal[0]:
            return goal_map

        sem_weights = np.random.rand(
            self.num_of_semantic_categories, self.num_of_semantic_categories
        )
        sem_weights = sem_weights[category]

        frontier_map = self.get_frontier_map(map_features)
        structure = np.ones((3, 3))  # 8-connectivity
        labeled_map, num_features = label(frontier_map, structure=structure)

        frontiers = [
            frontiers.append(np.argwhere(labeled_map == i))
            for i in range(1, num_features + 1)
        ]

        sem_layers = map_features[
            0,
            2 * MC.NON_SEM_CHANNELS : 2 * MC.NON_SEM_CHANNELS
            + self.num_of_semantic_categories,
            :,
            :,
        ]
        r = 40
        frontier_scores = []
        for frontier in frontiers:
            center = int(frontier.mean(axis=0))
            local_map = sem_layers[
                :, center[0] - r : center[0] + r, center[1] - r : center[1] + r
            ]
            neighbor_classes = np.where(local_map.any(axis=(1, 2)))[0]

            if len(neighbor_classes) > 0:
                frontier_sem_score = np.mean(
                    sem_weights[neighbor_classes]
                ) / np.linalg.norm(
                    robot_position, center
                )  #! MyTODO: Geodesic distance
            else:
                frontier_sem_score = np.mean(sem_weights) / np.linalg.norm(
                    robot_position, center
                )  #! MyTODO: Geodesic distance
            frontier_scores.append(frontier_sem_score)

        # Select the frontier with the highest score
        best_frontier = np.argmax(frontier_scores)
        goal_map = np.zeros(map_features.shape[-2:], dtype=np.uint8)
        goal_map[tuple(best_frontier.T)] = 1
        return goal_map

    def reach_goal_if_in_map(
        self,
        map_features,
        goal_category,
        reject_visited_targets=False,
    ):
        """If the desired goal is in the semantic map, reach it."""
        batch_size, _, height, width = map_features.shape
        device = map_features.device

        goal_map = torch.zeros((batch_size, height, width), device=device)
        found_goal_current = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for e in range(batch_size):
            # if the category goal was not found previously
            if not found_goal_current[e]:
                # the category to navigate to
                category_map = map_features[
                    e, goal_category[e] + 2 * MC.NON_SEM_CHANNELS, :, :
                ]

                if reject_visited_targets:
                    # remove the target objects that the agent has already been close to
                    category_map = category_map * (
                        1 - map_features[e, MC.BLACKLISTED_TARGETS_MAP, :, :]
                    )
                # if the desired category is found with required constraints, set goal for navigation
                if (category_map == 1).sum() > 0:
                    goal_map[e] = category_map == 1
                    found_goal_current[e] = True
        return goal_map, found_goal_current

    def get_frontier_map(self, map_features):
        # Select unexplored area
        if self.exploration_strategy == "seen_frontier":
            frontier_map = (map_features[:, [MC.EXPLORED_MAP], :, :] == 0).float()
        elif self.exploration_strategy == "been_close_to_frontier":
            frontier_map = (map_features[:, [MC.BEEN_CLOSE_MAP], :, :] == 0).float()

        # Dilate explored area
        frontier_map = 1 - binary_dilation(
            1 - frontier_map, self.dilate_explored_kernel
        )

        # Select the frontier
        frontier_map = (
            binary_dilation(frontier_map, self.select_border_kernel) - frontier_map
        )
        return frontier_map

    def explore_otherwise(self, map_features, goal_map, found_goal):
        """Explore closest unexplored region otherwise."""
        frontier_map = self.get_frontier_map(map_features)
        batch_size = map_features.shape[0]
        for e in range(batch_size):
            if not found_goal[e]:
                goal_map[e] = frontier_map[e]

        return goal_map


def visualize_maps(
    frontier_map, obstacle_map, known_map, robot_pos=None, title="Frontier Debug View"
):
    """
    Visualizes the frontier map alongside obstacles and explored area.

    Args:
        frontier_map (torch.Tensor or np.ndarray): binary map of frontiers (1 = frontier)
        obstacle_map (torch.Tensor or np.ndarray): binary map of obstacles (1 = obstacle)
        known_map (torch.Tensor or np.ndarray): binary map of known/explored cells (1 = known)
        robot_pos (tuple or None): (x, y) position to plot (optional)
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    frontier_map = frontier_map.squeeze().cpu().numpy()
    obstacle_map = obstacle_map.squeeze().cpu().numpy()
    known_map = known_map.squeeze().cpu().numpy()

    H, W = known_map.shape
    vis_map = np.ones((H, W, 3), dtype=np.uint8) * 255

    vis_map[obstacle_map == 1] = [0, 0, 0]
    vis_map[frontier_map == 1] = [255, 0, 0]

    vis_map[known_map == 1] = [117, 117, 117]

    plt.imshow(vis_map)
    plt.title(title)
    plt.axis("off")
    plt.show()


class LanguageNavFrontierExplorationPolicy(nn.Module):
    """
    Policy to select high-level goals for Object Goal Navigation:
    go to object goal if it is mapped and explore frontier (closest
    unexplored region) otherwise.
    """

    def __init__(self, exploration_strategy: str, close_frontier_radius: float = 20.0, goto_past_pose=False):
        super().__init__()
        assert exploration_strategy in [
            "seen_frontier",
            "been_close_to_frontier",
            "fixed",
        ]
        self.exploration_strategy = exploration_strategy

        self.dilate_explored_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(10))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.select_border_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(1))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )

        self.close_frontier_radius = close_frontier_radius

        self.goto_past_pose = goto_past_pose 

    @property
    def goal_update_steps(self):
        return 1

    def reach_single_category(
        self, map_features, category, reject_visited_targets, location=None, instance_memory=None, num_sem_categories=None
    ):
        # if the goal is found, reach it
        goal_map, found_goal, goal_pose = self.reach_goal_if_in_map(
            map_features, category, reject_visited_targets=reject_visited_targets, instance_memory=instance_memory, num_sem_categories=num_sem_categories, location=location
        )
        # otherwise, do frontier exploration
        goal_map = self.explore_otherwise(map_features, goal_map, found_goal, location)
        if found_goal[0]==True:
            logger.info(f"Found goal category {category[0]} in the map, returning it as goal.")
        else:
            logger.info(
                f"Did not find goal category {category[0]} in the map, exploring frontier instead. {found_goal[0]}"
            )

        return goal_map, found_goal, goal_pose

    def forward(
        self,
        map_features,
        object_category=None,
        reject_visited_targets=False,
        location=None,
        instance_memory=None,
        num_sem_categories=None,
    ):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 9 + num_sem_categories, M, M)
            object_category: object goal category
        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
            goal category of shape (batch_size,)
        """
        assert object_category is not None

        # Here, the goal is specified by a single object
        return self.reach_single_category(
            map_features, object_category, reject_visited_targets, location=location, instance_memory=instance_memory, num_sem_categories=num_sem_categories
        )

    def cluster_filtering(self, m):
        # m is a 480x480 goal map
        if not m.any():
            return m
        device = m.device

        # cluster goal points
        k = DBSCAN(eps=4, min_samples=1)
        m = m.cpu().numpy()
        data = np.array(m.nonzero()).T
        k.fit(data)

        # mask all points not in the largest cluster
        mode = scipy.stats.mode(k.labels_, keepdims=True).mode.item()
        mode_mask = (k.labels_ != mode).nonzero()
        x = data[mode_mask]

        m_filtered = np.copy(m)
        m_filtered[x] = 0.0
        m_filtered = torch.tensor(m_filtered, device=device)

        return m_filtered

    def reach_goal_if_in_map(
        self,
        map_features,
        goal_category,
        reject_visited_targets=False,
        instance_memory=None,
        num_sem_categories=None,
        location=None,
    ):
        """If the desired goal is in the semantic map, reach it."""
        logger.debug(f"Searching for goal category {goal_category[0]} in the map.")
        goal_pose = None

        batch_size, _, height, width = map_features.shape
        device = map_features.device

        goal_map = torch.zeros((batch_size, height, width), device=device)
        found_goal_current = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for e in range(batch_size):
            # if the category goal was not found previously
            if not found_goal_current[e]:
                # the category to navigate to
                category_map = map_features[
                    e, goal_category[e] + 2 * MC.NON_SEM_CHANNELS, :, :
                ]

                if reject_visited_targets:
                    # remove the target objects that the agent has already been close to
                    category_map = category_map * (
                        1 - map_features[e, MC.BLACKLISTED_TARGETS_MAP, :, :]
                    )
                # if the desired category is found with required constraints, set goal for navigation
                if (category_map == 1).sum() > 0:
                    logger.debug("Found goal category in the map.")

                    found_goal_current[e] = True

                    if self.goto_past_pose:
                        logger.debug("Returning past pose for the goal.")
                        goal_map[e], goal_pose = self.get_goal_map_for_category(goal_category[e], instance_memory, map_features, num_sem_categories, location)
                    else:
                        logger.debug("Returning goal cells as the goal map.")
                        goal_map[e] = category_map == 1
                else:
                    logger.debug("Did not find goal category in the map.")
        return goal_map, found_goal_current, goal_pose
    
    def get_goal_map_for_category(self, category, instance_memory, local_map, num_sem_categories, location, mode="closest_pose"):
        """
        Get the goal map for a specific category from the instance memory, by checking adding all the poses.
        location is (y,x) in the grid. local_map[y,x] is the value for the cell robot is currently at.
        """
        #! myTODO: Instead of this, we can return all poses and all possible goals, and later convert them back to best location for that pose, and then choose the closest.
        instance_map = local_map[0][
            MC.NON_SEM_CHANNELS
            + num_sem_categories : MC.NON_SEM_CHANNELS
            + 2 * num_sem_categories,
            :,
            :,
        ]

        best_view = None
        best_inst_key = None
        best_metric = 0
        for (inst_key, inst) in instance_memory.instance_views[0].items():
            if inst.category_id not in category:
                continue
            views = inst.instance_views
            max_coverage_view = np.argmax([view.object_coverage for view in views])
            if mode == "best_pose":
                if views[max_coverage_view].object_coverage > best_metric:
                    best_metric = views[max_coverage_view].object_coverage
                    best_view = views[max_coverage_view]
                    best_inst_key = inst_key
            elif mode == "closest_pose":
                if np.linalg.norm(views[max_coverage_view].pose[:2]- location.cpu()) < best_metric or best_metric == 0:
                    best_metric = np.linalg.norm(views[max_coverage_view].pose[:2]- location.cpu())
                    best_view = views[max_coverage_view]
                    best_inst_key = inst_key

        pose = best_view.pose
        curr_x, curr_y, curr_o, gy1, _, gx1, _ = pose.tolist()

        inst_map_idx = instance_map == best_inst_key
        inst_map_idx = torch.argmax(torch.sum(inst_map_idx, axis=(1, 2)))
        goal_map = (instance_map[inst_map_idx] == best_inst_key).to(torch.float)
        
        #! The output goal pose is the actual index in global map
        goal_pose = [[curr_o, curr_y * 100.0 / 5 , curr_x * 100.0 / 5]]

        logger.info(f">>> Goal instance {best_inst_key} best past pose is: {goal_pose}, with coverage {best_metric}. Returning goal_pose in addition to goal_map.")
        return goal_map, goal_pose

    def remove_close_frontiers(
        self, frontier_map: torch.Tensor, location: torch.Tensor
    ) -> torch.Tensor:
        """
        Remove frontiers closer than 'radius' to 'location' from the frontier map.

        Args:
            frontier_map (torch.Tensor): shape [1, 1, H, W] binary map of frontiers
            location (torch.Tensor): shape [2] -> (y, x) indices in grid
            radius (float): distance threshold (in pixels)

        Returns:
            torch.Tensor: updated frontier_map with close frontiers removed
        """
        assert frontier_map.dim() == 4, "Frontier map must be of shape [1,1,H,W]"
        _, _, H, W = frontier_map.shape
        device = frontier_map.device

        y_coords = torch.arange(H, device=device).unsqueeze(1).expand(H, W)
        x_coords = torch.arange(W, device=device).unsqueeze(0).expand(H, W)

        dist = torch.sqrt((x_coords - location[1]) ** 2 + (y_coords - location[0]) ** 2)
        close_mask = dist <= self.close_frontier_radius
        frontier_map = (
            frontier_map.clone()
        )
        frontier_map[0, 0][close_mask] = 0

        return frontier_map

    def get_frontier_map_fixed(self, map_features):
        """
        Detect frontiers: free cells adjacent to unknown areas.

        Args:
            known_map (np.ndarray): binary 2D map (1=known, 0=unknown)
            obstacle_map (np.ndarray): binary 2D map (1=obstacle, 0=non-obstacle)

        Returns:
            frontier_map (np.ndarray): binary map (1=frontier, 0=non-frontier)
        """

        def remove_small_frontiers(frontier_tensor, min_size=10):
            frontier_map = frontier_tensor[0, 0].cpu().numpy()
            labeled_map, num_features = label(frontier_map)
            cleaned_map = np.zeros_like(frontier_map)
            for region_id in range(1, num_features + 1):
                region = labeled_map == region_id
                if np.sum(region) >= min_size:
                    cleaned_map[region] = 1
            cleaned_map = (
                torch.tensor(cleaned_map, dtype=frontier_tensor.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(frontier_tensor.device)
            )
            return cleaned_map

        import torch.nn.functional as F

        known_map = (map_features[:, [MC.EXPLORED_MAP], :, :] != 0).float()
        obstacle_map = (map_features[:, [MC.OBSTACLE_MAP], :, :] != 0).float()
        assert known_map.shape[:2] == obstacle_map.shape[:2]
        assert known_map.shape[:2] == (1, 1)

        device = known_map.device
        free_space = (known_map == 1) & (obstacle_map == 0)

        unknown = (known_map == 0).to(torch.float32)

        kernel = (
            torch.tensor(
                [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32, device=device
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        unknown_neighbors = F.conv2d(unknown, kernel, padding=1)
        frontier_map = (free_space & (unknown_neighbors > 0)).float()
        frontier_map = remove_small_frontiers(frontier_map, min_size=15)
        # visualize_maps(
        #     frontier_map=frontier_map,
        #     obstacle_map=obstacle_map,
        #     known_map=known_map,
        #     # robot_pos=(50, 50),  # optional
        #     # title="Map with Frontiers"
        # )
        return frontier_map

    def get_frontier_map(self, map_features):
        # Select unexplored area
        if self.exploration_strategy == "seen_frontier":
            frontier_map = (map_features[:, [MC.EXPLORED_MAP], :, :] == 0).float()
        elif self.exploration_strategy == "been_close_to_frontier":
            frontier_map = (map_features[:, [MC.BEEN_CLOSE_MAP], :, :] == 0).float()
        elif self.exploration_strategy == "fixed":
            return self.get_frontier_map_fixed(map_features)

        # Dilate explored area
        frontier_map = 1 - binary_dilation(
            1 - frontier_map, self.dilate_explored_kernel
        )

        # Select the frontier
        frontier_map = (
            binary_dilation(frontier_map, self.select_border_kernel) - frontier_map
        )
        return frontier_map

    def explore_otherwise(self, map_features, goal_map, found_goal, location=None):
        """Explore closest unexplored region otherwise."""
        frontier_map = self.get_frontier_map(map_features)
        if location is not None:
            frontier_map = self.remove_close_frontiers(frontier_map, location)
        batch_size = map_features.shape[0]
        for e in range(batch_size):
            if not found_goal[e]:
                goal_map[e] = frontier_map[e]

        return goal_map
