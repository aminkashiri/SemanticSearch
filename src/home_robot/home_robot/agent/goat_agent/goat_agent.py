# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import scipy
import torch
from sklearn.cluster import DBSCAN
from torch.nn import DataParallel

import home_robot.utils.pose as pu

from home_robot.agent.imagenav_agent.visualizer import NavVisualizer
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory
from home_robot.perception.detection.maskrcnn.coco_categories import coco_categories

from .goat_agent_module import GoatAgentModule
from .goat_matching import GoatMatching

from home_robot.utils.logger import get_logger
import psutil

logger = get_logger()

from home_robot.navigation_policy.language_navigation.languagenav_frontier_exploration_policy import (
    LanguageNavFrontierExplorationPolicy,
)
from home_robot.navigation_planner.fixed_discrete_planner import (
    DiscretePlanner,
)

from home_robot.utils.visualization import visualize_map


class GoatAgent(Agent):
    """Simple object nav agent based on a 2D semantic map"""

    # Flag for debugging data flow and task configuraiton
    verbose = False

    def __init__(self, config, device_id: int = 0):
        # self.max_steps = config.AGENT.max_steps
        # self.max_steps = [500, 500, 500, 500, 500]
        self.max_steps = [500] * 10
        # self.max_steps = [500, 400, 300, 200, 200, 200, 200, 200, 200, 200, 200]
        # self.max_steps = [400, 300, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        self.num_environments = config.NUM_ENVIRONMENTS
        self.store_all_categories_in_map = getattr(
            config.AGENT, "store_all_categories", False
        )

        self.goal_matching_vis_dir = f"{config.DUMP_LOCATION}/goal_grounding_vis"
        Path(self.goal_matching_vis_dir).mkdir(parents=True, exist_ok=True)

        self.instance_memory = None
        self.record_instance_ids = getattr(
            config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
        )

        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                config.AGENT.SEMANTIC_MAP.du_scale,
                # debug_visualize=config.PRINT_IMAGES,
                config=config,
                mask_cropped_instances=False,
                padding_cropped_instances=200,
            )

        ## imagenav stuff
        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

        self.goal_policy_config = config.AGENT.SUPERGLUE

        # self.instance_seg = Detic(config.AGENT.DETIC)
        self.matching = GoatMatching(
            device=0,  # config.simulator_gpu_id
            score_func=self.goal_policy_config.score_function,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            config=config.AGENT.SUPERGLUE,
            default_vis_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
            print_images=config.PRINT_IMAGES,
            instance_memory=self.instance_memory,
        )

        self.image_matching_function = self.matching.match_image_to_image

        self._module = GoatAgentModule(
            config, matching=self.matching, instance_memory=self.instance_memory
        )

        if config.NO_GPU:
            self.device = torch.device("cpu")
            self.module = self._module
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")
            self._module = self._module.to(self.device)
            # Use DataParallel only as a wrapper to move model inputs to GPU
            self.module = DataParallel(self._module, device_ids=[self.device_id])

        self.visualize = config.VISUALIZE or config.PRINT_IMAGES
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            evaluate_instance_tracking=getattr(
                config.ENVIRONMENT, "evaluate_instance_tracking", False
            ),
            instance_memory=self.instance_memory,
        )
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        self.max_num_sub_task_episodes = config.ENVIRONMENT.max_num_sub_task_episodes


        if config.AGENT.panorama_start:
            panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            panorama_start_steps = 0

        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualize=config.VISUALIZE,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            agent_cell_radius=agent_cell_radius,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=config.AGENT.PLANNER.discrete_actions,
            min_goal_distance_cm=config.AGENT.PLANNER.min_goal_distance_cm,
            panorama_start_steps=panorama_start_steps,
        )
        self.one_hot_encoding = torch.eye(
            config.AGENT.SEMANTIC_MAP.num_sem_categories, device=self.device
        )

        self.sub_task_timesteps = None
        self.total_timesteps = None
        self.last_poses = None
        self.reject_visited_targets = False
        self.blacklist_target = False

        self.current_task_idx = 0

        self.imagenav_visualizer = NavVisualizer(
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
        )
        # self.imagenav_visualizer = None
        self.instance_map = None
        self.view_loc = None
        self.goal_filtering = config.AGENT.SEMANTIC_MAP.goal_filtering

        self.policy = LanguageNavFrontierExplorationPolicy(
            exploration_strategy=config.AGENT.exploration_strategy,
            goto_past_pose=config.AGENT.SUPERGLUE.goto_past_pose,
        )

    @torch.no_grad()
    def prepare_planner_inputs(
        self,
        obs: torch.Tensor,
        pose_delta: torch.Tensor,
        camera_pose: torch.Tensor = None,
        reject_visited_targets: bool = False,
        blacklist_target: bool = False,
        confidence=None,
        frame_matches_local_instance_ids=None,
        all_confidences=None,
        instance_ids=None,
        score_thresh=0.0,
        obstacle_locations: torch.Tensor = None,
        free_locations: torch.Tensor = None,
    ) -> Tuple[List[dict], List[dict]]:
        """Prepare low-level planner inputs from an observation - this is
                the main inference function of the agent that lets it interact with
                vectorized environments.
        s
                This function assumes that the agent has been initialized.

                Args:
                    obs: current frame containing (RGB, depth, segmentation) of shape
                     (3 + 1 + num_sem_categories, frame_height, frame_width)
                    pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
                     of shape (3)
                    object_goal_category: semantic category of small object goals
                    camera_pose: camera extrinsic pose of shape (4, 4)

                Returns:
                    planner_inputs: list of num_environments planner inputs dicts containing
                        obstacle_map: (M, M) binary np.ndarray local obstacle map
                         prediction
                        sensor_pose: (7,) np.ndarray denoting global pose (x, y, o)
                         and local map boundaries planning window (gx1, gx2, gy1, gy2)
                        goal_map: (M, M) binary np.ndarray denoting goal location
                    vis_inputs: list of num_environments visualization info dicts containing
                        explored_map: (M, M) binary np.ndarray local explored map
                         prediction
                        semantic_map: (M, M) np.ndarray containing local semantic map
                         predictions
        """
        # * before module call obs.shape is [380+3+1+num_instances]

        (
            self.instance_map,
            self.view_loc,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
        ) = self.module(
            obs,
            pose_delta,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
            camera_pose=camera_pose,
            reject_visited_targets=reject_visited_targets,
            blacklist_target=blacklist_target,
            confidence=confidence,
            frame_matches_local_instance_ids=frame_matches_local_instance_ids,
            all_confidences=all_confidences,
            instance_ids=instance_ids,
            score_thresh=score_thresh,
            obstacle_locations=obstacle_locations,
            free_locations=free_locations,
            # timestep=self.total_timesteps+1
        )

        if self._module.instance_goal_found:
            visualize_map(
                self.instance_map.shape,
                self.planner.vis_dir,
                f"{self.sub_task_timesteps[self.current_task_idx]+1}_01.instance_map.png",
                traversible=1 - self.semantic_map.get_obstacle_map(),
                goal_map=self.instance_map,
            )

        self.policy.vis_dir = self.planner.vis_dir
        frontier_map = self.policy.get_frontier_map(
            self.semantic_map.local_map,
            self.semantic_map.local_loc,
            self.total_timesteps + 1,
        )
        self.semantic_map.frontier_map = frontier_map.cpu().numpy()

        self._keep_only_largest_cluster_in_instance_map()

        self.total_timesteps = self.total_timesteps + 1
        self.sub_task_timesteps[self.current_task_idx] += 1

        planner_inputs = {
            "instance_goal_found": self._module.instance_goal_found,
            "instance_map": self.instance_map,
            "view_loc": self.view_loc,
            "obstacle_map": self.semantic_map.get_obstacle_map(),
            "frontier_map": self.semantic_map.frontier_map,
            "global_pose": self.semantic_map.global_pose,
            "lmb": self.semantic_map.lmb,
            "total_timesteps": self.total_timesteps,
            "timestep": self.sub_task_timesteps[self.current_task_idx],
        }
        if self.visualize:
            vis_inputs = {
                "explored_map": self.semantic_map.get_explored_map(),
                "semantic_map": self.semantic_map.get_semantic_map(),
                "been_close_map": self.semantic_map.get_been_close_map(),
                "timestep": self.total_timesteps,
            }
            if self.record_instance_ids:
                vis_inputs["instance_map"] = self.semantic_map.get_instance_map()
        else:
            vis_inputs = {}

        return planner_inputs, vis_inputs

    def reset_sub_episode(self) -> None:
        """Reset for a new sub-episode since pre-processing is temporally dependent."""
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self._module.reset_sub_episode()

    def reset(self):
        """Initialize agent state. Reset is at the beginning of a new episode (not each task)."""
        self.total_timesteps = 0
        self.sub_task_timesteps = [0] * self.max_num_sub_task_episodes
        self.last_poses = np.zeros(3)
        self.semantic_map.init_map_and_pose()
        if self.instance_memory is not None:
            self.instance_memory.reset()
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.current_task_idx = 0
        self.navigate_to_best = False

        if self.imagenav_visualizer is not None:
            self.imagenav_visualizer.reset()

        self.instance_map = None
        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None
        self.planner.reset()
        self._module.reset()

    def score_thresh(self, task_type):
        # If we have fully explored the environment, set the matching threshold to 0.0
        # to go to the highest scoring instance
        if self.navigate_to_best:
            return 0.0

        if task_type == "languagenav":
            return self.goal_policy_config.score_thresh_lang
        elif task_type == "imagenav":
            return self.goal_policy_config.score_thresh_image
        else:
            return 0.0

    def act(
        self, obs: Observations, stop=False
    ) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """Act end-to-end."""
        logger.info(
            f"---------------- Subtask step {self.sub_task_timesteps[self.current_task_idx]+1} ----------------"
        )
        logger.debug(f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")
        current_task = obs.task_observations["tasks"][self.current_task_idx]
        task_type = current_task["type"]

        # 1 - Obs preprocessing
        (
            obs_preprocessed,
            pose_delta,
            img_goal,
            camera_pose,
            confidence,
            frame_matches_local_instance_ids,
            all_confidences,
            instance_ids,
        ) = self._preprocess_obs(obs, task_type)

        # 2 - Semantic mapping + policy
        planner_inputs, vis_inputs = self.prepare_planner_inputs(
            obs_preprocessed,
            pose_delta,
            camera_pose=camera_pose,
            reject_visited_targets=self.reject_visited_targets,
            confidence=confidence,
            frame_matches_local_instance_ids=frame_matches_local_instance_ids,
            all_confidences=all_confidences,
            instance_ids=instance_ids,
            score_thresh=self.score_thresh(task_type),
        )

        # 3 - Planning
        (
            action,
            closest_goal_map,
            short_term_goal,
            dilated_obstacle_map,
            found_path,
            reset_module,
        ) = self.planner.plan(**planner_inputs)

        if reset_module:
            self.reset_sub_episode()

        if (
            self.sub_task_timesteps[self.current_task_idx]
            >= self.max_steps[self.current_task_idx]
        ) or stop:
            logger.warning(
                "Reached max number of steps for subgoal, or stuck somewhere, calling STOP"
            )
            action = DiscreteNavigationAction.STOP

        if not found_path and action != DiscreteNavigationAction.STOP:
            #! myTODO: This adds one step, I can fix this later
            if self.navigate_to_best:
                #! If we are fully explored and we are here again, we should just stop
                logger.info(
                    "Already fully explored and no path to our best match. Stopping"
                )
                action = DiscreteNavigationAction.STOP
            else:
                # * No need to rest module, we already have
                logger.info(
                    "Map fully explored. Setting the goal for next step to the best match in memory, even if it is less than threshold..."
                )
                self.navigate_to_best = True
                action = DiscreteNavigationAction.TURN_RIGHT

        if self.visualize:
            #! myTODO: If I want visualiztion for both goals, I need to modify this.
            vis_inputs["dilated_obstacle_map"] = dilated_obstacle_map
            if task_type == "imagenav":
                collision = {"is_collision": False}
                info = {
                    **planner_inputs,
                    **vis_inputs,
                    "rgb_frame": obs.rgb,
                    "semantic_frame": obs.semantic,
                    "closest_goal_map": closest_goal_map,
                    "last_goal_image": obs.task_observations["tasks"][
                        self.current_task_idx
                    ]["image"],
                    "last_collisions": collision,
                    "last_td_map": obs.task_observations.get("top_down_map"),
                    "short_term_goal": short_term_goal,
                }
                if self.imagenav_visualizer is not None:
                    info.pop("instance_goal_found", None)
                    self.imagenav_visualizer.visualize(**info)
            else:
                goal_text_desc = {
                    x: y
                    for x, y in obs.task_observations["tasks"][
                        self.current_task_idx
                    ].items()
                    if x != "image"
                }
                vis_inputs["goal_name"] = goal_text_desc
                vis_inputs["semantic_frame"] = obs.task_observations["semantic_frame"]
                vis_inputs["closest_goal_map"] = closest_goal_map
                vis_inputs["third_person_image"] = obs.third_person_image
                vis_inputs["short_term_goal"] = None
                vis_inputs["instance_memory"] = self.instance_memory

                info = {
                    **planner_inputs,
                    **vis_inputs,
                    "short_term_goal": short_term_goal,
                }
        else:
            info = None

        if action == DiscreteNavigationAction.STOP:
            if len(obs.task_observations["tasks"]) - 1 > self.current_task_idx:
                self.current_task_idx += 1
                self.total_timesteps = 0
                self.reset_sub_episode()
        return action, info

    def _preprocess_obs(self, obs: Observations, task_type: str):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map."""

        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        )  # m to cm

        current_task = obs.task_observations["tasks"][self.current_task_idx]

        semantic = obs.semantic
        instance_ids = None

        (
            confidences,
            frame_matches_local_instance_ids,
            all_confidences,
            instance_ids,
        ) = (None, None, [], [])


        if not self._module.instance_goal_found:
            if task_type == "imagenav":
                if self.goal_image is None:
                    img_goal = obs.task_observations["tasks"][self.current_task_idx][
                        "image"
                    ]
                    (
                        self.goal_image,
                        self.goal_image_keypoints,
                    ) = self.matching.get_goal_image_keypoints(img_goal)
                    # self.goal_mask, _ = self.instance_seg.get_goal_mask(img_goal)

                (
                    confidences,
                    frame_matches_local_instance_ids,
                ) = self.matching.get_matches_against_current_frame(
                    self.image_matching_function,
                    self.total_timesteps,
                    image_goal=self.goal_image,
                    goal_image_keypoints=self.goal_image_keypoints,
                    categories=[current_task["semantic_id"]],
                    use_full_image=False,
                )

            elif task_type == "languagenav":
                (
                    confidences,
                    frame_matches_local_instance_ids,
                ) = self.matching.get_matches_against_current_frame(
                    self.matching.match_language_to_image,
                    self.total_timesteps,
                    language_goal=current_task["description"],
                    categories=[current_task["semantic_id"]],
                    use_full_image=True,
                )
            elif task_type == "objectnav":
                (
                    confidences,
                    frame_matches_local_instance_ids,
                ) = self.matching.get_matches_against_current_frame(
                    None,
                    self.total_timesteps,
                    categories=[current_task["semantic_id"]],
                )

        # * Semantics becomes (W,H,NumClasses) which NumClasses is read from the config files, and is 380. Note that because I am using less classes (52 in all_ovon_categires) most of these layers are zero and actually useless.
        # * Maybe I should change the config. But nevertheles, this works even with 380.
        semantic = self.one_hot_encoding[torch.from_numpy(semantic).to(self.device)]

        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)

        if self.record_instance_ids:
            # * Why using instance_map which are the raw semantics? To differentiate between objects with diff raw semantics but same category in our ovon classes.
            instances = obs.task_observations["instance_map"]
            # first create a mapping to 1, 2, ... num_instances
            instance_ids = np.unique(instances)
            # map instance id to index
            instance_id_to_idx = {
                instance_id: idx for idx, instance_id in enumerate(instance_ids)
            }
            # convert instance ids to indices, use vectorized lookup
            instances = torch.from_numpy(
                np.vectorize(instance_id_to_idx.get)(instances)
            ).to(self.device)
            # create a one-hot encoding
            instances = torch.eye(len(instance_ids), device=self.device)[instances]

            obs_preprocessed = torch.cat([obs_preprocessed, instances], dim=-1)

        obs_preprocessed = obs_preprocessed.permute(2, 0, 1)

        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_poses), device=rgb.device
        )
        self.last_poses = curr_pose

        # NOT USED AT ALL? ->
        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose))

        # Match a goal against every instance in memory the moment we get it
        # or when the map just got fully explored
        if (
            # task_type in ["languagenav", "imagenav"]
            # and self.record_instance_ids
            # and
            self.sub_task_timesteps[self.current_task_idx] == 0
            or self.navigate_to_best
        ):
            if self.navigate_to_best:
                logger.info("Force a match against the memory")
            all_confidences, instance_ids = self._match_against_memory(
                task_type, current_task
            )

        # * preprocessed obs shape is (1, 3+1+num_sem_classes+num_instances, H, W)
        return (
            obs_preprocessed,
            pose_delta,
            self.goal_image,
            camera_pose,
            confidences,
            frame_matches_local_instance_ids,
            all_confidences,
            instance_ids,
        )

    def _match_against_memory(self, task_type: str, current_task: Dict):
        logger.info("--------Matching against memory!--------")
        if task_type == "languagenav":
            (
                all_confidences,
                instance_ids,
            ) = self.matching.get_matches_against_memory(
                self.matching.match_language_to_image,
                self.total_timesteps,
                language_goal=current_task["description"],
                use_full_image=True,
                categories=[current_task["semantic_id"]],
            )

        elif task_type == "imagenav":
            (
                all_confidences,
                instance_ids,
            ) = self.matching.get_matches_against_memory(
                self.image_matching_function,
                self.sub_task_timesteps[self.current_task_idx],
                image_goal=self.goal_image,
                goal_image_keypoints=self.goal_image_keypoints,
                use_full_image=True,
                categories=[current_task["semantic_id"]],
            )
        elif task_type == "objectnav":
            (
                all_confidences,
                instance_ids,
            ) = self.matching.get_matches_against_memory(
                None,
                self.total_timesteps,
                categories=[current_task["semantic_id"]],
            )

        stats = {
            i: {
                "mean": float(scores.mean()),
                "median": float(np.median(scores)),
                "max": float(scores.max()),
                "min": float(scores.min()),
                "all": scores.flatten().tolist(),
            }
            for i, scores in zip(instance_ids, all_confidences)
        }
        with open(
            f"{self.goal_matching_vis_dir}/goal{self.current_task_idx}_{task_type}_stats.json",
            "w",
        ) as f:
            json.dump(stats, f, indent=4)
        return all_confidences, instance_ids

    def _keep_only_largest_cluster_in_instance_map(self) -> None:
        """
        Perform optional clustering of the goal channel to mitigate noisy projection
        splatter.
        """

        if not self._module.instance_goal_found:
            return

        if not self.goal_filtering:
            return

        logger.debug("Clustering Goal map and selecting the largest cluster.")
        instance_map = self.instance_map
        init_goal_map_count = instance_map.sum()

        # cluster goal points
        c = DBSCAN(eps=4, min_samples=1)
        # * data is index of nonzero elements
        data = np.array(instance_map.nonzero()).T
        c.fit(data)

        # mask all points not in the largest cluster
        mode = scipy.stats.mode(c.labels_, keepdims=False).mode.item()
        mode_mask = (c.labels_ != mode).nonzero()
        x = data[mode_mask]
        goal_map_ = np.copy(instance_map)
        goal_map_[x] = 0.0

        # adopt masked map if non-empty
        if goal_map_.sum() > 0:
            logger.debug("Changing instance map")
            self.instance_map = goal_map_
            logger.debug(
                f"Goal map cells count changed from {init_goal_map_count} to {goal_map_.sum()}"
            )
        else:
            logger.debug(
                "Instance map not changed. Largest cluster is empty for some reason!"
            )

        visualize_map(
            instance_map.shape,
            self.planner.vis_dir,
            f"{self.sub_task_timesteps[self.current_task_idx]+1}_02.cluster_goal.png",
            goal_map=self.instance_map,
            dilated_goal_map=instance_map,
            traversible=1 - self.semantic_map.get_obstacle_map(),
        )
